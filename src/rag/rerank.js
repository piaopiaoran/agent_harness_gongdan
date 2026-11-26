import { config, getSecret } from "../config.js";
import { lexicalOverlap } from "./segment.js";

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dedupeByHash(ranked) {
  const seen = new Set();
  return ranked.filter((c) => {
    const key = c.row.content_hash ?? c.row.content;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/**
 * Cross-Encoder 的本地基线替身：sigmoid(相似度 + 词法重合 + 时效) 后按阈值过滤。
 * 这是原来的 rerank()，搬到这里成为 provider 之一，签名保持不变。
 */
export function rerankHeuristic(candidates, queryTokens) {
  try {
    const scored = candidates.map((c) => {
      const similarity = Number(c.row.similarity ?? 0);
      const lexical = lexicalOverlap(queryTokens, c.row.content);
      const ageDays = (Date.now() - new Date(c.row.created_at ?? Date.now()).getTime()) / 86400000;
      const recency = Math.max(0, 1 - ageDays / 365);
      const raw = similarity * 2.2 + lexical * 1.3 + recency * 0.3 - 0.55;
      return {
        ...c, rerankScore: sigmoid(raw), similarity, lexical,
        rerankSource: "heuristic",
      };
    });
    const kept = scored.filter((c) => c.rerankScore >= config.RAG_RERANK_THRESHOLD);
    return dedupeByHash(kept).sort((a, b) => b.rerankScore - a.rerankScore);
  } catch (err) {
    return { error: err.message, fallback: candidates };
  }
}

/**
 * 真 cross-encoder：SiliconFlow 的 BAAI/bge-reranker-v2-m3。
 * 一次请求把全部候选送过去批量打分（不是逐对 20 次调用），用 results[].index
 * 映射回候选对象；阈值走独立的 RERANK_API_THRESHOLD，不能复用 heuristic 的 0.30。
 *
 * 失败一律抛错，由 rerankCandidates 降级回 heuristic；usage 通过回调上报，
 * 只进审计 trace，不并入主 LLM 的 costLedger（该模型当前免费）。
 */
export async function rerankViaSiliconFlow(candidates, query, {
  onUsage,
  threshold = config.RERANK_API_THRESHOLD,
  endpoint = config.rerankEndpoint,
  apiKey = getSecret("rerank"),
  model = config.RERANK_MODEL,
  timeoutMs = config.RERANK_TIMEOUT_MS,
  maxChars = config.RERANK_MAX_CHARS,
  fetchImpl = globalThis.fetch,
} = {}) {
  const res = await fetchImpl(`${endpoint}/rerank`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${apiKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      query,
      documents: candidates.map((c) => String(c.row.content ?? "").slice(0, maxChars)),
      top_n: candidates.length,
      return_documents: false,
      // 候选已在本地截短，不让服务端再切块（切块会改变分数含义并增加用量）。
      max_chunks_per_doc: 1,
      overlap_tokens: 0,
    }),
    signal: AbortSignal.timeout(timeoutMs),
  });

  if (!res.ok) {
    let detail = "";
    try { detail = (await res.text()).slice(0, 200); } catch { /* 响应体不可读时只看状态码 */ }
    throw new Error(`[rerank] HTTP ${res.status} ${detail}`);
  }

  const json = await res.json();
  if (!Array.isArray(json.results)) {
    throw new Error("[rerank] 响应缺少 results 数组");
  }
  onUsage?.(json.meta?.tokens ?? json.tokens ?? null);

  const scored = [];
  for (const r of json.results) {
    const c = candidates[Number(r.index)];
    const score = Number(r.relevance_score ?? r.score);
    if (!c || !Number.isFinite(score)) continue;
    scored.push({ ...c, rerankScore: score, rerankSource: "cross-encoder" });
  }
  // BGE 分数分布随语料与 query 变化，绝对阈值可能把候选全部滤掉；
  // 此时保留 top1 作为保底，避免「精排成功但证据为空」的静默降质。
  const sorted = dedupeByHash(scored).sort((a, b) => b.rerankScore - a.rerankScore);
  const kept = sorted.filter((c) => c.rerankScore >= threshold);
  return kept.length ? kept : sorted.slice(0, 1);
}

/**
 * 统一入口。始终先算 heuristic 基线：
 *  - provider=heuristic → 直接用基线（降级 / 离线兜底，零外部调用）
 *  - provider=siliconflow + shadow=false → 用 BGE 分数排序（主路默认）
 *  - provider=siliconflow + shadow=true → 双跑，仍用基线排序，BGE 分数只写进审计
 *  - API 异常 → 降级回基线；基线自身异常由 retrieve.js 继续降级回 RRF
 */
export async function rerankCandidates({
  candidates, query, queryTokens, embedder, queryEmbedding, onUsage, options = {},
}) {
  const provider = options.provider ?? config.RERANK_PROVIDER;
  const shadow = options.shadow ?? (config.RERANK_SHADOW === "true");
  const out = {
    ranked: rerankHeuristic(candidates, queryTokens, embedder, queryEmbedding),
    provider: "heuristic",
    model: null,
    latencyMs: 0,
    usage: null,
    shadow: null,
    error: null,
  };
  if (provider !== "siliconflow") return out;

  out.model = options.model ?? config.RERANK_MODEL;
  const started = Date.now();
  try {
    const apiRanked = await rerankViaSiliconFlow(candidates, query, {
      ...options,
      onUsage: (u) => { out.usage = u; onUsage?.(u); },
    });
    out.latencyMs = Date.now() - started;
    if (shadow) {
      out.shadow = {
        provider: "siliconflow",
        model: out.model,
        kept: apiRanked.length,
        latencyMs: out.latencyMs,
        topScores: apiRanked.slice(0, 5).map((c) => Number(c.rerankScore.toFixed(4))),
        topIds: apiRanked.slice(0, 5).map((c) => c.row.id),
      };
    } else {
      out.ranked = apiRanked;
      out.provider = "siliconflow";
    }
  } catch (err) {
    out.latencyMs = Date.now() - started;
    out.error = err.message;
  }
  return out;
}
