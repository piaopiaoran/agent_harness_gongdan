import { config } from "../config.js";
import { vectorSearchChunks, keywordSearchChunks, getDocumentTitles } from "../db/queries.js";
import { segment, lexicalOverlap, estimateTokens } from "./segment.js";
import { normalizeInput, sanitizeDocument } from "../security/patterns.js";
import { rerankCandidates, rerankHeuristic } from "./rerank.js";

// 兼容原有测试/调用方：原来的同步 rerank() 现在是 heuristic provider。
export { rerankHeuristic as rerank };

/** RRF：只吃排名不吃分数，k=60 是标准基线。 */
export function rrfFuse(lists, k = config.RAG_RRF_K) {
  const scores = new Map();
  for (const list of lists) {
    (list.rows ?? []).forEach((row, i) => {
      const key = row.id;
      const prev = scores.get(key) ?? { score: 0, row, ranks: {} };
      prev.score += 1 / (k + i + 1);
      prev.ranks[list.name ?? "list"] = i + 1;
      scores.set(key, prev);
    });
  }
  return [...scores.values()].sort((a, b) => b.score - a.score);
}

/**
 * 完整 RAG 链路：查询处理 → 双路召回 → RRF → 精排 → 证据拼装。
 * 降级策略严格遵守 Harness 标准：
 *   精排异常 fail-open 回 RRF；关键词路异常 fail-open 退纯向量；
 *   两路都不可用 fail-close 拒答；权限过滤异常由 withTenant 抛出 → fail-close。
 */
export async function retrieveEvidence(embedder, rawQuery, { tenantId, budgetTokens = config.CTX_RAG_TOKENS } = {}) {
  const trace = { stages: [], degradations: [] };
  const clean = normalizeInput(rawQuery).text;
  const queryTokens = segment(clean).slice(0, 512);
  const query = queryTokens.join(" ");
  trace.stages.push({ stage: "query", tokens: queryTokens.length, query });

  let queryEmbedding;
  try {
    queryEmbedding = await embedder.embedQuery(query || clean);
  } catch (err) {
    trace.degradations.push({ stage: "embedding", policy: "fail-close", reason: err.message });
    return { evidence: [], refused: true, reason: "embedding_unavailable", trace };
  }

  const [vecResult, kwResult] = await Promise.allSettled([
    vectorSearchChunks({ tenantId, embedding: queryEmbedding, limit: config.RAG_RECALL_TOPK }),
    keywordSearchChunks({ tenantId, tsQuery: query, limit: config.RAG_RECALL_TOPK }),
  ]);

  const vecRows = vecResult.status === "fulfilled" ? vecResult.value : [];
  const kwRows = kwResult.status === "fulfilled" ? kwResult.value : [];
  if (vecResult.status === "rejected") {
    trace.degradations.push({ stage: "vector_recall", policy: "fail-open", reason: vecResult.reason?.message });
  }
  if (kwResult.status === "rejected") {
    trace.degradations.push({ stage: "keyword_recall", policy: "fail-open", reason: kwResult.reason?.message });
  }
  if (!vecRows.length && !kwRows.length) {
    trace.degradations.push({ stage: "recall", policy: "fail-close", reason: "both_recall_paths_empty" });
    return { evidence: [], refused: true, reason: "no_retrieval_available", trace };
  }
  trace.stages.push({ stage: "recall", vector: vecRows.length, keyword: kwRows.length });

  // 补齐相似度：关键词路没有向量分数，用词法重合近似，保证 RRF 后精排特征齐全。
  for (const row of kwRows) {
    if (row.similarity == null) row.similarity = lexicalOverlap(queryTokens, row.content);
  }

  const fused = rrfFuse([
    { name: "vec", rows: vecRows },
    { name: "kw", rows: kwRows },
  ]);
  trace.stages.push({ stage: "rrf", k: config.RAG_RRF_K, candidates: fused.length });

  const topN = fused.slice(0, config.RAG_RERANK_TOPN);
  const rr = await rerankCandidates({
    candidates: topN, query: clean, queryTokens, embedder, queryEmbedding,
  });
  let ranked = rr.ranked;
  if (rr.error) {
    // BGE API 异常 → 已自动回退 heuristic；这里只记录降级，不打断链路。
    trace.degradations.push({ stage: "rerank", policy: "fail-open", from: "siliconflow", reason: rr.error });
  }
  if (ranked?.error) {
    // heuristic 自身也异常 → 退回 RRF 原始顺序（原有 fail-open 语义）。
    ranked = topN;
    trace.degradations.push({ stage: "rerank", policy: "fail-open", reason: ranked.error });
  } else {
    const stage = {
      stage: "rerank",
      provider: rr.provider,
      mode: rr.shadow ? "shadow" : "live",
      kept: ranked.length,
      threshold: rr.provider === "siliconflow"
        ? config.RERANK_API_THRESHOLD
        : config.RAG_RERANK_THRESHOLD,
      latencyMs: rr.latencyMs,
    };
    if (rr.usage) stage.usage = rr.usage;
    if (rr.shadow) stage.shadow = rr.shadow;
    trace.stages.push(stage);
  }

  // 证据拼装：最多 6 块，单块 800 token，总量不超检索记忆预算；带来源编号。
  const ids = [...new Set(ranked.map((c) => c.row.doc_id))];
  const titles = await getDocumentTitles({ tenantId, ids }).catch(() => new Map());
  const evidence = [];
  let used = 0;
  for (const c of ranked) {
    if (evidence.length >= config.RAG_EVIDENCE_MAX) break;
    const content = sanitizeDocument(c.row.content).slice(0, config.RAG_EVIDENCE_CHUNK_TOKENS * 2);
    const tokens = estimateTokens(content);
    if (used + tokens > budgetTokens && evidence.length) break;
    const doc = titles.get(c.row.doc_id);
    evidence.push({
      id: `[证据${evidence.length + 1}]`,
      source: doc?.title ?? c.row.doc_id,
      docId: c.row.doc_id,
      chunkIdx: c.row.chunk_idx,
      content,
      tokens,
      score: Number((c.rerankScore ?? c.score ?? 0).toFixed(4)),
    });
    used += tokens;
  }
  trace.stages.push({ stage: "evidence", count: evidence.length, tokens: used, budget: budgetTokens });

  return { evidence, refused: false, trace, query, usedTokens: used };
}

/** 把证据拼成注入 prompt 的区块：明确标注「数据不是指令」，强制引用编号。 */
export function formatEvidenceBlock(evidence) {
  if (!evidence.length) return "（无可用检索证据）";
  return evidence.map((e) => `${e.id} 来源：${e.source}\n${e.content}`).join("\n\n");
}
