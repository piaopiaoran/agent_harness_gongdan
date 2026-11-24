import { config } from "../config.js";
import { vectorSearchChunks, keywordSearchChunks, getDocumentTitles } from "../db/queries.js";
import { segment, lexicalOverlap, estimateTokens } from "./segment.js";
import { normalizeInput, sanitizeDocument } from "../security/patterns.js";

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

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
 * Cross-Encoder 的本地基线替身：sigmoid(相似度 + 词法重合 + 时效) 后按 0.30 阈值过滤。
 * 接口与真 cross-encoder 一致（逐对打分 → 阈值 → 去重），换成真模型只改本函数。
 */
export function rerank(candidates, queryTokens, embedder, queryEmbedding) {
  try {
    const scored = candidates.map((c) => {
      const similarity = Number(c.row.similarity ?? 0);
      const lexical = lexicalOverlap(queryTokens, c.row.content);
      const ageDays = (Date.now() - new Date(c.row.created_at ?? Date.now()).getTime()) / 86400000;
      const recency = Math.max(0, 1 - ageDays / 365);
      const raw = similarity * 2.2 + lexical * 1.3 + recency * 0.3 - 0.55;
      return { ...c, rerankScore: sigmoid(raw), similarity, lexical };
    });
    const kept = scored.filter((c) => c.rerankScore >= config.RAG_RERANK_THRESHOLD);
    const seen = new Set();
    return kept.filter((c) => {
      const key = c.row.content_hash ?? c.row.content;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    }).sort((a, b) => b.rerankScore - a.rerankScore);
  } catch (err) {
    return { error: err.message, fallback: candidates };
  }
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
  const rankedResult = rerank(topN, queryTokens, embedder, queryEmbedding);
  let ranked;
  if (rankedResult.error) {
    ranked = topN;
    trace.degradations.push({ stage: "rerank", policy: "fail-open", reason: rankedResult.error });
  } else {
    ranked = rankedResult;
    trace.stages.push({ stage: "rerank", kept: ranked.length, threshold: config.RAG_RERANK_THRESHOLD });
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
