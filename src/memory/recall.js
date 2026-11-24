import { config } from "../config.js";
import { searchEpisodes, searchFacts } from "../db/queries.js";

/**
 * 在线召回：情景记忆与语义记忆并行检索，各自独立、分区块注入。
 * 严禁把两者混成一个向量库 —— 一个是「发生过什么」，一个是「稳定事实」。
 * 检索结果一律标注来源类型，便于回答时区分依据。
 */
export async function recallMemories(embedder, { tenantId, userId, query }) {
  if (!query) return { episodes: [], facts: [], trace: [] };
  const embedding = await embedder.embedQuery(query);
  const [epRes, factRes] = await Promise.allSettled([
    searchEpisodes({ tenantId, userId, embedding, limit: 5 }),
    searchFacts({ tenantId, userId, embedding, limit: 5 }),
  ]);

  const episodes = (epRes.status === "fulfilled" ? epRes.value : []).map((e) => ({
    sourceType: "episodic",
    id: e.id,
    text: `[情景:${e.goal}] ${e.summary}（结果：${e.outcome}）`,
    similarity: Number(e.similarity ?? 0),
    traceIds: e.source_trace_ids,
    createdAt: e.created_at,
  }));
  const facts = (factRes.status === "fulfilled" ? factRes.value : []).map((f) => ({
    sourceType: "semantic",
    id: f.id,
    text: `[事实:${f.category}] ${f.fact}`,
    similarity: Number(f.similarity ?? 0),
    confidence: f.confidence,
    createdAt: f.created_at,
  }));

  return {
    episodes,
    facts,
    trace: [
      { source: "episodic", ok: epRes.status === "fulfilled", count: episodes.length, reason: epRes.status === "rejected" ? epRes.reason?.message : undefined },
      { source: "semantic", ok: factRes.status === "fulfilled", count: facts.length, reason: factRes.status === "rejected" ? factRes.reason?.message : undefined },
    ],
  };
}

/** 远端摘要：情景记忆的有损压缩带索引，支持回捞原始 trace（此处不做二次喂给事实层）。 */
export function buildRemoteSummary(episodes, maxChars = config.CTX_REMOTE_TOKENS * 2) {
  if (!episodes.length) return { text: "", traceIds: [] };
  const lines = episodes.map((e, i) => `${i + 1}. ${e.text.slice(0, 80)} (${e.id})`);
  let text = "";
  for (const line of lines) {
    if ((text + line).length > maxChars) break;
    text += `${line}\n`;
  }
  return { text: text.trim(), traceIds: episodes.flatMap((e) => e.traceIds ?? []) };
}
