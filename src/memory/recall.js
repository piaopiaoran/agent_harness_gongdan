import { config } from "../config.js";
import { searchEpisodes, searchFacts } from "../db/queries.js";
import { applyForgetting } from "./forget.js";

/**
 * 在线召回：情景记忆与语义记忆并行检索，各自独立、分区块注入。
 * 严禁把两者混成一个向量库 —— 一个是「发生过什么」，一个是「稳定事实」。
 * 检索结果一律标注来源类型，便于回答时区分依据。
 *
 * 遗忘层：按 created_at 做半衰期衰减（默认 70 天），而不是无脑 top-k；
 * 越久没被强化的记忆分数越低，跑越久也不会把检索槽位占满。
 */
export async function recallMemories(embedder, { tenantId, userId, query, halfLifeDays } = {}) {
  if (!query) return { episodes: [], facts: [], trace: [] };
  const embedding = await embedder.embedQuery(query);
  const [epRes, factRes] = await Promise.allSettled([
    searchEpisodes({ tenantId, userId, embedding, limit: 10 }),
    searchFacts({ tenantId, userId, embedding, limit: 10 }),
  ]);

  const decayOpts = { halfLifeDays: halfLifeDays ?? config.memory.halfLifeDays };
  const episodes = applyForgetting(
    (epRes.status === "fulfilled" ? epRes.value : []).map((e) => ({
      sourceType: "episodic",
      id: e.id,
      text: `[情景:${e.goal}] ${e.summary}（结果：${e.outcome}）`,
      similarity: Number(e.similarity ?? 0),
      traceIds: e.source_trace_ids,
      createdAt: e.created_at,
    })),
    { ...decayOpts, limit: 5 }
  );
  const facts = applyForgetting(
    (factRes.status === "fulfilled" ? factRes.value : []).map((f) => ({
      sourceType: "semantic",
      id: f.id,
      text: `[事实:${f.category}] ${f.fact}`,
      similarity: Number(f.similarity ?? 0),
      confidence: f.confidence,
      sourceKind: f.source_kind,
      createdAt: f.created_at,
    })),
    { ...decayOpts, limit: 5 }
  );

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
