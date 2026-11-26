import { z } from "zod";

/**
 * 事实整合决策（Mem0 风格 ADD / UPDATE / DELETE / NOOP）。
 *
 * 纯函数 + zod schema，方便单测与「换模型不改口径」。真实项目里决策可以交给
 * deepseek-flash（DISTILL_CONSOLIDATE_LLM=true）；这里保留阈值启发式作为离线 / 降级兜底。
 * 生产对照：Mem0 2504.19413 用 tool-call 做四操作，余弦阈值≈0.82；
 * Zep/Graphiti 用双时态标 SUPERSEDED；AgentCore 标 INVALID。
 */

export const FACT_OPS = Object.freeze(["ADD", "UPDATE", "DELETE", "NOOP"]);

export const ConsolidationDecision = z.object({
  op: z.enum(FACT_OPS),
  targetId: z.string().nullable().optional(),
  reason: z.string().max(200).optional().default(""),
});

/** 解析 LLM 决策：只认合法 JSON 与白名单操作，非法输入一律降级为 ADD（fail-open 到新事实）。 */
export function parseDecision(text, { fallback = "ADD" } = {}) {
  const raw = String(text ?? "");
  const match = raw.match(/\{[\s\S]*\}/);
  if (!match) return { op: fallback, reason: "decision_not_json" };
  try {
    const parsed = ConsolidationDecision.parse(JSON.parse(match[0]));
    return { op: parsed.op, targetId: parsed.targetId ?? null, reason: parsed.reason ?? "" };
  } catch {
    return { op: fallback, reason: "decision_invalid_schema" };
  }
}

/**
 * 阈值启发式：没有相似事实 → ADD；相似度极高且文本几乎一致 → NOOP；
 * 相似但不完全一致 → UPDATE（旧行标 SUPERSEDED，不删）。DELETE 留给 LLM 判定矛盾。
 */
export function decideOperation({ fact, similar = [], threshold = 0.82, decision = null }) {
  const normalized = String(fact ?? "").trim();
  const best = similar.reduce((a, b) => ((b.similarity ?? 0) > (a?.similarity ?? 0) ? b : a), null);
  const similarity = Number(best?.similarity ?? 0);
  const base = { fact: normalized, similarity, targetId: best?.id ?? null, reason: "" };

  if (decision) {
    const parsed = decision.op ? decision : parseDecision(decision.text);
    if (FACT_OPS.includes(parsed.op)) {
      const targetId = parsed.targetId && String(parsed.targetId) !== "$(id)"
        ? String(parsed.targetId)
        : best?.id ?? null;
      if (targetId) return { ...base, op: parsed.op, targetId, reason: parsed.reason ?? "llm_decision" };
      // 没有合法 target：UPDATE / DELETE / NOOP 无处作用，退化为 ADD。
      return { ...base, op: "ADD", targetId: null, reason: "missing_target_fallback_add" };
    }
  }

  if (!best || similarity < threshold) return { ...base, op: "ADD", targetId: null, reason: "no_similar_fact" };
  const old = String(best.fact ?? "").trim();
  if (normalized === old) return { ...base, op: "NOOP", targetId: best.id, reason: "exact_duplicate" };
  return { ...base, op: "UPDATE", targetId: best.id, reason: "similar_fact_refresh" };
}

/** 决策 prompt：把候选事实与相似项喂给同一把 deepseek-flash，要求只输出 JSON。 */
export function buildDecisionPrompt({ fact, similar = [] }) {
  const candidates = similar.map((s, i) =>
    `${i + 1}. id=${s.id} sim=${Number(s.similarity ?? 0).toFixed(3)} fact=${s.fact}`
  ).join("\n") || "(无相似事实)";
  return [
    "你是记忆整合器。判断新事实相对已有事实应执行哪个操作，只输出 JSON，不要解释。",
    "规则：完全相同或语义等价→NOOP；补充/修正同一件事→UPDATE（旧事实标 SUPERSEDED）；",
    "明确冲突或用户撤回→DELETE（旧事实标 INVALID）；全新信息→ADD。",
    "输出格式：{\"op\":\"ADD|UPDATE|DELETE|NOOP\",\"targetId\":\"命中的 id 或 null\",\"reason\":\"不超过30字\"}",
    `新事实：${fact}`,
    `已有相似事实：\n${candidates}`,
  ].join("\n");
}
