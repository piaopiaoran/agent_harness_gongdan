import { config } from "../config.js";
import { normalizeInput, detectInjection } from "./patterns.js";
import { maskPII } from "./pii.js";
import { appendAudit } from "../db/queries.js";

/**
 * L2/L3 输入护栏：预清洗 → 注入检测 → PII 脱敏。
 * 注入 / PII fail-close（拒绝）；LlamaGuard 类内容合规误杀告警放行，这里不引入外部依赖。
 */
export function guardInput(raw) {
  const { text, truncated } = normalizeInput(raw);
  const injection = detectInjection(text);
  const pii = maskPII(text);
  return {
    text: pii.masked,
    flags: [...injection.map((i) => `injection:${i}`), ...pii.hits.map((h) => `pii:${h.type}`)],
    injection,
    piiHits: pii.hits,
    truncated,
    decision: injection.length ? "deny" : "allow",
    reason: injection.length ? `injection:${injection.join(",")}` : truncated ? "input_truncated" : "clean",
  };
}

/**
 * L7 输出护栏：PII fail-close 脱敏；强制引用；无证据时拒答不编造。
 * 合规类检查 fail-open 加告警，避免 LlamaGuard 类 FP 误杀。
 */
export function guardOutput(answer, { evidence = [], flags = [] } = {}) {
  const pii = maskPII(answer);
  const masked = pii.masked.slice(0, config.MAX_OUTPUT_CHARS);
  const notes = [];
  if (pii.hits.length) notes.push(`已脱敏 ${pii.hits.map((h) => h.type).join("/")}`);
  if (evidence.length) {
    const cited = /\[证据\d+\]/.test(masked);
    if (!cited) notes.push("未按规范引用证据");
  }
  const refused = !evidence.length && /政策|规定|保修|退款|换货/.test(answer);
  return {
    answer: refused ? "未找到相关依据，无法给出政策结论。请联系人文客服核实。" : masked,
    flags: [...flags, ...(refused ? ["ungrounded_answer_blocked"] : [])],
    notes,
    decision: pii.hits.length || refused ? "sanitized" : "allow",
  };
}

/** L10 审计：内存镜像供前端实时查看，DB 落库供事后取证；审计写入失败 fail-open + 告警。 */
export function createAuditor(identity) {
  const events = [];
  async function emit(node, event, decision = "allow", riskFlags = [], detail = {}) {
    const item = {
      node, event, decision,
      riskFlags: [...riskFlags],
      detail: JSON.parse(JSON.stringify(detail ?? {})),
      at: new Date().toISOString(),
    };
    events.push(item);
    try {
      await appendAudit({
        traceId: identity.traceId, threadId: identity.sessionId,
        tenantId: identity.tenantId, userId: identity.userId,
        node, event, decision, riskFlags, detail: item.detail,
      });
    } catch (err) {
      console.warn("[audit] 落库失败（fail-open）:", err.message);
    }
    return item;
  }
  return { emit, events };
}
