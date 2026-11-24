import { Annotation, MessagesAnnotation } from "@langchain/langgraph";

const last = (initial) =>
  Annotation({
    reducer: (prev, next) => (next === undefined ? prev : next),
    default: () => initial,
  });

/**
 * State 全部是可 JSON 序列化的纯数据（要写进 Postgres checkpoint）。
 * 身份来自可信会话，不来自模型输出；checkpoint 中不存凭据与未脱敏原始 PII。
 */
export const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,

  identity: last({ traceId: "", sessionId: "", tenantId: "", userId: "", role: "user" }),
  input: last({ raw: "", text: "", decision: "allow", reason: "", flags: [], injection: [], piiHits: [], truncated: false }),
  memories: last({ episodes: [], facts: [], trace: [] }),
  rag: last({ evidence: [], refused: false, reason: "", trace: { stages: [], degradations: [] }, usedTokens: 0 }),
  contextReport: last(null),
  systemText: last(""),
  toolPlan: last([]),
  approval: last(null),
  toolResults: last([]),
  steps: last(0),
  verdict: last(""),
  refusalReason: last(""),
  finalAnswer: last(""),
  distillQueued: last(false),
});
