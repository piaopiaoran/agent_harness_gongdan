import { awaitAllCallbacks } from "@langchain/core/callbacks/promises";
import { config, PREFIX } from "../config.js";

/**
 * LangSmith 接入：LangChain / LangGraph Runnable 自动读取 LANGSMITH_* 环境变量，
 * 无需手写 tracer。这里只做三件事：
 *   1. 启动自检（绝不打印 key）
 *   2. 把非敏感标识挂到 trace metadata，和审计表 trace_id 双向定位
 *   3. 退出前 flush，避免后台提交的 trace 被进程结束截断
 *
 * ⚠️ 数据出口：真实 LLM 收到 prompt / 证据 / 工具描述；LangSmith 收到 trace 内容。
 * 本项目入模前做 PII 脱敏；生产需评估数据合规，金融医疗场景应自托管。
 */
export function tracingStatus() {
  return {
    enabled: config.langsmith.enabled,
    project: config.langsmith.project,
    endpoint: config.langsmith.endpoint,
    modelMode: config.modelMode,
    llmEndpoint: config.llmEndpoint,
  };
}

export function logTracingStatus() {
  const s = tracingStatus();
  console.log(`[tracing] LangSmith ${s.enabled ? "ON" : "OFF"} · project=${s.project}`);
  if (s.enabled) {
    console.log("[tracing] trace 会包含 prompt / 工具参数 / 输出；生产请先评估数据合规。");
  }
}

export function traceConfig(identity, extra = {}) {
  return {
    runName: extra.runName ?? PREFIX,
    tags: [PREFIX, `project:${config.langsmith.project}`, `stage:${extra.stage ?? "ticket"}`],
    metadata: {
      traceId: identity.traceId,
      sessionId: identity.sessionId,
      tenantId: identity.tenantId,
      userId: identity.userId,
      role: identity.role,
      modelMode: config.modelMode,
      embeddingMode: config.embeddingMode,
      ...(extra.metadata ?? {}),
    },
  };
}

export async function flushTraces() {
  if (!config.langsmith.enabled) return;
  try {
    await awaitAllCallbacks();
  } catch (err) {
    console.error("[tracing] flush failed:", err.message);
  }
}
