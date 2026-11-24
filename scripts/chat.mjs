import { createRuntime, createSession, extractInterrupt } from "../src/runtime.js";
import { flushTraces } from "../src/observability/tracing.js";

const message = process.argv.slice(2).join(" ") || "ORD-1001 左耳没声音，我想退款，可以吗？";
const runtime = await createRuntime();
const sess = createSession(runtime, {
  tenantId: process.env.DEMO_TENANT ?? "tenant-a",
  userId: process.env.DEMO_USER ?? "u-1",
  role: process.env.DEMO_ROLE === "admin" ? "admin" : "user",
});

try {
  const result = await sess.start(message);
  console.log("\n=== 回答 ===\n", result.finalAnswer);
  console.log("\n=== 状态 ===", result.verdict);
  console.log("=== RAG ===", result.rag?.evidence?.map((e) => `${e.id} ${e.source}`));
  console.log("=== 上下文 ===", result.contextReport);
  console.log("=== 审计链 ===");
  for (const e of sess.auditor.events) console.log(`  ${e.node}.${e.event} -> ${e.decision}`, e.detail ? JSON.stringify(e.detail).slice(0, 160) : "");
  const pending = extractInterrupt(result);
  if (pending) {
    console.log("\n=== 等待人工审批 ===", JSON.stringify(pending[0]));
    const resumed = await sess.resume({ decision: "approved", approverId: "ops-admin", reason: "CLI 演示批准" });
    console.log("=== 审批后回答 ===\n", resumed.finalAnswer);
  }
} catch (err) {
  console.error("[chat] 失败:", err.message, "\n", err.stack?.split("\n").slice(0, 12).join("\n"));
  process.exitCode = 1;
} finally {
  await flushTraces();
  await runtime.close();
}
