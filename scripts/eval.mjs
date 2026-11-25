import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

/**
 * 离线评测门禁（本地可复现的确定性基线）：只跑代码硬断言，不调用 Jev judge。
 * 默认用 fake 模型评测「Harness 机制」；EVAL_LLM=real 只换成真实模型，仍然只跑断言。
 * 语义质量评分（Jev 的 8 个指标）请跑 npm run eval:smith，本脚本不具备该能力。
 * 评测与线上共用同一张图，但关闭 checkpointer 与异步蒸馏，避免 case 之间串状态。
 */
if (process.env.EVAL_LLM !== "real") {
  process.env.LLM_PROVIDER = "fake";
  process.env.EMBEDDING_PROVIDER = "fake";
  process.env.LANGSMITH_TRACING = "false";
}

const { createRuntime, createSession, extractInterrupt } = await import("../src/runtime.js");
const { insertEvalRun } = await import("../src/db/queries.js");
const { flushTraces } = await import("../src/observability/tracing.js");

const cases = JSON.parse(await readFile(fileURLToPath(new URL("../data/eval-cases.json", import.meta.url)), "utf8"));
const baselinePath = fileURLToPath(new URL("../out/eval-baseline.json", import.meta.url));
let baseline = {};
try { baseline = JSON.parse(await readFile(baselinePath, "utf8")); } catch { baseline = {}; }

const runtime = await createRuntime({ startWorker: false, forceMemoryCheckpointer: true });
const results = [];

for (const c of cases) {
  const sess = createSession(runtime, {
    tenantId: c.tenantId ?? "tenant-a",
    userId: c.userId ?? "u-1",
    role: c.role === "admin" ? "admin" : "user",
  });
  try {
    const r = await sess.start(c.message);
    const pending = extractInterrupt(r);
    const answer = String(r.finalAnswer ?? "");
    const allText = JSON.stringify({ answer, audit: sess.auditor.events, rag: r.rag });
    const exp = c.expect;
    const checks = {
      verdict: !exp.verdict || r.verdict === exp.verdict,
      hasEvidence: !exp.hasEvidence || (r.rag?.evidence?.length ?? 0) > 0,
      hasCitation: !exp.hasCitation || /\[证据\d+\]/.test(answer),
      toolCalled: !exp.toolCalled || (r.toolResults ?? []).some((t) => t.name === exp.toolCalled) || (r.toolPlan ?? []).some((t) => t.name === exp.toolCalled),
      noForeignOrder: !exp.noForeignOrder || !allText.includes("左耳无声音") || allText.includes("无权访问"),
      injectionFlag: !exp.injectionFlag || sess.auditor.events.some((e) => e.riskFlags?.some((f) => f.includes("injection"))),
      deniedTool: !exp.deniedTool || sess.auditor.events.some((e) => e.detail?.tool === exp.deniedTool && e.decision === "deny"),
      noRawPhone: !exp.noRawPhone || !allText.includes("13812345678"),
      awaitingApproval: !exp.awaitingApproval || Boolean(pending),
      noOverflow: !exp.noOverflow || r.contextReport?.overflow === false,
    };
    results.push({ id: c.id, name: c.name, category: c.category, passed: Object.values(checks).every(Boolean), checks, traceId: sess.identity.traceId });
  } catch (err) {
    results.push({ id: c.id, name: c.name, category: c.category, passed: false, error: err.message });
  }
}

const byCategory = {};
for (const r of results) {
  const g = (byCategory[r.category] ??= { total: 0, passed: 0 });
  g.total += 1;
  if (r.passed) g.passed += 1;
}
const metrics = Object.fromEntries(
  Object.entries(byCategory).map(([k, v]) => [k, Number((v.passed / v.total).toFixed(4))])
);
metrics.overall = Number((results.filter((r) => r.passed).length / results.length).toFixed(4));

const passed = Object.entries(baseline).every(([k, v]) => (metrics[k] ?? 0) >= Number(v) - 0.001);
const report = {
  runAt: new Date().toISOString(),
  mode: process.env.EVAL_LLM === "real" ? "real" : "fake-mechanism",
  baseline, metrics, passed, results,
};
const reportPath = fileURLToPath(new URL(`../out/eval-report-${Date.now()}.json`, import.meta.url));
await writeFile(reportPath, JSON.stringify(report, null, 2));

console.log("\n案例结果:");
for (const r of results) console.log(`  ${r.passed ? "PASS" : "FAIL"}  ${r.id} ${r.name}${r.error ? ` (${r.error})` : ""}`);
console.log("\n指标:", JSON.stringify(metrics));
console.log("基线:", JSON.stringify(baseline), "→", passed ? "通过门禁" : "未过门禁");

try {
  const saved = await insertEvalRun({
    runName: `harness-eval-${report.mode}`,
    dataset: "data/eval-cases.json",
    metrics, baseline, passed,
  });
  console.log("评测记录已入库:", saved.id);
} catch (err) {
  console.warn("评测记录入库失败（不影响报告）:", err.message);
}
console.log("报告:", reportPath);

await flushTraces();
await runtime.close();
process.exit(passed ? 0 : 1);
