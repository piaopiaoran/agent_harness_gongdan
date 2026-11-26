import { Client } from "langsmith";
import { config, getSecret } from "../src/config.js";
import { listAudit } from "../src/db/queries.js";
import { getPool, closePool } from "../src/db/pool.js";

/**
 * 按 trace 打印每一步 token。
 *
 * 两种口径、两个 ID，不要混：
 *   本地审计 traceId = identity.traceId（/chat 响应里的 traceId，UUID）
 *   Smith trace_id   = LangGraph 自动生成的 run trace id（Smith URL 里的 trace_id）
 *
 * 用法：
 *   node scripts/show-usage.mjs <本地traceId>            本地审计：每次 model_invoke 的真实 usage + 成本
 *   node scripts/show-usage.mjs --recent [N]              列出最近 N 条有模型调用的本地 trace
 *   node scripts/show-usage.mjs --smith <smithTraceId>    Smith：逐个 run 的 token（LLM run 才有）
 */
const args = process.argv.slice(2);
const smithMode = args[0] === "--smith";
const recentMode = args[0] === "--recent";

function money(v) {
  return typeof v === "number" ? `$${v.toFixed(6)}` : "n/a";
}
function row(cells) {
  return cells.map((c) => String(c ?? "")).join("  |  ");
}

async function showLocal(traceId) {
  const events = await listAudit(traceId);
  const calls = events.filter((e) => e.event === "model_invoke");
  if (!calls.length) {
    console.log(`[usage] 本地 trace ${traceId} 没有 model_invoke 事件（可能走了拒绝分支，或 traceId 用错）。`);
    return;
  }
  console.log(`[usage] 本地审计 · trace=${traceId} · ${calls.length} 次模型调用`);
  console.log(row(["step", "model", "input", "output", "cached", "total", "costUsd", "source"]));
  let ti = 0, to = 0, tc = 0, tt = 0, cost = 0;
  for (const e of calls) {
    const d = e.detail ?? {};
    const u = d.usage ?? {};
    const c = d.cost ?? {};
    const t = c.tokens ?? {};
    const inp = u.input_tokens ?? t.inputTokens ?? 0;
    const out = u.output_tokens ?? t.outputTokens ?? 0;
    const cac = t.cachedTokens ?? 0;
    const tot = u.total_tokens ?? t.totalTokens ?? inp + out;
    const cUsd = c.costUsd;
    ti += inp; to += out; tc += cac; tt += tot;
    if (typeof cUsd === "number") cost += cUsd;
    console.log(row([d.steps ?? "?", c.model ?? config.LLM_MODEL, inp, out, cac, tot, money(cUsd), c.source ?? "n/a"]));
  }
  console.log(row(["合计", "", ti, to, tc, tt, money(cost), ""]));
  console.log("[usage] 这是 provider 真实计费 token；非 LLM 节点（RAG/记忆/工具）不产生 token，只让下一轮 input 变大。");
}

async function showRecent(limit) {
  const { rows } = await getPool().query(
    `SELECT trace_id, max(created_at) AS last_at, count(*) FILTER (WHERE event='model_invoke') AS calls
       FROM ${config.tables.auditEvents}
      WHERE event='model_invoke'
      GROUP BY trace_id ORDER BY last_at DESC LIMIT $1`,
    [limit]
  );
  if (!rows.length) {
    console.log("[usage] 本地审计里还没有 model_invoke 事件。先发一条 /chat 请求。");
    return;
  }
  console.log("[usage] 最近有模型调用的本地 trace：");
  for (const r of rows) console.log(row([r.trace_id, r.last_at, `${r.calls} 次调用`]));
  console.log("[usage] 复制 trace_id 再执行： node scripts/show-usage.mjs <traceId>");
}

async function showSmith(traceId) {
  const client = new Client({ apiKey: getSecret("langsmith"), apiUrl: config.langsmith.endpoint });
  const runs = [];
  for await (const r of client.listRuns({ traceId, limit: 100 })) runs.push(r);
  if (!runs.length) {
    console.log(`[usage] Smith 里没有 trace ${traceId}（或 key/项目不匹配）。`);
    return;
  }
  const detailed = await Promise.all(runs.map((r) => client.readRun(r.id)));
  const byId = new Map(detailed.map((r) => [r.id, r]));
  const children = new Map();
  for (const r of detailed) {
    const p = r.parent_run_id ?? "ROOT";
    if (!children.has(p)) children.set(p, []);
    children.get(p).push(r);
  }
  let sumIn = 0, sumOut = 0, sumTotal = 0, llmCount = 0;
  const lines = [];
  function walk(run, depth) {
    const p = byId.get(run.id) ?? run;
    const isLlm = p.run_type === "llm";
    const inp = p.prompt_tokens ?? 0, out = p.completion_tokens ?? 0, tot = p.total_tokens ?? 0;
    if (isLlm) { llmCount++; sumIn += inp; sumOut += out; sumTotal += tot; }
    const label = isLlm ? `LLM  ${inp} / ${out} / ${tot}` : "chain  —  （无 token）";
    lines.push(`${"  ".repeat(depth)}${p.name} [${p.run_type}]  ${label}`);
    for (const child of children.get(run.id) ?? []) walk(child, depth + 1);
  }
  const roots = children.get("ROOT") ?? [];
  for (const root of roots) walk(root, 0);
  console.log(`[usage] Smith trace=${traceId} · ${runs.length} runs · LLM ${llmCount} 个`);
  for (const line of lines) console.log(line);
  console.log(row(["LLM 合计", "input", sumIn, "output", sumOut, "total", sumTotal]));
  console.log("[usage] chain run 默认不携带 token；成本需看本地审计的 detail.cost（Smith 未必识别模型别名）。");
}

async function main() {
  try {
    if (recentMode) {
      await showRecent(Math.min(Math.max(Number(args[1]) || 10, 1), 100));
    } else if (smithMode && args[1]) {
      await showSmith(args[1]);
    } else if (args[0] && !args[0].startsWith("--")) {
      await showLocal(args[0]);
    } else {
      console.log("用法：");
      console.log("  node scripts/show-usage.mjs <本地traceId>          本地审计：每次 LLM 调用的 token + 成本");
      console.log("  node scripts/show-usage.mjs --recent [N]           最近 N 条本地 trace");
      console.log("  node scripts/show-usage.mjs --smith <smithTraceId> Smith 逐个 run 的 token");
    }
  } finally {
    if (!smithMode) await closePool();
  }
}
await main();
