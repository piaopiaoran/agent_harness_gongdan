import { Client } from "langsmith";
import { config, getSecret, PREFIX } from "../src/config.js";

/**
 * 在线评测健康检查 —— LangSmith Developer 免费计划即可运行，不需要 LLM Gateway。
 *
 * 背景：在线评测失败时，Evaluator Runs / Spend 列表仍然显示 "–" / "$0"，
 * Automation Logs 也可能是空的。错误实际以 feedback comment 的形式挂在根
 * trace 上。这个脚本直接读 feedback_stats，能在密钥失效（401）、judge 模型
 * 参数不兼容（400 / thinking）等情况下第一时间报警。
 *
 * 用法：npm run check:eval -- [检查条数，默认 5]
 */
const limit = Number(process.argv[2] ?? 5);
const key = getSecret("langsmith");
const client = new Client({ apiKey: key, apiUrl: config.langsmith.endpoint });
const api = client.openAPIClient;
const unwrap = (res) => res?.body ?? res;

const KEYS = ["judge_decision", "judge_faithfulness"];

try {
  const projects = unwrap(await api.sessions.listV2({ limit: 50 }));
  const project = (projects?.items ?? []).find((s) => s.name === config.langsmith.project);
  if (!project) {
    console.error(`[eval] 项目 "${config.langsmith.project}" 不存在。`);
    process.exit(1);
  }

  const stats = unwrap(await api.sessions.retrieveV2(project.id, { include_stats: true }));
  console.log(`[eval] 项目 ${stats.name} · 根 trace 在线评测状态`);
  console.log(`[eval] 控制台 https://smith.langchain.com/o/${stats.tenant_id}/projects/p/${project.id}`);
  console.log("");

  // 末尾空格是 LangSmith trace_filter 的 tag 精确匹配写法，不能去掉。
  const runs = unwrap(await api.runs.queryV2({
    project_ids: [project.id],
    is_root: true,
    trace_filter: `tags:${PREFIX} `,
    limit,
    selects: ["ID", "NAME", "START_TIME", "TAGS"],
  }));
  // queryV2 对 limit 有自己的下限，这里再切一次，保证输出条数可控。
  const items = (runs?.items ?? []).slice(0, limit);
  if (!items.length) {
    console.error(`[eval] 没找到带 tag "${PREFIX}" 的根 trace。`);
    process.exit(1);
  }

  const details = [];
  for (const run of items) {
    const detail = await client.readRun(run.id, { includeFeedback: true });
    const fb = detail.feedback_stats ?? {};
    details.push({ run, fb });
    const parts = KEYS.map((k) => {
      const v = fb[k];
      return v ? `${k}: n=${v.n} errors=${v.errors} avg=${v.avg ?? "-"}` : `${k}: -`;
    });
    console.log(`${run.start_time}  ${run.id.slice(0, 8)}  ${parts.join(" | ")}`);
  }

  // 只按真正最新的一条判定，避免用历史成功掩盖当前失败。
  const latest = details[0];
  if (!KEYS.some((k) => latest.fb[k])) {
    console.log("\n[eval] 最新 trace 还没有 feedback（写入通常延迟 2–3 分钟），稍后重跑。");
    process.exit(0);
  }

  const errors = KEYS.flatMap((k) => {
    const v = latest.fb[k];
    if (!v || !v.errors) return [];
    return [`${k}: ${(v.comments?.[0] ?? "无错误详情").slice(0, 200)}`];
  });

  if (errors.length) {
    console.error("\n[eval] 最新一条存在评测失败：");
    for (const e of errors) console.error("  " + e);
    console.error("\n常见原因：workspace secret 过期（401）、judge 模型参数不兼容（400 / thinking）。");
    console.error("检查顺序：Settings → Secrets 里的 provider key → evaluator Model Configuration 的 Extra Parameters。");
    process.exitCode = 1;
  } else {
    console.log("\n[eval] 在线评测正常（免费 Developer 计划即可，无需 LLM Gateway / Plus）。");
  }
} catch (err) {
  console.error(`[eval] 校验失败: HTTP ${err.status ?? "?"} ${err.message}`);
  process.exitCode = 1;
}
