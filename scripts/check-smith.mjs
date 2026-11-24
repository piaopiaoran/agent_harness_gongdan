import { Client } from "langsmith";
import { config, getSecret } from "../src/config.js";

/**
 * LangSmith 连通性自检。
 *
 * 注意：必须用 v2 端点。v1 的 /api/v1/sessions 已废弃，
 * 对同一把有效 key 会返回误导性的 401 Invalid token。
 * 这里用官方 SDK 的 sessions.listV2 / retrieveV2 / runs.queryV2，
 * 既验证 key，也直接证明 trace 真的写进了项目。
 */
const key = getSecret("langsmith");
const client = new Client({ apiKey: key, apiUrl: config.langsmith.endpoint });
const api = client.openAPIClient;

function unwrap(res) {
  return res?.body ?? res;
}

try {
  const projects = unwrap(await api.sessions.listV2({ limit: 50 }));
  const items = projects?.items ?? [];
  const project = items.find((s) => s.name === config.langsmith.project);

  if (!project) {
    console.error(`[smith] key 有效，但项目 "${config.langsmith.project}" 不存在。`);
    console.error("[smith] 可见项目：" + items.map((s) => s.name).join(", "));
    process.exitCode = 1;
  } else {
    const stats = unwrap(await api.sessions.retrieveV2(project.id, { include_stats: true }));
    console.log(`[smith] key 有效 · 项目 ${stats.name}`);
    console.log(`[smith] project_id  ${stats.id}`);
    console.log(`[smith] runs ${stats.run_count ?? 0} · tokens ${stats.total_tokens ?? 0} · errors ${stats.error_rate ?? 0}`);
    console.log(`[smith] last run   ${stats.last_run_start_time ?? "n/a"}`);
    console.log(`[smith] console    https://smith.langchain.com/o/${stats.tenant_id}/projects/p/${stats.id}`);

    const runs = unwrap(await api.runs.queryV2({
      project_ids: [project.id],
      limit: 20,
      is_root: true,
      selects: ["ID", "NAME", "TRACE_ID", "START_TIME", "TAGS"],
    }));
    const all = runs?.items ?? [];
    // 优先给出带本项目 tag 的完整 Agent trace，而不是裸 LLM 调用。
    const run = all.find((r) => r.tags?.includes("agent_harness_gongdan")) ?? all[0];
    if (run?.trace_id) {
      const link = unwrap(await api.runs.getURL(run.id, {
        project_id: project.id,
        trace_id: run.trace_id,
      }));
      console.log(`[smith] latest     ${run.name} @ ${run.start_time}`);
      console.log(`[smith] trace      ${link?.url ?? "n/a"}`);
    }
  }
} catch (err) {
  console.error(`[smith] 校验失败: HTTP ${err.status ?? "?"} ${err.message}`);
  process.exitCode = 1;
}
