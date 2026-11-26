import { randomUUID } from "node:crypto";
import { config } from "../config.js";
import { listTenants, archiveExpiredMemory } from "../db/queries.js";
import { appendAudit } from "../db/queries.js";

/**
 * 遗忘层：近因衰减（召回时）+ TTL 归档（cron 时）+ 矛盾扫描（整合时已做）。
 *
 * 行业对照：Zep/Graphiti 半衰期≈70 天；Mem0/AgentCore 90/180 天 TTL 归档。
 * 这里只改状态与 valid_to（软归档），原始行保留，审计可追溯；不做物理删除。
 */
export function decayedSimilarity(similarity, createdAt, {
  halfLifeDays = config.memory.halfLifeDays,
  now = Date.now(),
} = {}) {
  const s = Number(similarity ?? 0);
  const ts = Date.parse(createdAt);
  if (!Number.isFinite(ts) || !Number.isFinite(s)) return s;
  const ageDays = Math.max(0, (now - ts) / 86400000);
  return s * 0.5 ** (ageDays / halfLifeDays);
}

/** 按衰减后分数重排并过滤；保留原始 similarity 供观测对比。 */
export function applyForgetting(items, options = {}) {
  const min = options.minSimilarity ?? config.memory.minSimilarity;
  return items
    .map((x) => ({ ...x, rawSimilarity: x.similarity, similarity: decayedSimilarity(x.similarity, x.createdAt, options) }))
    .filter((x) => x.similarity >= min)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, options.limit ?? items.length);
}

/** 定时扫描：枚举租户 → 逐租户归档过期记忆 → 单行审计（node=forgetSweep）。 */
export async function sweepOnce({
  factTtlDays = config.memory.factTtlDays,
  episodeTtlDays = config.memory.episodeTtlDays,
  onAudit = null,
} = {}) {
  const tenants = await listTenants();
  const results = [];
  for (const tenantId of tenants) {
    const r = await archiveExpiredMemory({ tenantId, factTtlDays, episodeTtlDays });
    results.push({ tenantId, ...r });
    const detail = { tenantId, ...r, factTtlDays, episodeTtlDays };
    if (onAudit) {
      try { onAudit(detail); } catch { /* 观测失败不影响归档 */ }
    } else {
      // 审计表 append-only；无 trace 上下文时用合成 trace id。
      await appendAudit({
        traceId: randomUUID(), threadId: "", tenantId, userId: "system",
        node: "forgetSweep", event: "memory_archived", decision: "allow",
        riskFlags: [], detail,
      }).catch((err) => console.warn("[forget] audit failed:", err.message));
    }
  }
  return { tenants: tenants.length, results };
}

/** 进程内定时器（不引入 cron 依赖）；调用方负责在 close() 时 stop。 */
export function createForgetScheduler({
  intervalMs = config.memory.sweepIntervalMs,
  sweep = sweepOnce,
  autoStart = true,
} = {}) {
  let timer = null;
  let last = null;
  const run = async () => {
    try {
      last = { at: new Date().toISOString(), ...(await sweep()) };
    } catch (err) {
      last = { at: new Date().toISOString(), error: err.message };
      console.warn("[forget] sweep failed:", err.message);
    }
    return last;
  };
  const api = {
    start() { if (!timer) timer = setInterval(run, intervalMs); timer?.unref?.(); return api; },
    stop() { if (timer) clearInterval(timer); timer = null; return api; },
    sweep: run,
    status: () => ({ intervalMs, running: Boolean(timer), last }),
  };
  if (autoStart) api.start();
  return api;
}
