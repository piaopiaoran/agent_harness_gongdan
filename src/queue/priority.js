import { randomUUID } from "node:crypto";
import Redis from "ioredis";
import { config } from "../config.js";

/**
 * BullMQ 风格的可靠优先级队列（Redis ZSET 实现，避免额外 worker 进程依赖）。
 *
 * 可靠性要点（对比旧的 zpopmin 先弹后跑 = worker 崩了任务就丢）：
 *   1. claim 用 Lua 脚本原子完成「到期重试回迁 → 租约超时回收 → 弹出 → 写 in-flight」；
 *   2. 任务跑成功才 ack，从 in-flight 移除；失败按 attempts 指数 backoff 重入 delayed；
 *   3. 超过 maxAttempts 进 dead-letter，不再无限重试；
 *   4. jobId 去重，避免同一批消息重复入队。
 * Redis 不可用时降级进程内数组，保持相同语义（fail-open，保证离线 demo 链路不断）。
 */

// KEYS: pending, inflight, delayed, dead  ARGV: now, leaseMs, maxAttempts
const CLAIM_LUA = `
local pending, inflight, delayed, dead = KEYS[1], KEYS[2], KEYS[3], KEYS[4]
local now = tonumber(ARGV[1])
local lease = tonumber(ARGV[2])
local maxAttempts = tonumber(ARGV[3])

-- 到期的重试任务回迁主队列；优先级放在 payload._priority。
local due = redis.call('ZRANGEBYSCORE', delayed, '-inf', now, 'LIMIT', 0, 50)
for _, member in ipairs(due) do
  redis.call('ZREM', delayed, member)
  local ok, task = pcall(cjson.decode, member)
  if ok and task then
    redis.call('ZADD', pending, tonumber(task._priority or 10), member)
  end
end

-- 租约超时：worker 崩溃遗留的 in-flight 任务重新入队（失败次数由 fail 路径计数）。
local stale = redis.call('ZRANGEBYSCORE', inflight, '-inf', now - lease, 'LIMIT', 0, 50)
for _, member in ipairs(stale) do
  redis.call('ZREM', inflight, member)
  local ok, task = pcall(cjson.decode, member)
  if ok and task then
    redis.call('ZADD', pending, tonumber(task._priority or 10), member)
  end
end

local popped = redis.call('ZPOPMIN', pending, 1)
if #popped < 2 then return nil end
local member = popped[1]
redis.call('ZADD', inflight, now, member)
return member
`;

// KEYS: inflight, delayed, dead  ARGV: member, attempts, maxAttempts, backoffMs, now
const FAIL_LUA = `
local inflight, delayed, dead = KEYS[1], KEYS[2], KEYS[3]
local member = ARGV[1]
local attempts = tonumber(ARGV[2])
local maxAttempts = tonumber(ARGV[3])
local backoff = tonumber(ARGV[4])
local now = tonumber(ARGV[5])
redis.call('ZREM', inflight, member)
local ok, task = pcall(cjson.decode, member)
if not ok or not task then return 'discard' end
task.attempts = attempts
task._lastErrorAt = now
local encoded = cjson.encode(task)
if attempts >= maxAttempts then
  redis.call('ZADD', dead, now, encoded)
  return 'dead'
end
local delay = backoff * (2 ^ math.max(0, attempts - 1))
redis.call('ZADD', delayed, now + delay, encoded)
return 'retry'
`;

export function createPriorityQueue({
  url = config.REDIS_URL,
  concurrency = config.QUEUE_CONCURRENCY,
  startWorker = true,
  maxAttempts = config.queue.maxAttempts,
  backoffMs = config.queue.backoffMs,
  claimLeaseMs = config.queue.claimLeaseMs,
  forceMemory = false, // 单测可强制内存后端，确定性地验证 ack / retry / dead-letter
} = {}) {
  const KEY = "agent_harness_gongdan:priority_queue";
  const k = (suffix) => `${KEY}:${suffix}`;
  const PENDING = k("pending");
  const INFLIGHT = k("inflight");
  const DELAYED = k("delayed");
  const DEAD = k("dead");
  const SEEN = k("seen");

  const handlers = new Map();
  let local = [];
  let localDelayed = [];
  let localInflight = new Map();
  let localDead = [];
  let localSeen = new Set();
  let redis = null;
  let degraded = false;
  let running = false;
  let loops = [];
  let stats = { claimed: 0, acked: 0, retried: 0, dead: 0, recovered: 0, duplicate: 0 };

  if (forceMemory) {
    degraded = true;
  } else {
    try {
      redis = new Redis(url, { maxRetriesPerRequest: 1, retryStrategy: (n) => Math.min(n * 200, 2000) });
      redis.on("error", () => {
        if (!degraded) { degraded = true; console.warn("[queue] Redis 不可用，降级进程内可靠队列"); }
      });
    } catch {
      degraded = true;
    }
  }

  const encode = (task) => JSON.stringify(task);

  async function enqueue(type, payload, priority = 10, jobId = null) {
    const task = {
      id: randomUUID(), jobId: jobId ?? null, type, payload, priority,
      attempts: 0, enqueuedAt: Date.now(), _priority: priority,
    };
    const member = encode(task);
    if (redis && !degraded) {
      try {
        if (jobId) {
          const added = await redis.sadd(SEEN, String(jobId));
          if (!added) { stats.duplicate += 1; return { task, duplicate: true }; }
          await redis.expire(SEEN, 7 * 24 * 3600);
        }
        await redis.zadd(PENDING, priority, member);
        return { task, duplicate: false };
      } catch { degraded = true; }
    }
    if (jobId) {
      if (localSeen.has(String(jobId))) { stats.duplicate += 1; return { task, duplicate: true }; }
      localSeen.add(String(jobId));
    }
    local.push({ task, member, score: priority });
    local.sort((a, b) => a.score - b.score);
    return { task, duplicate: false };
  }

  /** 原子领取：到期重试回迁 + 租约超时回收 + 弹出 + 写 in-flight。 */
  async function dequeue() {
    const now = Date.now();
    if (redis && !degraded) {
      try {
        const member = await redis.eval(
          CLAIM_LUA, 4, PENDING, INFLIGHT, DELAYED, DEAD,
          String(now), String(claimLeaseMs), String(maxAttempts)
        );
        if (!member) return null;
        stats.claimed += 1;
        return { member: String(member), task: JSON.parse(String(member)) };
      } catch { degraded = true; }
    }
    // 本地降级：与 Lua 相同语义
    const due = localDelayed.filter((x) => x.score <= now);
    for (const x of due) {
      localDelayed = localDelayed.filter((y) => y !== x);
      local.push({ ...x, score: x.task._priority ?? 10 });
      stats.retried += 1;
    }
    for (const [member, x] of [...localInflight]) {
      if (now - x.claimedAt > claimLeaseMs) {
        localInflight.delete(member);
        local.push({ ...x, score: x.task._priority ?? 10 });
        stats.recovered += 1;
      }
    }
    local.sort((a, b) => a.score - b.score);
    const item = local.shift();
    if (!item) return null;
    localInflight.set(item.member, { ...item, claimedAt: now });
    stats.claimed += 1;
    return item;
  }

  async function ack(item) {
    if (!item) return;
    if (redis && !degraded) {
      try { await redis.zrem(INFLIGHT, item.member); stats.acked += 1; return; } catch { degraded = true; }
    }
    localInflight.delete(item.member);
    stats.acked += 1;
  }

  async function fail(item, err) {
    if (!item) return;
    const attempts = Number(item.task?.attempts ?? 0) + 1;
    const now = Date.now();
    const reason = String(err?.message ?? err ?? "task_failed").slice(0, 300);
    if (redis && !degraded) {
      try {
        const result = await redis.eval(
          FAIL_LUA, 3, INFLIGHT, DELAYED, DEAD,
          String(item.member), String(attempts), String(maxAttempts), String(backoffMs), String(now)
        );
        if (result === "dead") stats.dead += 1; else stats.retried += 1;
        return result;
      } catch { degraded = true; }
    }
    localInflight.delete(item.member);
    const task = { ...item.task, attempts, _lastError: reason, _lastErrorAt: now };
    const member = encode(task);
    if (attempts >= maxAttempts) { localDead.push({ task, member, score: now }); stats.dead += 1; return "dead"; }
    const delay = backoffMs * 2 ** Math.max(0, attempts - 1);
    localDelayed.push({ task, member, score: now + delay });
    stats.retried += 1;
    return "retry";
  }

  async function loop(workerId) {
    while (running) {
      try {
        const item = await dequeue();
        if (!item) { await new Promise((r) => setTimeout(r, 300)); continue; }
        const handler = handlers.get(item.task.type);
        if (!handler) {
          console.warn(`[queue#${workerId}] 无 handler: ${item.task.type}`);
          await ack(item); // 无 handler 的任务直接丢弃，避免 poison pill 反复重试
          continue;
        }
        try {
          await handler(item.task.payload, item.task);
          await ack(item);
        } catch (err) {
          const result = await fail(item, err);
          console.warn(`[queue#${workerId}] task failed (${result}, attempts=${item.task.attempts + 1}):`, err.message);
        }
      } catch (err) {
        console.warn(`[queue#${workerId}] loop error:`, err.message);
      }
    }
  }

  const api = {
    backend: () => (degraded ? "memory" : "redis"),
    register(type, handler) { handlers.set(type, handler); return api; },
    enqueue,
    dequeue,
    ack,
    fail,
    async size() {
      if (redis && !degraded) {
        try {
          const [p, d, i] = await Promise.all([redis.zcard(PENDING), redis.zcard(DELAYED), redis.zcard(INFLIGHT)]);
          return p + d + i;
        } catch { degraded = true; }
      }
      return local.length + localDelayed.length + localInflight.size;
    },
    async deadLetters(limit = 20) {
      if (redis && !degraded) {
        try { return (await redis.zrevrange(DEAD, 0, limit - 1)).map((m) => JSON.parse(m)); } catch { degraded = true; }
      }
      return localDead.slice(-limit).map((x) => x.task);
    },
    stats: () => ({ ...stats }),
    start() {
      if (running) return api;
      running = true;
      loops = Array.from({ length: concurrency }, (_, i) => loop(i));
      return api;
    },
    async stop() {
      running = false;
      await Promise.all(loops);
      loops = [];
      if (redis) await redis.quit().catch(() => {});
    },
  };

  if (startWorker) api.start();
  return api;
}
