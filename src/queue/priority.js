import { randomUUID } from "node:crypto";
import Redis from "ioredis";
import { config } from "../config.js";

/**
 * BullMQ 风格的优先级队列（Redis ZSET 实现，避免额外 worker 进程依赖）。
 * 分数越小优先级越高：交互任务 0，批量 / 蒸馏任务 10+。
 * Redis 不可用时降级进程内数组，保证离线 demo 链路不断（fail-open）。
 */
export function createPriorityQueue({ url = config.REDIS_URL, concurrency = config.QUEUE_CONCURRENCY, startWorker = true } = {}) {
  const KEY = "agent_harness_gongdan:priority_queue";
  const handlers = new Map();
  const local = [];
  let redis = null;
  let degraded = false;
  let running = false;
  let loops = [];

  try {
    redis = new Redis(url, { maxRetriesPerRequest: 1, retryStrategy: (n) => Math.min(n * 200, 2000) });
    redis.on("error", () => {
      if (!degraded) { degraded = true; console.warn("[queue] Redis 不可用，降级进程内队列"); }
    });
  } catch {
    degraded = true;
  }

  async function enqueue(type, payload, priority = 10) {
    const task = { id: randomUUID(), type, payload, priority, enqueuedAt: Date.now() };
    if (redis && !degraded) {
      try {
        await redis.zadd(KEY, priority, JSON.stringify(task));
        return task;
      } catch { degraded = true; }
    }
    local.push(task);
    local.sort((a, b) => a.priority - b.priority);
    return task;
  }

  async function dequeue() {
    if (redis && !degraded) {
      try {
        // ioredis 返回顺序是 [member, score]，成员才是任务 JSON。
        const out = await redis.zpopmin(KEY, 1);
        if (out?.length >= 2) return JSON.parse(out[0]);
        return null;
      } catch { degraded = true; }
    }
    return local.shift() ?? null;
  }

  async function loop(workerId) {
    while (running) {
      try {
        const task = await dequeue();
        if (!task) {
          await new Promise((r) => setTimeout(r, 300));
          continue;
        }
        const handler = handlers.get(task.type);
        if (!handler) {
          console.warn(`[queue#${workerId}] 无 handler: ${task.type}`);
          continue;
        }
        await handler(task.payload, task);
      } catch (err) {
        console.warn(`[queue#${workerId}] task failed:`, err.message);
      }
    }
  }

  const api = {
    backend: () => (degraded ? "memory" : "redis"),
    register(type, handler) { handlers.set(type, handler); return api; },
    enqueue,
    dequeue,
    async size() {
      if (redis && !degraded) {
        try { return await redis.zcard(KEY); } catch { degraded = true; }
      }
      return local.length;
    },
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
