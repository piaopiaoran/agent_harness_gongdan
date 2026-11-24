import Redis from "ioredis";
import { config } from "../config.js";

/**
 * L1 接入限流：Redis 固定窗口计数。
 * 过载时快速失败（429），不排队拖垮后端。Redis 不可用时降级内存计数（fail-open + 告警），
 * 因为限流是基础设施层，不能因为缓存故障让全部请求失败。
 */
export function createRateLimiter({ url = config.REDIS_URL, limit = config.RATE_LIMIT_PER_MIN, windowMs = 60_000 } = {}) {
  let redis = null;
  let redisDown = false;
  try {
    // 保留 offlineQueue：ioredis 建立首连接前禁用队列会让首个 INCR 直接失败。
    redis = new Redis(url, { maxRetriesPerRequest: 1, retryStrategy: (n) => Math.min(n * 200, 2000) });
    redis.on("error", () => { if (!redisDown) { redisDown = true; console.warn("[rate-limit] Redis 不可用，降级内存计数"); } });
  } catch (err) {
    redisDown = true;
    console.warn("[rate-limit] Redis 初始化失败，降级内存计数:", err.message);
  }
  const memory = new Map();

  return {
    backend: () => (redisDown ? "memory" : "redis"),
    async check(key) {
      const now = Date.now();
      if (redis && !redisDown) {
        try {
          const rk = `agent_harness_gongdan:rl:${key}`;
          const n = await redis.incr(rk);
          if (n === 1) await redis.pexpire(rk, windowMs);
          return n <= limit
            ? { ok: true, remaining: limit - n }
            : { ok: false, retryAfterMs: await redis.pttl(rk) };
        } catch {
          redisDown = true;
        }
      }
      const entry = memory.get(key);
      if (!entry || now - entry.start > windowMs) {
        memory.set(key, { start: now, count: 1 });
        return { ok: true, remaining: limit - 1 };
      }
      entry.count += 1;
      return entry.count <= limit
        ? { ok: true, remaining: limit - entry.count }
        : { ok: false, retryAfterMs: windowMs - (now - entry.start) };
    },
    async close() {
      if (redis) await redis.quit().catch(() => {});
    },
  };
}
