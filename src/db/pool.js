import pg from "pg";
import { config, assertSharedDatabase } from "../config.js";

const { Pool } = pg;

let pool = null;

export function getPool() {
  if (!pool) {
    assertSharedDatabase(config.DATABASE_URL);
    pool = new Pool({
      connectionString: config.DATABASE_URL,
      // 共享服务是单实例 PGlite（wasm），底层同一时刻只能处理一个 IO。
      // 多连接并发会出现 "Only one IO can be handed out" / "unnamed prepared statement does not exist"。
      // 用 max=1 把 checkpoint 写入、审计、租户事务串行化；真实 PostgreSQL 可调大。
      max: 1,
      idleTimeoutMillis: 30_000,
      connectionTimeoutMillis: 10_000,
    });
    pool.on("error", (err) => console.error("[pg] idle client error", err.message));
  }
  return pool;
}

export async function closePool() {
  if (pool) {
    await pool.end();
    pool = null;
  }
}

/**
 * 租户上下文：用 SET LOCAL 把 tenant_id 绑在事务里，配合 RLS 做数据库层兜底。
 * 面试点：
 *  - 为什么不在 SQL 里字符串拼 WHERE tenant_id？→ 一处漏写就是横向越权。
 *  - 为什么 set_config($1, true) 而不是 SET？→ 参数化 + 事务级，连接归还池后不残留身份。
 */
export async function withTenant(tenantId, fn) {
  const client = await getPool().connect();
  try {
    await client.query("BEGIN");
    await client.query("SELECT set_config('app.tenant_id', $1, true)", [String(tenantId)]);
    const result = await fn(client);
    await client.query("COMMIT");
    return result;
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    throw err;
  } finally {
    client.release();
  }
}
