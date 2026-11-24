import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { config, CHECKPOINT_SCHEMA } from "../src/config.js";
import { getPool, closePool } from "../src/db/pool.js";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

const schemaPath = fileURLToPath(new URL("../db/schema.sql", import.meta.url));
const sql = await readFile(schemaPath, "utf8");
const pool = getPool();

try {
  await pool.query(sql);
  console.log("[db:init] 业务表 / 索引 / RULE / RLS 策略已就绪（全部 agent_harness_gongdan_ 前缀）");

  const saver = new PostgresSaver(pool, undefined, { schema: CHECKPOINT_SCHEMA });
  await saver.setup();
  console.log(`[db:init] LangGraph checkpoint 表已就绪（schema "${CHECKPOINT_SCHEMA}"，官方表名硬编码只支持改 schema）`);

  const tables = await pool.query(
    `SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE $1 ORDER BY 1`,
    ["agent_harness_gongdan_%"]
  );
  console.log("[db:init] 本项目表:", tables.rows.map((r) => r.tablename).join(", "));

  const ck = await pool.query(
    `SELECT tablename FROM pg_tables WHERE schemaname = $1 ORDER BY 1`,
    [CHECKPOINT_SCHEMA]
  );
  console.log("[db:init] checkpoint 表:", ck.rows.map((r) => r.tablename).join(", "));
} catch (err) {
  console.error("[db:init] 失败:", err.message);
  process.exitCode = 1;
} finally {
  await closePool();
}
