import { config, CHECKPOINT_SCHEMA, PREFIX } from "../src/config.js";
import { getPool, closePool } from "../src/db/pool.js";

const pool = getPool();
try {
  const info = await pool.query(
    `SELECT current_database() AS db, current_user AS usr,
            (SELECT count(*)::int FROM pg_extension WHERE extname = 'vector') AS vector_ext`
  );
  console.log("[db] 连接:", info.rows[0]);

  const objects = await pool.query(
    `SELECT 'table' AS kind, tablename AS name FROM pg_tables
       WHERE schemaname='public' AND tablename LIKE $1
     UNION ALL
     SELECT 'index', indexname FROM pg_indexes
       WHERE schemaname='public' AND indexname LIKE $1
     UNION ALL
     SELECT 'policy', policyname FROM pg_policies
       WHERE schemaname='public' AND policyname LIKE $1
     ORDER BY kind, name`,
    [`${PREFIX}%`]
  );
  console.log(`[db] 已创建 ${objects.rows.length} 个带前缀的对象:`);
  for (const row of objects.rows) console.log(`  ${row.kind.padEnd(7)} ${row.name}`);

  const ck = await pool.query(
    `SELECT tablename FROM pg_tables WHERE schemaname=$1 ORDER BY 1`, [CHECKPOINT_SCHEMA]
  );
  console.log(`[db] checkpoint schema "${CHECKPOINT_SCHEMA}":`, ck.rows.map((r) => r.tablename).join(", ") || "(空)");
} catch (err) {
  console.error("[db] 检查失败:", err.message);
  process.exitCode = 1;
} finally {
  await closePool();
}
