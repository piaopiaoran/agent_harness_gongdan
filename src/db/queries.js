import { config } from "../config.js";
import { getPool, withTenant } from "./pool.js";

const T = config.tables;
const vectorLiteral = (vec) => `[${(vec ?? []).join(",")}]`;

// ---------------------------------------------------------------------------
// 工作记忆：全量原始流水（append-only 语义，只读蒸馏标记会更新）
// ---------------------------------------------------------------------------
export async function appendWorkingMemory(row) {
  await getPool().query(
    `INSERT INTO ${T.workingMemory}
       (trace_id, thread_id, tenant_id, user_id, role, kind, content, tool_calls)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb)`,
    [row.traceId, row.threadId, row.tenantId, row.userId, row.role, row.kind,
      String(row.content ?? "").slice(0, 12000), JSON.stringify(row.toolCalls ?? [])]
  );
}

export async function listUndistilled(tenantId, limit = 8) {
  return withTenant(tenantId, (c) => c.query(
    `SELECT id, trace_id, thread_id, user_id, role, kind, content, tool_calls, created_at
       FROM ${T.workingMemory}
      WHERE distilled = false AND kind IN ('human','ai')
      ORDER BY id ASC LIMIT $1`,
    [limit]
  )).then((r) => r.rows);
}

export async function markDistilled(ids) {
  if (!ids?.length) return;
  await getPool().query(
    `UPDATE ${T.workingMemory} SET distilled = true WHERE id = ANY($1::bigint[])`,
    [ids]
  );
}

// ---------------------------------------------------------------------------
// 情景记忆：蒸馏后的 episode 摘要
// ---------------------------------------------------------------------------
export async function insertEpisode(row) {
  // 幂等：重跑同一批消息时，唯一索引 (tenant,user,idempotency_key) 冲突 → 返回 null，
  // 调用方视为「已写入」并照常标记 distilled，避免重复 episode。
  return withTenant(row.tenantId, (c) => c.query(
    `INSERT INTO ${T.episodes}
       (tenant_id, user_id, thread_id, goal, summary, outcome, failure_points, source_trace_ids,
        embedding, idempotency_key, extractor_version)
     VALUES ($1,$2,$3,$4,$5,$6,$7::text[],$8::text[],$9::vector,$10,$11)
     ON CONFLICT DO NOTHING
     RETURNING id, created_at`,
    [row.tenantId, row.userId, row.threadId, row.goal, row.summary, row.outcome,
      row.failurePoints ?? [], row.sourceTraceIds ?? [], vectorLiteral(row.embedding),
      row.idempotencyKey ?? "", row.extractorVersion ?? ""]
  )).then((r) => r.rows[0] ?? null);
}

export async function searchEpisodes({ tenantId, userId, embedding, limit = 5 }) {
  return withTenant(tenantId, (c) => c.query(
    `SELECT id, goal, summary, outcome, failure_points, source_trace_ids, created_at,
            1 - (embedding <=> $3::vector) AS similarity
       FROM ${T.episodes}
      WHERE tenant_id = $1 AND user_id = $2 AND embedding IS NOT NULL
        AND status = 'active' AND (valid_to IS NULL OR valid_to > now())
      ORDER BY embedding <=> $3::vector, created_at DESC
      LIMIT $4`,
    [tenantId, userId, vectorLiteral(embedding), limit]
  )).then((r) => r.rows);
}

// ---------------------------------------------------------------------------
// 语义记忆：稳定事实（Mem0 风格 add/search，PG 承载）
// ---------------------------------------------------------------------------
export async function upsertFact(row) {
  return withTenant(row.tenantId, (c) => c.query(
    `INSERT INTO ${T.semanticFacts}
       (tenant_id, user_id, fact, category, confidence, source_trace_id, embedding,
        extractor_version, source_kind)
     VALUES ($1,$2,$3,$4,$5,$6,$7::vector,$8,$9)
     ON CONFLICT DO NOTHING
     RETURNING id, created_at`,
    [row.tenantId, row.userId, row.fact, row.category ?? "preference",
      row.confidence ?? 0.8, row.sourceTraceId ?? null, vectorLiteral(row.embedding),
      row.extractorVersion ?? "", row.sourceKind ?? "user_said"]
  )).then((r) => r.rows[0] ?? null);
}

/** 冲突判定：pgvector 余弦相似度找同用户 active 事实，阈值由 config 决定。 */
export async function findSimilarFacts({ tenantId, userId, embedding, limit = 3, threshold = 0.82 }) {
  return withTenant(tenantId, (c) => c.query(
    `SELECT id, fact, category, confidence, created_at, valid_from,
            1 - (embedding <=> $3::vector) AS similarity
       FROM ${T.semanticFacts}
      WHERE tenant_id = $1 AND user_id = $2 AND embedding IS NOT NULL
        AND status = 'active' AND (valid_to IS NULL OR valid_to > now())
        AND 1 - (embedding <=> $3::vector) >= $4
      ORDER BY embedding <=> $3::vector
      LIMIT $5`,
    [tenantId, userId, vectorLiteral(embedding), threshold, limit]
  )).then((r) => r.rows);
}

export async function searchFacts({ tenantId, userId, embedding, limit = 5 }) {
  return withTenant(tenantId, (c) => c.query(
    `SELECT id, fact, category, confidence, created_at, source_kind,
            1 - (embedding <=> $3::vector) AS similarity
       FROM ${T.semanticFacts}
      WHERE tenant_id = $1 AND user_id = $2 AND embedding IS NOT NULL
        AND status = 'active' AND (valid_to IS NULL OR valid_to > now())
      ORDER BY embedding <=> $3::vector
      LIMIT $4`,
    [tenantId, userId, vectorLiteral(embedding), limit]
  )).then((r) => r.rows);
}

export async function listFacts({ tenantId, userId, limit = 20 }) {
  return withTenant(tenantId, (c) => c.query(
    `SELECT id, fact, category, confidence, created_at, status, source_kind
       FROM ${T.semanticFacts}
      WHERE tenant_id = $1 AND user_id = $2 AND status = 'active'
      ORDER BY updated_at DESC LIMIT $3`,
    [tenantId, userId, limit]
  )).then((r) => r.rows);
}

/** 双时态更新：旧事实不删除，标 SUPERSEDED 并关闭 valid 区间，指向新行。 */
export async function supersedeFact({ tenantId, id, newId, reason = "" }) {
  return withTenant(tenantId, (c) => c.query(
    `UPDATE ${T.semanticFacts}
        SET status = 'superseded', valid_to = now(), superseded_by = $2, updated_at = now()
      WHERE id = $1 AND tenant_id = $3 AND status = 'active'`,
    [id, newId ?? null, tenantId]
  )).then((r) => r.rowCount ?? 0);
}

/** 矛盾 / 过期事实标 INVALID；保留行与向量，后续不再召回。 */
export async function invalidateFact({ tenantId, id, reason = "" }) {
  return withTenant(tenantId, (c) => c.query(
    `UPDATE ${T.semanticFacts}
        SET status = 'invalid', valid_to = now(), updated_at = now()
      WHERE id = $1 AND tenant_id = $2 AND status = 'active'`,
    [id, tenantId]
  )).then((r) => r.rowCount ?? 0);
}

export async function appendFactHistory(row) {
  await withTenant(row.tenantId, (c) => c.query(
    `INSERT INTO ${T.factHistory}
       (tenant_id, user_id, fact_id, op, fact, reason, similarity, extractor_version, source_trace_id)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`,
    [row.tenantId, row.userId, row.factId ?? null, row.op, row.fact ?? "", row.reason ?? "",
      row.similarity ?? null, row.extractorVersion ?? "", row.sourceTraceId ?? null]
  ));
}

/** 工作记忆无 RLS：cron 需要从这里枚举租户，再逐租户带着 tenant context 归档。 */
export async function listTenants() {
  const r = await getPool().query(
    `SELECT DISTINCT tenant_id FROM ${T.workingMemory} ORDER BY tenant_id`
  );
  return r.rows.map((x) => x.tenant_id);
}

/**
 * TTL 归档：只改状态与 valid_to，历史仍可查；返回归档条数供观测。
 * FORCE RLS 下必须在 withTenant 事务里执行，否则策略会静默过滤掉所有行。
 */
export async function archiveExpiredMemory({ factTtlDays, episodeTtlDays, tenantId }) {
  return withTenant(tenantId, async (c) => {
    const facts = await c.query(
      `UPDATE ${T.semanticFacts} SET status = 'archived', valid_to = now(), updated_at = now()
        WHERE status = 'active' AND valid_to IS NULL
          AND tenant_id = $2
          AND created_at < now() - ($1::int * INTERVAL '1 day')`,
      [factTtlDays, tenantId]
    );
    const episodes = await c.query(
      `UPDATE ${T.episodes} SET status = 'archived', valid_to = now()
        WHERE status = 'active' AND valid_to IS NULL
          AND tenant_id = $2
          AND created_at < now() - ($1::int * INTERVAL '1 day')`,
      [episodeTtlDays, tenantId]
    );
    return { facts: facts.rowCount ?? 0, episodes: episodes.rowCount ?? 0 };
  });
}

// ---------------------------------------------------------------------------
// RAG：文档版本与 chunks（两路召回的 SQL 都在此，权限条件下推）
// ---------------------------------------------------------------------------
export async function upsertDocument({ tenantId, source, title, acl = [], contentHash }) {
  const pool = getPool();
  const found = await withTenant(tenantId, (c) => c.query(
    `SELECT id, version, content_hash FROM ${T.ragDocuments}
      WHERE tenant_id = $1 AND source = $2 ORDER BY version DESC LIMIT 1`,
    [tenantId, source]
  ));
  const prev = found.rows[0];
  if (prev && prev.content_hash === contentHash) {
    return { id: prev.id, version: prev.version, changed: false };
  }
  const version = (prev?.version ?? 0) + 1;
  const inserted = await withTenant(tenantId, (c) => c.query(
    `INSERT INTO ${T.ragDocuments} (tenant_id, source, title, version, acl, content_hash)
     VALUES ($1,$2,$3,$4,$5::text[],$6) RETURNING id`,
    [tenantId, source, title, version, acl, contentHash]
  ));
  if (prev) {
    // 更新走新版本 + 软删旧 chunk，保证已入库的历史版本可追溯
    await pool.query(`UPDATE ${T.ragChunks} SET active = false WHERE doc_id = $1`, [prev.id]);
    await pool.query(`UPDATE ${T.ragDocuments} SET active = false WHERE id = $1`, [prev.id]);
  }
  return { id: inserted.rows[0].id, version, changed: true };
}

export async function insertChunks(chunks) {
  if (!chunks.length) return;
  const client = await getPool().connect();
  try {
    await client.query("BEGIN");
    await client.query("SELECT set_config('app.tenant_id', $1, true)", [chunks[0].tenantId]);
    for (const ch of chunks) {
      await client.query(
        `INSERT INTO ${T.ragChunks}
           (doc_id, tenant_id, chunk_idx, content, tokens, search_text, tsv, embedding, content_hash)
         VALUES ($1,$2,$3,$4,$5, $6, to_tsvector('simple', $6), $7::vector, $8)`,
        [ch.docId, ch.tenantId, ch.chunkIdx, ch.content, ch.tokens, ch.searchText,
          vectorLiteral(ch.embedding), ch.contentHash]
      );
    }
    await client.query("COMMIT");
  } catch (err) {
    await client.query("ROLLBACK").catch(() => {});
    throw err;
  } finally {
    client.release();
  }
}

export async function getDocumentTitles({ tenantId, ids }) {
  if (!ids?.length) return new Map();
  return withTenant(tenantId, (c) => c.query(
    `SELECT id, title, source, version FROM ${T.ragDocuments}
      WHERE tenant_id = $1 AND id = ANY($2::uuid[])`,
    [tenantId, ids]
  )).then((r) => new Map(r.rows.map((d) => [d.id, d])));
}

/**
 * 知识库文档目录（只读）：只暴露当前租户的 active 版本文档与其 active chunks。
 * 不查 search_text / tsv / embedding：这些是检索内部数据，对前端没有展示价值且体积大。
 */
export async function listRagDocuments({ tenantId, limit = 50, chunkLimit = 200 }) {
  const documents = await withTenant(tenantId, (c) => c.query(
    `SELECT id, source, title, version, acl, created_at
       FROM ${T.ragDocuments}
      WHERE tenant_id = $1 AND active = true
      ORDER BY source ASC, version DESC LIMIT $2`,
    [tenantId, limit]
  )).then((r) => r.rows);
  if (!documents.length) return { documents: [], chunks: [] };

  const chunks = await withTenant(tenantId, (c) => c.query(
    `SELECT id, doc_id, chunk_idx, content, tokens, created_at
       FROM ${T.ragChunks}
      WHERE tenant_id = $1 AND active = true AND doc_id = ANY($2::uuid[])
      ORDER BY doc_id, chunk_idx LIMIT $3`,
    [tenantId, documents.map((d) => d.id), chunkLimit]
  )).then((r) => r.rows);
  return { documents, chunks };
}

/** 向量路：HNSW 余弦距离，tenant 条件下推（RLS 之外再显式过滤一次，双保险）。 */
export async function vectorSearchChunks({ tenantId, embedding, limit = 30 }) {
  return withTenant(tenantId, (c) => c.query(
    `SELECT id, doc_id, chunk_idx, content, tokens, content_hash, created_at,
            1 - (embedding <=> $2::vector) AS similarity
       FROM ${T.ragChunks}
      WHERE tenant_id = $1 AND active = true AND embedding IS NOT NULL
      ORDER BY embedding <=> $2::vector
      LIMIT $3`,
    [tenantId, vectorLiteral(embedding), limit]
  )).then((r) => r.rows);
}

/**
 * 关键词路：jieba 风格分词后落 tsvector，ts_rank_cd 打分（非严格 BM25）。
 * 用 OR 连接查询词（plainto_tsquery 默认 AND 会让长 query 零召回）。
 * 查询词已由 segmenter 限定为 CJK / 字母数字，再白名单过滤一次，杜绝 tsquery 注入。
 */
export async function keywordSearchChunks({ tenantId, tsQuery, limit = 30 }) {
  const safe = String(tsQuery ?? "")
    .split(/\s+/)
    .filter((t) => /^[\u4e00-\u9fa5a-z0-9]+$/.test(t))
    .slice(0, 64)
    .join(" | ");
  if (!safe) return [];
  return withTenant(tenantId, (c) => c.query(
    `SELECT id, doc_id, chunk_idx, content, tokens, content_hash, created_at,
            ts_rank_cd(tsv, to_tsquery('simple', $2)) AS rank
       FROM ${T.ragChunks}
      WHERE tenant_id = $1 AND active = true AND tsv @@ to_tsquery('simple', $2)
      ORDER BY rank DESC
      LIMIT $3`,
    [tenantId, safe, limit]
  )).then((r) => r.rows);
}

// ---------------------------------------------------------------------------
// L10 审计（append-only，RULE 拦截删改）
// ---------------------------------------------------------------------------
export async function appendAudit(row) {
  await getPool().query(
    `INSERT INTO ${T.auditEvents}
       (trace_id, thread_id, tenant_id, user_id, node, event, decision, risk_flags, detail)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8::text[],$9::jsonb)`,
    [row.traceId, row.threadId ?? "", row.tenantId, row.userId, row.node, row.event,
      row.decision ?? "allow", row.riskFlags ?? [], JSON.stringify(row.detail ?? {})]
  );
}

export async function listAudit(traceId, limit = 200) {
  return getPool().query(
    `SELECT node, event, decision, risk_flags, detail, created_at
       FROM ${T.auditEvents} WHERE trace_id = $1 ORDER BY id ASC LIMIT $2`,
    [traceId, limit]
  ).then((r) => r.rows);
}

// ---------------------------------------------------------------------------
// 工具落库：审批后的退款工单
// ---------------------------------------------------------------------------
export async function insertRefundTicket(row) {
  return getPool().query(
    `INSERT INTO ${T.refundTickets}
       (tenant_id, order_id, user_id, amount, reason, status, approver_id, trace_id)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING id, status, created_at`,
    [row.tenantId, row.orderId, row.userId, row.amount, row.reason,
      row.status ?? "approved_pending_fulfill", row.approverId, row.traceId]
  ).then((r) => r.rows[0]);
}

export async function listRefundTickets({ tenantId, userId, limit = 20 }) {
  return withTenant(tenantId, (c) => c.query(
    `SELECT id, order_id, amount, reason, status, approver_id, created_at
       FROM ${T.refundTickets} WHERE user_id = $2 ORDER BY created_at DESC LIMIT $3`,
    [tenantId, userId, limit]
  )).then((r) => r.rows);
}

// ---------------------------------------------------------------------------
// 评测记录
// ---------------------------------------------------------------------------
export async function insertEvalRun({ runName, dataset, metrics, baseline, passed }) {
  return getPool().query(
    `INSERT INTO ${T.evalRuns} (run_name, dataset, metrics, baseline, passed)
     VALUES ($1,$2,$3::jsonb,$4::jsonb,$5) RETURNING id, created_at`,
    [runName, dataset, JSON.stringify(metrics), JSON.stringify(baseline), Boolean(passed)]
  ).then((r) => r.rows[0]);
}

/** 评测看板只读查询：eval_runs 无 tenant 列，查询不入 withTenant。 */
export async function listEvalRuns(limit = 20) {
  const n = Math.max(1, Math.min(100, Number(limit) || 20));
  return getPool().query(
    `SELECT id, run_name, dataset, metrics, baseline, passed, created_at
       FROM ${T.evalRuns} ORDER BY created_at DESC, id DESC LIMIT $1`,
    [n]
  ).then((r) => r.rows);
}
