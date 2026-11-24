-- ============================================================================
-- agent_harness_gongdan / 售后工单 Agent —— 数据库基线
--
-- 运行环境：与其他项目共用的本机 PGlite（PostgreSQL 18.7 + pgvector），
--           连接串 postgresql://postgres@127.0.0.1:5432/postgres。
-- 命名约定：本项目所有自建对象（表 / 索引 / 序列 / RULE / RLS 策略）统一
--           以 agent_harness_gongdan_ 开头。
-- LangGraph 官方 PostgresSaver 的四张表名在 dist/sql.js 里硬编码、无法通过
-- 参数改前缀，只支持自定义 schema；因此放独立 schema "agent_harness_gongdan"，
-- 完整限定名 agent_harness_gongdan.checkpoints 同样带项目命名空间，且与共享库
-- 里其他项目的 checkpointer 完全隔离（与 agent-security-lab 的处理方式一致）。
--
-- 设计要点（对应 Harness 标准）：
--   1. 工作记忆 = 全量原始流水，append-only，只供审计与蒸馏，不进 prompt
--   2. 情景记忆 / 语义记忆 / RAG chunks 共用 pgvector，但分区分表、独立召回
--   3. 情景 / 语义 / 知识表开 RLS + FORCE RLS，tenant_id 隔离下沉到数据库层
--   4. 审计表用 RULE 拦截 UPDATE/DELETE，保证取证链不可篡改
--   5. HNSW 余弦索引 + GIN 倒排索引，对应双路召回
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
-- gen_random_uuid() 是 PG13+ 内置函数，无需 pgcrypto，避免对共享库提权。

-- ---------------------------------------------------------------------------
-- 工作记忆：会话原始流水（审计 + 蒸馏源，不参与 prompt 注入）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_harness_gongdan_working_memory (
  id          BIGSERIAL PRIMARY KEY,
  trace_id    UUID NOT NULL,
  thread_id   TEXT NOT NULL,
  tenant_id   TEXT NOT NULL,
  user_id     TEXT NOT NULL,
  role        TEXT NOT NULL,
  kind        TEXT NOT NULL,              -- human | ai | tool | system | reject
  content     TEXT NOT NULL,
  tool_calls  JSONB NOT NULL DEFAULT '[]'::jsonb,
  distilled   BOOLEAN NOT NULL DEFAULT false,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_wm_trace_idx
  ON agent_harness_gongdan_working_memory (trace_id, id);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_wm_undistilled_idx
  ON agent_harness_gongdan_working_memory (created_at)
  WHERE distilled = false;

-- ---------------------------------------------------------------------------
-- 情景记忆：蒸馏后的事件摘要（目标 / 过程 / 结果 / 失败点），带向量与时间标签
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_harness_gongdan_episodes (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id        TEXT NOT NULL,
  user_id          TEXT NOT NULL,
  thread_id        TEXT NOT NULL,
  goal             TEXT NOT NULL,
  summary          TEXT NOT NULL,
  outcome          TEXT NOT NULL DEFAULT '',
  failure_points   TEXT[] NOT NULL DEFAULT '{}',
  source_trace_ids TEXT[] NOT NULL DEFAULT '{}',
  embedding        vector(1024),
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_episode_tenant_time_idx
  ON agent_harness_gongdan_episodes (tenant_id, user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_episode_embedding_hnsw
  ON agent_harness_gongdan_episodes USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- 语义记忆：稳定事实 / 用户偏好 / 业务规则（Mem0 风格接口，PG 承载）
-- 只喂原始消息抽取，禁止把 episode 摘要再喂回来污染事实。
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_harness_gongdan_semantic_facts (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id       TEXT NOT NULL,
  user_id         TEXT NOT NULL,
  fact            TEXT NOT NULL,
  category        TEXT NOT NULL DEFAULT 'preference', -- preference | order | business_rule
  confidence      REAL NOT NULL DEFAULT 0.8,
  source_trace_id TEXT,
  embedding       vector(1024),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS agent_harness_gongdan_fact_dedup_idx
  ON agent_harness_gongdan_semantic_facts (tenant_id, user_id, category, fact);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_fact_tenant_idx
  ON agent_harness_gongdan_semantic_facts (tenant_id, user_id, category);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_fact_embedding_hnsw
  ON agent_harness_gongdan_semantic_facts USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- RAG 文档（版本、来源、ACL）与 chunks（原文、tokens、tsvector、向量）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_harness_gongdan_rag_documents (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id    TEXT NOT NULL,
  source       TEXT NOT NULL,
  title        TEXT NOT NULL,
  version      INT NOT NULL DEFAULT 1,
  acl          TEXT[] NOT NULL DEFAULT '{}',
  content_hash TEXT NOT NULL,
  active       BOOLEAN NOT NULL DEFAULT true,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_rag_doc_tenant_idx
  ON agent_harness_gongdan_rag_documents (tenant_id, source);

CREATE TABLE IF NOT EXISTS agent_harness_gongdan_rag_chunks (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doc_id       UUID NOT NULL REFERENCES agent_harness_gongdan_rag_documents(id) ON DELETE CASCADE,
  tenant_id    TEXT NOT NULL,
  chunk_idx    INT NOT NULL,
  content      TEXT NOT NULL,
  tokens       INT NOT NULL DEFAULT 0,
  search_text  TEXT NOT NULL DEFAULT '',  -- 分词后的空格串，供 simple config 建 tsvector
  tsv          TSVECTOR,
  embedding    vector(1024),
  content_hash TEXT NOT NULL,
  active       BOOLEAN NOT NULL DEFAULT true,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_rag_chunk_doc_idx
  ON agent_harness_gongdan_rag_chunks (doc_id, chunk_idx);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_rag_chunk_tsv_gin
  ON agent_harness_gongdan_rag_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_rag_chunk_embedding_hnsw
  ON agent_harness_gongdan_rag_chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- L10 审计事件：append-only（RULE 拦截 UPDATE / DELETE）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_harness_gongdan_audit_events (
  id          BIGSERIAL PRIMARY KEY,
  trace_id    UUID NOT NULL,
  thread_id   TEXT NOT NULL DEFAULT '',
  tenant_id   TEXT NOT NULL,
  user_id     TEXT NOT NULL,
  node        TEXT NOT NULL,
  event       TEXT NOT NULL,
  decision    TEXT NOT NULL,             -- allow | deny | escalate | approve | reject | degrade
  risk_flags  TEXT[] NOT NULL DEFAULT '{}',
  detail      JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_audit_trace_idx
  ON agent_harness_gongdan_audit_events (trace_id, id);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_audit_tenant_time_idx
  ON agent_harness_gongdan_audit_events (tenant_id, created_at DESC);

-- RULE 名字也带前缀；DO INSTEAD NOTHING 让删改静默失败而不是抛错。
DROP RULE IF EXISTS agent_harness_gongdan_audit_no_update ON agent_harness_gongdan_audit_events;
CREATE RULE agent_harness_gongdan_audit_no_update AS
  ON UPDATE TO agent_harness_gongdan_audit_events DO INSTEAD NOTHING;
DROP RULE IF EXISTS agent_harness_gongdan_audit_no_delete ON agent_harness_gongdan_audit_events;
CREATE RULE agent_harness_gongdan_audit_no_delete AS
  ON DELETE TO agent_harness_gongdan_audit_events DO INSTEAD NOTHING;

-- ---------------------------------------------------------------------------
-- 人工审批后的退款工单（L5 审批结果落库，供工具查询与审计对账）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_harness_gongdan_refund_tickets (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id   TEXT NOT NULL,
  order_id    TEXT NOT NULL,
  user_id     TEXT NOT NULL,
  amount      NUMERIC(12,2) NOT NULL DEFAULT 0,
  reason      TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'approved_pending_fulfill',
  approver_id TEXT NOT NULL,
  trace_id    UUID NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_refund_tenant_order_idx
  ON agent_harness_gongdan_refund_tickets (tenant_id, order_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- 评测结果（离线门禁产物，可追溯每次参数变更）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_harness_gongdan_eval_runs (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_name    TEXT NOT NULL,
  dataset     TEXT NOT NULL,
  metrics     JSONB NOT NULL DEFAULT '{}'::jsonb,
  baseline    JSONB NOT NULL DEFAULT '{}'::jsonb,
  passed      BOOLEAN NOT NULL DEFAULT false,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_harness_gongdan_eval_run_time_idx
  ON agent_harness_gongdan_eval_runs (created_at DESC);

-- ---------------------------------------------------------------------------
-- 行级安全：租户隔离下沉到数据库层，避免应用层一处漏写 WHERE 就横向越权。
-- current_setting('app.tenant_id', true) 未设置时返回 NULL → 策略拒绝全部行（fail-close）。
-- 所有读写必须通过 withTenant() 用 set_config($1, true) 在事务内绑定身份。
-- ---------------------------------------------------------------------------
DO $$
DECLARE t TEXT;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'agent_harness_gongdan_episodes',
    'agent_harness_gongdan_semantic_facts',
    'agent_harness_gongdan_rag_documents',
    'agent_harness_gongdan_rag_chunks'
  ]
  LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS %I ON %I', t || '_tenant_isolation', t);
    EXECUTE format(
      'CREATE POLICY %I ON %I USING (tenant_id = current_setting(''app.tenant_id'', true)) '
      || 'WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))',
      t || '_tenant_isolation', t
    );
  END LOOP;
END $$;
