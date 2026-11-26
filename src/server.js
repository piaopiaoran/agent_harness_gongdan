import { randomUUID } from "node:crypto";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import { config, PREFIX } from "./config.js";
import { createRuntime, createSession, extractInterrupt } from "./runtime.js";
import { listAudit, listEvalRuns, listFacts, listRagDocuments, searchEpisodes } from "./db/queries.js";
import { buildEvalDashboard } from "./eval/dashboard.js";
import { buildDocumentCatalog } from "./rag/catalog.js";
import { getPool } from "./db/pool.js";
import { createEmbedder } from "./llm/embedding.js";
import { flushTraces, logTracingStatus, tracingStatus } from "./observability/tracing.js";
import { PRICING_NOTE } from "./observability/cost.js";
import { TOOL_META } from "./tools.js";

const publicDir = join(dirname(fileURLToPath(import.meta.url)), "..", "public");
const MAX_SESSIONS = 300;

/**
 * 会话 ID 即 LangGraph 的 thread_id。空字符串会让所有匿名访客共用同一个 checkpoint，
 * 历史越堆越长，还把上一次未回应的 tool_calls 带进下一轮请求。这里强制生成独立 UUID。
 */
export function resolveSessionId(input) {
  const value = String(input ?? "").trim();
  return value && value.length <= 128 ? value : randomUUID();
}

/** 功能场景归因：只收低基数白名单字符，防止污染 Smith tag / metadata。 */
export function resolveFeature(input) {
  const value = String(input ?? "").trim();
  return /^[a-zA-Z0-9_-]{1,32}$/.test(value) ? value : "ticket";
}

/**
 * 从审计事件里提取精排用量，作为独立段暴露，不并入主 LLM 的 costLedger。
 * 来源：rag_retrieved.detail.stages 中 stage === "rerank" 的项。
 * 只读取已上报的 token / 延迟，不猜价、不把 rerank 计进主模型账本。
 */
export function extractRerankUsage(auditEvents = []) {
  const stages = [];
  const degradations = [];
  for (const event of auditEvents ?? []) {
    if (event?.event !== "rag_retrieved") continue;
    const detail = event.detail ?? {};
    for (const stage of detail.stages ?? []) {
      if (stage?.stage === "rerank") stages.push(stage);
    }
    for (const d of detail.degradations ?? []) degradations.push(d);
  }
  if (!stages.length) {
    const managed = (config.RERANK_PROVIDER ?? "") === "siliconflow";
    return {
      calls: 0, provider: config.RERANK_PROVIDER ?? "heuristic",
      model: config.RERANK_MODEL ?? null, mode: config.rerankMode ?? null,
      inputTokens: 0, outputTokens: 0, latencyMs: 0,
      kept: 0, threshold: null, degraded: degradations.length > 0,
      priced: false,
      note: managed
        ? "本次未触发外部精排；无 BGE token 消耗"
        : "本次未触发精排；当前 provider 走本地打分，无外部 token",
    };
  }
  const last = stages[stages.length - 1];
  const inputTokens = stages.reduce((n, s) => n + Number(s.usage?.input_tokens ?? s.usage?.inputTokens ?? 0), 0);
  const outputTokens = stages.reduce((n, s) => n + Number(s.usage?.output_tokens ?? s.usage?.outputTokens ?? 0), 0);
  const latencyMs = stages.reduce((n, s) => n + Number(s.latencyMs ?? 0), 0);
  const degraded = degradations.some((d) => d?.stage === "rerank");
  const managed = last.provider === "siliconflow";
  return {
    calls: stages.length,
    provider: last.provider ?? config.RERANK_PROVIDER,
    model: config.RERANK_MODEL ?? null,
    mode: last.mode ?? null,
    inputTokens, outputTokens, latencyMs,
    kept: Number(last.kept ?? 0),
    threshold: last.threshold ?? null,
    degraded,
    priced: false,
    note: managed
      ? "精排为独立外部服务，token 不计入主模型账本；该模型当前在 SiliconFlow 免费，配置价格表后可按实际账单折算"
      : "本地启发式精排，0 次外部调用、0 token",
  };
}

export function buildApp(runtime) {
  const app = express();
  const sessions = new Map();
  /** 主 LLM 账本 + 独立精排段；两者不合并，避免污染 token / 成本口径。 */
  const usageOf = (sess) => ({
    ...sess.costLedger.summary(),
    rerank: extractRerankUsage(sess.auditor.events),
  });

  app.use((_req, res, next) => {
    res.setHeader("Content-Security-Policy",
      "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; " +
      "img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'self'");
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Referrer-Policy", "no-referrer");
    next();
  });
  app.use(express.json({ limit: "16kb" }));
  app.use(express.static(publicDir, { index: "index.html" }));

  app.get("/health", (_req, res) => {
    res.json({
      ok: true,
      project: PREFIX,
      model: config.modelLabel,
      modelMode: config.modelMode,
      embeddingMode: config.embeddingMode,
      tracing: tracingStatus(),
      budgets: config.budget,
      context: {
        window: config.context.window,
        safetyMargin: config.context.safetyMargin,
        requestedProfile: config.context.requestedProfile,
        defaultProfile: config.context.defaultProfile,
        profiles: Object.fromEntries(Object.entries(config.context.profiles).map(([name, b]) => [name, {
          assemblyCap: b.assemblyCap,
          headroom: b.headroom,
          outputReserve: b.outputReserve,
          system: b.system, state: b.state, memory: b.memory,
          rag: b.rag, recent: b.recent, remote: b.remote,
        }])),
      },
      pricing: {
        currency: config.pricing.currency,
        overrideModels: Object.keys(config.pricing.overrides),
        note: PRICING_NOTE,
      },
      rag: {
        recallTopK: config.RAG_RECALL_TOPK, rrfK: config.RAG_RRF_K,
        rerankTopN: config.RAG_RERANK_TOPN, threshold: config.RAG_RERANK_THRESHOLD,
        evidenceMax: config.RAG_EVIDENCE_MAX,
      },
      queue: { backend: runtime.queue.backend(), ...runtime.queue.stats() },
      memory: {
        recallMode: config.memory.recallMode,
        factMergeThreshold: config.memory.factMergeThreshold,
        halfLifeDays: config.memory.halfLifeDays,
        ttlDays: { episode: config.memory.episodeTtlDays, fact: config.memory.factTtlDays },
        forget: runtime.forget?.status() ?? null,
        distillBudget: runtime.distill?.budget.status() ?? null,
      },
      distill: {
        model: config.distill.label, extractorVersion: config.distill.extractorVersion,
        keyIsolated: config.distill.keyIsolated, jsonMode: config.distill.jsonMode,
        consolidateLlm: config.distill.consolidateLlm,
      },
      rateLimit: { backend: runtime.rateLimiter.backend(), perMin: config.RATE_LIMIT_PER_MIN },
      tools: Object.entries(TOOL_META).map(([name, meta]) => ({ name, ...meta })),
    });
  });

  app.post("/chat", async (req, res) => {
    // 演示用身份来自请求头；生产必须从 JWT / OIDC session 注入。
    const tenantId = String(req.get("x-tenant-id") ?? "tenant-a");
    const userId = String(req.get("x-user-id") ?? "u-1");
    const role = req.get("x-role") === "admin" ? "admin" : "user";
    const feature = resolveFeature(req.get("x-feature"));

    const rl = await runtime.rateLimiter.check(`${tenantId}:${userId}`);
    if (!rl.ok) return res.status(429).json({ error: "rate_limited", retryAfterMs: rl.retryAfterMs });

    const sessionId = resolveSessionId(req.body?.sessionId);
    const sess = createSession(runtime, { tenantId, userId, role, feature, sessionId, rateResult: rl });
    sessions.set(sess.sessionId, sess);
    if (sessions.size > MAX_SESSIONS) sessions.delete(sessions.keys().next().value);

    try {
      const result = await sess.start(String(req.body?.message ?? ""));
      const pending = extractInterrupt(result);
      res.json({
        traceId: sess.identity.traceId,
        sessionId: sess.sessionId,
        status: pending ? "awaiting_approval" : "done",
        answer: result.finalAnswer ?? "",
        verdict: result.verdict ?? "",
        interrupt: pending?.[0] ?? null,
        role,
        input: result.input ?? null,
        rag: result.rag ?? null,
        memories: result.memories ?? null,
        contextReport: result.contextReport ?? null,
        toolPlan: result.toolPlan ?? [],
        toolResults: result.toolResults ?? [],
        audit: sess.auditor.events,
        usage: usageOf(sess),
        distillQueued: Boolean(result.distillQueued),
      });
    } catch (err) {
      console.error("[chat] failed:", String(err.message ?? "").slice(0, 200));
      res.status(500).json({ error: "internal_error", traceId: sess.identity.traceId });
    }
  });

  app.post("/approve", async (req, res) => {
    const sess = sessions.get(String(req.body?.sessionId ?? ""));
    if (!sess) return res.status(404).json({ error: "session_not_found" });

    // 职责分离：审批人必须是 admin，且不能是发起人本人。
    const approverId = String(req.get("x-approver-id") ?? "ops-admin");
    if (req.get("x-role") !== "admin" || approverId === sess.identity.userId) {
      return res.status(403).json({ error: "approval_requires_independent_admin" });
    }
    try {
      const result = await sess.resume({
        decision: req.body?.decision === "approved" ? "approved" : "rejected",
        approverId,
        reason: String(req.body?.reason ?? ""),
      });
      res.json({
        traceId: sess.identity.traceId,
        sessionId: sess.sessionId,
        status: "done",
        answer: result.finalAnswer ?? "",
        verdict: result.verdict ?? "",
        approval: result.approval ?? null,
        toolResults: result.toolResults ?? [],
        audit: sess.auditor.events,
        usage: usageOf(sess),
        contextReport: result.contextReport ?? null,
        rag: result.rag ?? null,
      });
    } catch (err) {
      console.error("[approve] failed:", String(err.message ?? "").slice(0, 200));
      res.status(500).json({ error: "internal_error" });
    }
  });

  app.get("/audit/:traceId", async (req, res) => {
    try {
      res.json({ events: await listAudit(req.params.traceId) });
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  app.get("/inspect", async (req, res) => {
    const tenantId = String(req.get("x-tenant-id") ?? "tenant-a");
    const userId = String(req.get("x-user-id") ?? "u-1");
    try {
      const [facts, episodes] = await Promise.all([
        listFacts({ tenantId, userId, limit: 20 }),
        (async () => {
          const e = createEmbedder();
          return searchEpisodes({ tenantId, userId, embedding: await e.embedQuery("售后 工单"), limit: 10 });
        })(),
      ]);
      const tables = await getPool().query(
        `SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename LIKE $1 ORDER BY 1`,
        [`${PREFIX}_%`]
      );
      res.json({
        facts: facts.rows ?? facts,
        episodes: episodes.rows ?? episodes,
        tables: tables.rows.map((r) => r.tablename),
        queueSize: await runtime.queue.size(),
      });
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  /** 只读知识库目录：tenant 从身份头注入，SQL 层 active + tenant 双过滤。 */
  app.get("/api/rag/documents", async (req, res) => {
    const tenantId = String(req.get("x-tenant-id") ?? "tenant-a");
    try {
      const { documents, chunks } = await listRagDocuments({ tenantId });
      res.json({
        ...buildDocumentCatalog(documents, chunks),
        tenantId,
        generatedAt: new Date().toISOString(),
      });
    } catch (err) {
      res.status(503).json({
        error: "rag_store_unavailable",
        message: String(err.message ?? "").slice(0, 200),
      });
    }
  });

  /** 只读评测看板数据：来自本地 eval_runs，不代理 LangSmith（在线反馈在 trace 上看）。 */
  app.get("/api/evals", async (req, res) => {
    try {
      const rows = await listEvalRuns(req.query.limit);
      res.json({ ...buildEvalDashboard(rows), generatedAt: new Date().toISOString() });
    } catch (err) {
      res.status(503).json({
        error: "eval_store_unavailable",
        message: String(err.message ?? "").slice(0, 200),
      });
    }
  });

  return app;
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const runtime = await createRuntime();
  const app = buildApp(runtime);
  const server = app.listen(config.PORT, () => {
    console.log(`[server] http://localhost:${config.PORT}  model=${runtime.modelMode}  embedding=${runtime.embeddingMode}`);
    console.log(`[web] 售后工单 Agent 控制台: http://localhost:${config.PORT}/`);
    logTracingStatus();
  });

  async function shutdown() {
    await flushTraces();
    await runtime.close();
    server.close(() => process.exit(0));
  }
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}
