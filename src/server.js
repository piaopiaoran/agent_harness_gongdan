import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import { config, PREFIX } from "./config.js";
import { createRuntime, createSession, extractInterrupt } from "./runtime.js";
import { listAudit, listFacts, searchEpisodes } from "./db/queries.js";
import { getPool } from "./db/pool.js";
import { createEmbedder } from "./llm/embedding.js";
import { flushTraces, logTracingStatus, tracingStatus } from "./observability/tracing.js";
import { TOOL_META } from "./tools.js";

const publicDir = join(dirname(fileURLToPath(import.meta.url)), "..", "public");
const MAX_SESSIONS = 300;

export function buildApp(runtime) {
  const app = express();
  const sessions = new Map();

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
      rag: {
        recallTopK: config.RAG_RECALL_TOPK, rrfK: config.RAG_RRF_K,
        rerankTopN: config.RAG_RERANK_TOPN, threshold: config.RAG_RERANK_THRESHOLD,
        evidenceMax: config.RAG_EVIDENCE_MAX,
      },
      queue: { backend: runtime.queue.backend() },
      rateLimit: { backend: runtime.rateLimiter.backend(), perMin: config.RATE_LIMIT_PER_MIN },
      tools: Object.entries(TOOL_META).map(([name, meta]) => ({ name, ...meta })),
    });
  });

  app.post("/chat", async (req, res) => {
    // 演示用身份来自请求头；生产必须从 JWT / OIDC session 注入。
    const tenantId = String(req.get("x-tenant-id") ?? "tenant-a");
    const userId = String(req.get("x-user-id") ?? "u-1");
    const role = req.get("x-role") === "admin" ? "admin" : "user";

    const rl = await runtime.rateLimiter.check(`${tenantId}:${userId}`);
    if (!rl.ok) return res.status(429).json({ error: "rate_limited", retryAfterMs: rl.retryAfterMs });

    const sessionId = String(req.body?.sessionId ?? "");
    const sess = createSession(runtime, { tenantId, userId, role, sessionId, rateResult: rl });
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
