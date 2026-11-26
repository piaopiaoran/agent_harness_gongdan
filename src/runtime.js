import { randomUUID } from "node:crypto";
import { Command, MemorySaver } from "@langchain/langgraph";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { config, CHECKPOINT_SCHEMA } from "./config.js";
import { getPool, closePool } from "./db/pool.js";
import { createEmbedder } from "./llm/embedding.js";
import { createModel } from "./llm/model.js";
import { createTools } from "./tools.js";
import { createRateLimiter } from "./security/rateLimit.js";
import { createPriorityQueue } from "./queue/priority.js";
import { createAuditor } from "./security/guard.js";
import { buildGraph } from "./graph/buildGraph.js";
import { distillOnce } from "./memory/distiller.js";
import { traceConfig } from "./observability/tracing.js";
import { createCostLedger, createTokenBudget } from "./observability/cost.js";
import { createForgetScheduler } from "./memory/forget.js";

/** 全局共享资源：连接池、模型、checkpointer、限流器、优先级队列与 worker。 */
export async function createRuntime({ startWorker = true, forceMemoryCheckpointer = false } = {}) {
  const embedder = createEmbedder();
  let checkpointer;
  if (forceMemoryCheckpointer) {
    checkpointer = new MemorySaver();
  } else {
    checkpointer = new PostgresSaver(getPool(), undefined, { schema: CHECKPOINT_SCHEMA });
    await checkpointer.setup();
  }
  const rateLimiter = createRateLimiter();
  const queue = createPriorityQueue({ startWorker: false });
  // 蒸馏独立账本 + 日预算：批处理不并入在线会话账本，也不抢在线配额。
  const distillLedger = createCostLedger(config.distill.model);
  const distillBudget = createTokenBudget({ dailyLimit: config.distill.dailyTokenBudget });
  queue.register("distill", (payload) => distillOnce(embedder, {
    ...payload, costLedger: distillLedger, budget: distillBudget,
  }));
  if (startWorker) queue.start();
  // TTL 归档：6h 一轮（配置项），只做软归档与审计，不碰数据库进程。
  const forget = createForgetScheduler({ autoStart: startWorker });

  return {
    embedder, checkpointer, rateLimiter, queue, forget,
    distill: { ledger: distillLedger, budget: distillBudget },
    modelMode: config.modelMode,
    embeddingMode: config.embeddingMode,
    async close() {
      forget.stop();
      await queue.stop();
      await rateLimiter.close();
      await closePool();
    },
  };
}

/** 每次请求创建独立 session：身份、审计、工具注册表、模型绑定都不跨请求复用。 */
export function createSession(runtime, { tenantId = "tenant-a", userId = "u-1", role = "user", feature = "ticket", sessionId = randomUUID(), rateResult = null } = {}) {
  const identity = {
    traceId: randomUUID(), sessionId, tenantId, userId, role, feature,
  };
  const auditor = createAuditor(identity);
  const tools = createTools({ identity, traceId: identity.traceId, embedder: runtime.embedder });
  const model = createModel({ tools: tools.langchainTools });
  // 每次请求的账本：调用级真实 usage → 成本，随会话存活，不跨进程持久化。
  const costLedger = createCostLedger();
  const graph = buildGraph({
    ...runtime, auditor, identity, model, registry: tools.registry, costLedger,
    rateResult: rateResult ?? { backend: runtime.rateLimiter.backend(), remaining: null },
  });
  const runConfig = {
    configurable: { thread_id: sessionId },
    recursionLimit: config.MAX_STEPS * 3,
    ...traceConfig(identity),
  };

  return {
    identity, auditor, graph, runConfig, sessionId, costLedger,
    start(raw) {
      return graph.invoke({ input: { raw: String(raw ?? "") }, identity }, runConfig);
    },
    resume(reply) {
      return graph.invoke(new Command({ resume: reply }), runConfig);
    },
  };
}

export function extractInterrupt(result) {
  const list = result?.__interrupt__;
  return Array.isArray(list) && list.length ? list : null;
}
