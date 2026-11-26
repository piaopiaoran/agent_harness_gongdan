import { createHash } from "node:crypto";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { config, getSecret } from "../config.js";
import { invokeModel, createFakeModel } from "../llm/model.js";
import { pendingForTenant, commitDistillation } from "./working.js";
import {
  insertEpisode, upsertFact, findSimilarFacts, supersedeFact, invalidateFact,
  appendFactHistory, appendAudit,
} from "../db/queries.js";
import { decideOperation, buildDecisionPrompt, parseDecision } from "./consolidate.js";

/**
 * 蒸馏抽取的结构化输出：zod 是唯一口径，JSON mode 负责「形状」，zod 负责「语义」。
 * 解析失败自动重问一次（industry: Pydantic/zod parse + repair），再失败才 fail-open。
 */
export const DistillOutput = z.object({
  goal: z.string().max(200).optional().default("售后咨询"),
  summary: z.string().max(600).optional().default(""),
  outcome: z.enum(["resolved", "unresolved", "escalated"]).optional().default("unresolved"),
  failure_points: z.array(z.string().max(200)).max(8).optional().default([]),
  facts: z.array(z.object({
    fact: z.string().min(2).max(200),
    category: z.enum(["preference", "order", "business_rule"]).optional().default("preference"),
    // 来源标注：用户说 / Agent 推断 / 工具返回（行业治理项）
    source_kind: z.enum(["user_said", "agent_inferred", "tool_returned"]).optional().default("user_said"),
    confidence: z.number().min(0).max(1).optional().default(0.75),
  })).max(8).optional().default([]),
});

export const DISTILL_SYSTEM = `你把售后工单会话流水压缩成结构化记忆。只输出 JSON，不要解释：
{
  "goal": "用户此轮要解决的问题（一句话）",
  "summary": "过程与关键事实（80字内）",
  "outcome": "resolved | unresolved | escalated",
  "failure_points": ["失败点或待办，没有则空数组"],
  "facts": [{"fact": "可复用的稳定事实或偏好", "category": "preference|order|business_rule", "source_kind": "user_said|agent_inferred|tool_returned", "confidence": 0.75}]
}
注意：facts 只能来自原始消息，不能把摘要再当事实；没有稳定事实就返回空数组。`;

/**
 * 蒸馏模型参数（纯函数，便于单测与审计）。
 *
 * 为什么必须关 thinking：
 * - DeepSeek V4 系列思考模式默认 enabled + effort=high，且思考模式下 temperature 不生效；
 * - 记忆抽取是「高频、短输入短输出、强结构化」任务，不需要多步推理；
 * - 思考会先吃掉输出预算、拉高延迟，还可能在 content 里夹带伪 JSON 导致解析失败。
 * 旧的 deepseek-chat 别名本来就是非思考模式，换成 deepseek-flash 后必须显式关掉，
 * 否则等于把抽取任务悄悄从「非思考」切成了「思考」。
 *
 * JSON 输出：object 是 OpenAI 兼容端点最通用的 JSON mode；schema 是严格 json_schema
 * （OpenAI 原生支持，DeepSeek 等兼容端点可能拒绝，由 DISTILL_JSON_MODE 控制）。
 */
export function distillModelOptions(cfg = config) {
  const mode = cfg.DISTILL_JSON_MODE ?? "object";
  const responseFormat = mode === "schema"
    ? { type: "json_schema", json_schema: DISTILL_JSON_SCHEMA }
    : mode === "object" ? { type: "json_object" } : undefined;
  return {
    model: cfg.DISTILL_LLM_MODEL || cfg.DISTILL_MODEL || cfg.LLM_MODEL,
    temperature: 0,
    maxTokens: cfg.DISTILL_MAX_TOKENS,
    timeout: cfg.DISTILL_TIMEOUT_MS ?? cfg.LLM_TIMEOUT_MS,
    maxRetries: cfg.DISTILL_MAX_RETRIES ?? 1,
    modelKwargs: {
      thinking: { type: "disabled" },
      ...(responseFormat ? { response_format: responseFormat } : {}),
    },
    configuration: { baseURL: cfg.DISTILL_BASE_URL || cfg.LLM_BASE_URL },
  };
}

const DISTILL_JSON_SCHEMA = {
  name: "memory_distill",
  strict: true,
  schema: {
    type: "object",
    properties: {
      goal: { type: "string", maxLength: 200 },
      summary: { type: "string", maxLength: 600 },
      outcome: { type: "string", enum: ["resolved", "unresolved", "escalated"] },
      failure_points: { type: "array", items: { type: "string" }, maxItems: 8 },
      facts: {
        type: "array", maxItems: 8,
        items: {
          type: "object",
          properties: {
            fact: { type: "string", minLength: 2, maxLength: 200 },
            category: { type: "string", enum: ["preference", "order", "business_rule"] },
            source_kind: { type: "string", enum: ["user_said", "agent_inferred", "tool_returned"] },
            confidence: { type: "number", minimum: 0, maximum: 1 },
          },
          required: ["fact", "category", "source_kind", "confidence"],
          additionalProperties: false,
        },
      },
    },
    required: ["goal", "summary", "outcome", "failure_points", "facts"],
    additionalProperties: false,
  },
};

function getDistillModel() {
  // 独立配置：DISTILL_API_KEY 未填时回退主 LLM key（共享配额，文档里明示）。
  if (!config.distill.hasKey) return createFakeModel();
  return new ChatOpenAI({
    ...distillModelOptions(config),
    apiKey: getSecret("distill"),
  });
}

/** 蒸馏只认真实 content；万一上游又开了 thinking，reasoning_content 里的伪 JSON 绝不能拿来解析。 */
function rawContent(message) {
  const raw = message?.content;
  if (typeof raw === "string") return raw;
  if (Array.isArray(raw)) return raw.map((b) => (typeof b === "string" ? b : b?.text ?? "")).join("");
  return "";
}

/** 只解析并校验；重问由调用方控制。 */
export function parseDistillOutput(text) {
  const raw = String(text ?? "");
  let candidate = raw;
  if (!candidate.trim().startsWith("{")) {
    const match = candidate.match(/\{[\s\S]*\}/);
    if (!match) throw new Error("distill_output_not_json");
    candidate = match[0];
  }
  return DistillOutput.parse(JSON.parse(candidate));
}

const REPAIR_HINT = "上次输出不是合法 JSON 或字段不符合 schema。只输出符合 schema 的 JSON，不要解释，不要 markdown 代码块。";

async function callWithRepair(model, rawText, callConfig) {
  let reply = await invokeModel(model, [
    new SystemMessage(DISTILL_SYSTEM),
    new HumanMessage(rawText.slice(0, 6000)),
  ], callConfig);
  try {
    return { data: parseDistillOutput(rawContent(reply)), reply, repaired: false };
  } catch (firstErr) {
    // 一次重问：把错误原因回灌，模型自行修正；仍失败则抛出交给 fail-open。
    reply = await invokeModel(model, [
      new SystemMessage(DISTILL_SYSTEM),
      new HumanMessage(rawText.slice(0, 6000)),
      new HumanMessage(REPAIR_HINT),
      new HumanMessage(`解析错误：${firstErr.message}`),
    ], callConfig);
    return { data: parseDistillOutput(rawContent(reply)), reply, repaired: true };
  }
}

/** 幂等键：租户 + thread + 本批消息 id。重跑同批不会重复 episode。 */
export function episodeIdempotencyKey({ tenantId, userId, threadId, rows }) {
  const ids = rows.map((r) => String(r.id)).sort().join(",");
  const body = `${tenantId}|${userId}|${threadId}|${ids}`;
  return createHash("sha256").update(body).digest("hex").slice(0, 40);
}

/**
 * 离线蒸馏：消费未蒸馏原始消息 → LLM 压缩 episode + 抽取事实 → 向量化写 PG → 标记已蒸馏。
 * 全程异步，不阻塞主请求。任何一步失败都保留 distilled=false，下一轮自动重试（fail-open）。
 */
export async function distillOnce(embedder, {
  tenantId,
  batchSize = config.DISTILL_BATCH_SIZE,
  costLedger = null,
  budget = null,
  onAudit = null,
} = {}) {
  const pending = await pendingForTenant(tenantId, batchSize);
  if (!pending.length) return { processed: 0 };

  const byThread = new Map();
  for (const row of pending) {
    const key = `${row.tenant_id}:${row.thread_id}`;
    if (!byThread.has(key)) byThread.set(key, []);
    byThread.get(key).push(row);
  }

  const model = getDistillModel();
  let episodes = 0;
  let facts = 0;
  const ops = { ADD: 0, UPDATE: 0, DELETE: 0, NOOP: 0 };
  const doneIds = [];

  const audit = (detail) => {
    if (!onAudit) return;
    try { onAudit(detail); } catch { /* 观测失败不影响蒸馏 */ }
  };

  for (const rows of byThread.values()) {
    try {
      // 预算门禁：批处理不能抢在线配额；超预算时本轮跳过（不标 distilled，下轮再试）。
      const estimated = Math.ceil(rows.reduce((n, r) => n + String(r.content).length, 0) / 2)
        + config.distill.maxTokens;
      if (budget && !budget.canSpend(estimated)) {
        audit({ event: "distill_budget_skipped", reason: "daily_budget_exceeded", estimated });
        break;
      }

      const raw = rows.map((r) => `${r.kind}: ${r.content}`).join("\n");
      const { data, reply, repaired } = await callWithRepair(model, raw, {
        tags: ["feature:distill", "stage:extract", `model:${config.distill.model}`],
        metadata: { feature: "distill", stage: "extract", model: config.distill.model,
          extractorVersion: config.distill.extractorVersion, tenantId, threadId: rows[0].thread_id },
      });
      const traceIds = [...new Set(rows.map((r) => r.trace_id))];
      const first = rows[0];
      if (costLedger) {
        costLedger.add(reply.usage_metadata, {
          feature: "distill", stage: "extract", tenantId,
          threadId: first.thread_id, model: config.distill.model, repaired,
        });
      }
      if (budget) budget.record(reply.usage_metadata);

      const episodeText = `${data.goal} ${data.summary}`.trim();
      const inserted = await insertEpisode({
        tenantId, userId: first.user_id, threadId: first.thread_id,
        goal: String(data.goal).slice(0, 200),
        summary: String(data.summary).slice(0, 600),
        outcome: String(data.outcome),
        failurePoints: data.failure_points.slice(0, 8).map(String),
        sourceTraceIds: traceIds,
        embedding: await embedder.embedQuery(episodeText),
        idempotencyKey: episodeIdempotencyKey({ tenantId, userId: first.user_id, threadId: first.thread_id, rows }),
        extractorVersion: config.distill.extractorVersion,
      });
      if (inserted) episodes += 1;

      for (const f of data.facts.slice(0, 6)) {
        const fact = String(f.fact ?? "").trim();
        if (!fact || fact.length < 4) continue;
        const embedding = await embedder.embedQuery(fact);
        const similar = await findSimilarFacts({
          tenantId, userId: first.user_id, embedding,
          threshold: config.memory.factMergeThreshold, limit: 3,
        });

        let decision = null;
        if (config.distill.consolidateLlm && similar.length) {
          try {
            const dReply = await invokeModel(model, [
              new SystemMessage("你是记忆整合器，只输出 JSON。"),
              new HumanMessage(buildDecisionPrompt({ fact, similar }).slice(0, 4000)),
            ], { tags: ["feature:distill", "stage:consolidate"], metadata: { feature: "distill", stage: "consolidate" } });
            decision = parseDecision(rawContent(dReply));
            if (costLedger) costLedger.add(dReply.usage_metadata, {
              feature: "distill", stage: "consolidate", tenantId, model: config.distill.model,
            });
            if (budget) budget.record(dReply.usage_metadata);
          } catch (err) {
            decision = { op: null, reason: `llm_decision_failed:${err.message}` };
          }
        }
        const d = decideOperation({ fact, similar, threshold: config.memory.factMergeThreshold, decision });
        ops[d.op] = (ops[d.op] ?? 0) + 1;
        if (d.op === "NOOP" || d.op === "DELETE") {
          if (d.op === "DELETE" && d.targetId) {
            await invalidateFact({ tenantId, id: d.targetId, reason: d.reason });
          }
          await appendFactHistory({
            tenantId, userId: first.user_id, factId: d.targetId, op: d.op, fact,
            reason: d.reason, similarity: d.similarity,
            extractorVersion: config.distill.extractorVersion, sourceTraceId: traceIds[0],
          });
          continue;
        }
        const created = await upsertFact({
          tenantId, userId: first.user_id, fact: fact.slice(0, 200),
          category: f.category, confidence: f.confidence, sourceKind: f.source_kind,
          sourceTraceId: traceIds[0], embedding,
          extractorVersion: config.distill.extractorVersion,
        });
        if (d.op === "UPDATE" && d.targetId && created) {
          await supersedeFact({ tenantId, id: d.targetId, newId: created.id, reason: d.reason });
        }
        await appendFactHistory({
          tenantId, userId: first.user_id, factId: created?.id ?? d.targetId, op: created ? d.op : "NOOP",
          fact, reason: d.reason, similarity: d.similarity,
          extractorVersion: config.distill.extractorVersion, sourceTraceId: traceIds[0],
        });
        if (created) facts += 1;
      }
      doneIds.push(...rows.map((r) => r.id));
      audit({ event: "distill_thread_done", threadId: first.thread_id, traceIds, repaired,
        facts, ops, extractorVersion: config.distill.extractorVersion });
    } catch (err) {
      // fail-open：保留 undistilled，下一轮重试；蒸馏失败不能影响主链路。
      audit({ event: "distill_thread_failed", threadId: rows[0].thread_id, reason: err.message });
      console.warn(`[distill] thread=${rows[0].thread_id} 失败: ${err.message}`);
    }
  }

  await commitDistillation(doneIds);
  return { processed: doneIds.length, episodes, facts, ops,
    budget: budget?.status() ?? null };
}

/** 供 cron 显式触发：审计走 append-only 审计表，便于和主链路共用查询。 */
export async function recordDistillAudit({ traceId, threadId, tenantId, userId, node, event, decision, detail }) {
  await appendAudit({
    traceId, threadId, tenantId, userId, node, event, decision,
    riskFlags: [], detail,
  });
}
