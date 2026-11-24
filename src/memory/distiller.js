import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { config, getSecret } from "../config.js";
import { invokeModel, createFakeModel } from "../llm/model.js";
import { pendingForTenant, commitDistillation } from "./working.js";
import { insertEpisode, upsertFact } from "../db/queries.js";

const DISTILL_SYSTEM = `你把售后工单会话流水压缩成结构化记忆。只输出 JSON，不要解释：
{
  "goal": "用户此轮要解决的问题（一句话）",
  "summary": "过程与关键事实（80字内）",
  "outcome": "结果：resolved | unresolved | escalated",
  "failure_points": ["失败点或待办，没有则空数组"],
  "facts": [{"fact": "可复用的稳定事实或偏好", "category": "preference|order|business_rule"}]
}
注意：facts 只能来自原始消息，不能把摘要再当事实。没有稳定事实就返回空数组。`;

function getDistillModel() {
  if (!config.hasLLM) return createFakeModel();
  return new ChatOpenAI({
    model: config.LLM_MODEL,
    apiKey: getSecret("llm"),
    temperature: 0.1,
    // reasoning 模型会先吃一大段思考 token，给足预算才能拿到完整 JSON content
    maxTokens: 2048,
    timeout: config.LLM_TIMEOUT_MS,
    maxRetries: 1,
    configuration: { baseURL: config.LLM_BASE_URL },
  });
}

/** 蒸馏只认真实 content；reasoning_content 里可能出现伪 JSON，绝不能拿来解析。 */
function rawContent(message) {
  const raw = message?.content;
  if (typeof raw === "string") return raw;
  if (Array.isArray(raw)) return raw.map((b) => (typeof b === "string" ? b : b?.text ?? "")).join("");
  return "";
}

function parseJson(text) {
  const match = String(text).match(/\{[\s\S]*\}/);
  if (!match) throw new Error("distill_output_not_json");
  return JSON.parse(match[0]);
}

/**
 * 离线蒸馏：消费未蒸馏原始消息 → LLM 压缩 episode + 抽取事实 → 向量化写 PG → 标记已蒸馏。
 * 全程异步，不阻塞主请求。任何一步失败都保留 distilled=false，下一轮自动重试（fail-open）。
 */
export async function distillOnce(embedder, { tenantId, batchSize = config.DISTILL_BATCH_SIZE } = {}) {
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
  const doneIds = [];

  for (const rows of byThread.values()) {
    try {
      const raw = rows.map((r) => `${r.kind}: ${r.content}`).join("\n");
      const reply = await invokeModel(model, [
        new SystemMessage(DISTILL_SYSTEM),
        new HumanMessage(raw.slice(0, 6000)),
      ]);
      const data = parseJson(rawContent(reply));
      const traceIds = [...new Set(rows.map((r) => r.trace_id))];
      const first = rows[0];

      const episodeText = `${data.goal ?? ""} ${data.summary ?? ""}`.trim();
      await insertEpisode({
        tenantId, userId: first.user_id, threadId: first.thread_id,
        goal: String(data.goal ?? "售后咨询").slice(0, 200),
        summary: String(data.summary ?? episodeText).slice(0, 600),
        outcome: String(data.outcome ?? "unresolved").slice(0, 40),
        failurePoints: Array.isArray(data.failure_points) ? data.failure_points.slice(0, 8).map(String) : [],
        sourceTraceIds: traceIds,
        embedding: await embedder.embedQuery(episodeText),
      });
      episodes += 1;

      for (const f of (Array.isArray(data.facts) ? data.facts : []).slice(0, 6)) {
        const fact = String(f?.fact ?? "").trim();
        if (!fact || fact.length < 4) continue;
        await upsertFact({
          tenantId, userId: first.user_id, fact: fact.slice(0, 200),
          category: ["preference", "order", "business_rule"].includes(f.category) ? f.category : "preference",
          confidence: 0.75, sourceTraceId: traceIds[0],
          embedding: await embedder.embedQuery(fact),
        });
        facts += 1;
      }
      doneIds.push(...rows.map((r) => r.id));
    } catch (err) {
      // fail-open：保留 undistilled，下一轮重试；蒸馏失败不能影响主链路。
      console.warn(`[distill] thread=${rows[0].thread_id} 失败: ${err.message}`);
    }
  }

  await commitDistillation(doneIds);
  return { processed: doneIds.length, episodes, facts };
}
