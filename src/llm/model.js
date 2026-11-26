import { randomUUID } from "node:crypto";
import { ChatOpenAI } from "@langchain/openai";
import { AIMessage } from "@langchain/core/messages";
import { config, getSecret } from "../config.js";

/** DeepSeek-reasoning 系列会把思考放在 reasoning_content；content 可能为空或数组块。 */
export function extractText(message) {
  const raw = message?.content;
  let text = "";
  if (typeof raw === "string") text = raw;
  else if (Array.isArray(raw)) {
    text = raw.map((b) => (typeof b === "string" ? b : b?.text ?? "")).join("");
  }
  if (!text.trim()) {
    const extra = message?.additional_kwargs?.reasoning_content;
    if (extra) text = `[思考摘要] ${String(extra).slice(0, 200)}`;
  }
  return text.trim();
}

/**
 * fake 模型：规则驱动、确定性，用来验证「安全是网关的事，不是模型自觉」。
 * 它故意会对高危请求直接发起工具调用，由 L4/L5 网关拦截。
 */
export function createFakeModel() {
  const model = {
    bindTools() { return model; },
    async invoke(messages) {
      const last = messages[messages.length - 1];
      const kind = (m) => m?.getType?.() ?? m?._getType?.() ?? (m?.tool_call_id ? "tool" : "generic");

      // 蒸馏任务：确定性 JSON，保证 fake 模式下记忆层也能跑通
      const isDistill = messages.some((m) => extractText(m).includes("结构化记忆"));
      if (isDistill) {
        return new AIMessage({
          id: `ai-${randomUUID()}`,
          content: JSON.stringify({
            goal: "处理售后咨询与订单问题",
            summary: "用户描述了商品问题并询问可采取的处理方式。",
            outcome: "unresolved",
            failure_points: [],
            facts: [],
          }),
        });
      }

      if (kind(last) === "tool") {
        const summary = extractText(last).split("\n")[0].slice(0, 160);
        return new AIMessage({
          id: `ai-${randomUUID()}`,
          content: summary.startsWith("[网关拒绝]")
            ? `该操作被安全网关拒绝：${summary.replace("[网关拒绝] ", "")}`
            : `依据已授权工具结果：${summary}。政策条款见 [证据1]。`,
        });
      }

      const human = [...messages].reverse().find((m) => kind(m) === "human");
      const text = String(human?.content ?? "");
      const call = (name, args) => ({ name, args, id: `call-${randomUUID()}`, type: "tool_call" });
      let tool_calls = [];

      if (/导出|全部订单|所有用户/.test(text)) {
        tool_calls = [call("export_all_orders", { format: "csv" })];
      } else if (/邮件|发送.*给/.test(text)) {
        tool_calls = [call("send_email", { to: "ops@example.com", subject: "工单", body: text.slice(0, 80) })];
      } else if (/退款|退钱|退费/.test(text) && !/政策|规定|怎么|能否|可以吗/.test(text)) {
        const id = text.match(/ORD-\d{4}/)?.[0] ?? "ORD-1001";
        tool_calls = [call("create_refund_ticket", { orderId: id, amount: 299, reason: text.slice(0, 80) })];
      } else if (/政策|规定|保修|多久|几天|手册|怎么|退货|换新|条件|时效|无理由/.test(text)) {
        tool_calls = [call("get_refund_policy", { topic: text.slice(0, 80) })];
      } else if (/ORD-\d{4}|订单|物流|进度/.test(text)) {
        const id = text.match(/ORD-\d{4}/)?.[0] ?? "ORD-1001";
        tool_calls = [call("get_order_status", { orderId: id })];
      }

      return new AIMessage({
        id: `ai-${randomUUID()}`,
        content: tool_calls.length ? "" : "我已收到你的售后请求，请在授权范围内描述订单号与问题。",
        tool_calls,
        usage_metadata: { total_tokens: 60 },
      });
    },
  };
  return model;
}

/**
 * 真实模型：OpenAI 兼容 /chat/completions（DeepSeek / 硅基流动 / OpenAI 通用）。
 * 换供应商只改 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY 三个环境变量，
 * 安全 / RAG / 记忆 / 上下文六层代码一行都不用动。
 */
export function createModel({ tools = [] } = {}) {
  if (!config.hasLLM) return createFakeModel();
  const llm = new ChatOpenAI({
    model: config.LLM_MODEL,
    apiKey: getSecret("llm"),
    temperature: config.LLM_TEMPERATURE,
    maxTokens: config.LLM_MAX_TOKENS,
    timeout: config.LLM_TIMEOUT_MS,
    maxRetries: config.LLM_MAX_RETRIES,
    configuration: { baseURL: config.LLM_BASE_URL },
  });
  return tools.length ? llm.bindTools(tools) : llm;
}

/** 带硬超时的调用：Worker 直连模型，本地超时 / 重试 / 熔断是最后一环。 */
export async function invokeModel(model, messages, callConfig = {}) {
  let timer;
  try {
    return await Promise.race([
      model.invoke(messages, callConfig),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`model_timeout_${config.LLM_TIMEOUT_MS}ms`)), config.LLM_TIMEOUT_MS + 2000);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}
