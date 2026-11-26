import { readFile, readdir } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { maskPII } from "./security/pii.js";
import { insertRefundTicket } from "./db/queries.js";
import { recallMemories } from "./memory/recall.js";

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = join(here, "..", "data");
let orderCache = null;

async function orders() {
  if (!orderCache) {
    orderCache = JSON.parse(await readFile(join(dataDir, "orders.json"), "utf8"));
  }
  return orderCache;
}

/** 工具注册表：risk 决定 L5/L6 的审批与拒绝策略，模型只能看到 allowlist 里的描述。 */
export const TOOL_META = {
  get_order_status: { risk: "low", roles: ["user", "admin"], desc: "查询指定订单的状态与关键信息" },
  get_refund_policy: { risk: "low", roles: ["user", "admin"], desc: "查询退换货 / 保修政策条款" },
  create_refund_ticket: { risk: "high", roles: ["user", "admin"], desc: "创建退款工单（资金动作，必须人工审批）" },
  send_email: { risk: "high", roles: ["admin"], desc: "对外发送邮件（需管理员 + 人工审批）" },
  export_all_orders: { risk: "forbidden", roles: [], desc: "导出全量订单（任何角色禁止）" },
  recall_memory: { risk: "low", roles: ["user", "admin"], desc: "按需检索用户历史对话情景与稳定事实" },
};

export function createTools({ identity, traceId, embedder }) {
  const getOrderStatus = tool(
    async ({ orderId }) => {
      const all = await orders();
      const row = all.find((o) => o.orderId === orderId && o.tenantId === identity.tenantId);
      if (!row) return JSON.stringify({ error: "订单不存在或无权访问" });
      // 工具输出也做 PII 脱敏：客服场景不需要完整手机号 / 地址。
      const safe = maskPII(JSON.stringify({
        orderId: row.orderId, item: row.item, amount: row.amount, status: row.status,
        purchasedAt: row.purchasedAt, issue: row.issue, phone: row.phone, address: row.address,
      }));
      return safe.masked;
    },
    {
      name: "get_order_status",
      description: TOOL_META.get_order_status.desc,
      schema: z.object({ orderId: z.string().describe("订单号，如 ORD-1001") }),
    }
  );

  const getRefundPolicy = tool(
    async ({ topic }) => {
      const files = (await readdir(join(dataDir, "docs"))).filter((f) => f.endsWith(".md"));
      const terms = String(topic).match(/[\u4e00-\u9fa5]{2,}|[a-zA-Z0-9]+/g) ?? [];
      const scored = [];
      for (const f of files) {
        const body = await readFile(join(dataDir, "docs", f), "utf8");
        const lines = body.split(/\n+/).map((s) => s.trim()).filter((s) => s.length > 12 && !s.startsWith("#"));
        for (const line of lines) {
          const hit = terms.filter((t) => line.includes(t)).length;
          if (hit > 0) scored.push({ source: f, hit, line });
        }
      }
      scored.sort((a, b) => b.hit - a.hit);
      return JSON.stringify({ matches: scored.slice(0, 3).map((s) => ({ source: s.source, text: s.line })) });
    },
    {
      name: "get_refund_policy",
      description: TOOL_META.get_refund_policy.desc,
      schema: z.object({ topic: z.string().describe("政策主题关键词") }),
    }
  );

  const createRefundTicket = tool(
    async ({ orderId, amount, reason }) => {
      // 只有审批节点放行后才会走到这里；这里再校验一次，避免状态被跳过。
      const row = await insertRefundTicket({
        tenantId: identity.tenantId, orderId, userId: identity.userId,
        amount: Number(amount) || 0, reason: String(reason).slice(0, 200),
        approverId: identity.approverId ?? "pending", traceId,
      });
      return JSON.stringify({ ticketId: row.id, status: row.status, note: "已登记，等待履约系统处理" });
    },
    {
      name: "create_refund_ticket",
      description: TOOL_META.create_refund_ticket.desc,
      schema: z.object({
        orderId: z.string(), amount: z.coerce.number().nonnegative(), reason: z.string(),
      }),
    }
  );

  const sendEmail = tool(
    async ({ to, subject, body }) => JSON.stringify({ to: maskPII(to).masked, subject, body: maskPII(body).masked, status: "queued" }),
    {
      name: "send_email",
      description: TOOL_META.send_email.desc,
      schema: z.object({ to: z.string(), subject: z.string(), body: z.string() }),
    }
  );

  const exportAllOrders = tool(
    async () => JSON.stringify({ error: "export_all_orders 已被安全策略永久禁止" }),
    {
      name: "export_all_orders",
      description: TOOL_META.export_all_orders.desc,
      schema: z.object({ format: z.string().optional() }),
    }
  );

  // memory-as-a-tool：Agent 自己判断何时需要历史记忆，省掉每轮无条件两路向量检索。
  // tenant/user 从可信 identity 注入，不接收模型传入的 tenant（防横向越权）。
  const recallMemoryTool = tool(
    async ({ query }) => {
      if (!embedder) return JSON.stringify({ episodes: [], facts: [], note: "embedder_unavailable" });
      const r = await recallMemories(embedder, {
        tenantId: identity.tenantId, userId: identity.userId,
        query: String(query ?? "").slice(0, 500),
      });
      return JSON.stringify({
        episodes: r.episodes.map((e) => ({ id: e.id, text: e.text, similarity: e.similarity })),
        facts: r.facts.map((f) => ({ id: f.id, text: f.text, similarity: f.similarity, confidence: f.confidence })),
      });
    },
    {
      name: "recall_memory",
      description: TOOL_META.recall_memory.desc + "（仅当前用户，入参为检索问题）",
      schema: z.object({ query: z.string().min(1).describe("要回忆的问题，例如：我上次说的偏好是什么") }),
    }
  );

  // 模型只看到白名单工具；forbidden 工具存在于注册表但绝不 bind 给模型，
  // 用来验证「即使模型幻觉出一个不在白名单里的工具，网关依然会拒绝」。
  const langchainTools = [getOrderStatus, getRefundPolicy, createRefundTicket, sendEmail, recallMemoryTool];
  const registry = new Map(
    [getOrderStatus, getRefundPolicy, createRefundTicket, sendEmail, exportAllOrders, recallMemoryTool]
      .map((t) => [t.name, t])
  );
  return { registry, langchainTools, names: () => [...registry.keys()] };
}
