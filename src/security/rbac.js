import { TOOL_META } from "../tools.js";

/**
 * L5/L6 工具网关：Casbin 风格的 (subject, object, action) 策略，扣在 LangGraph 节点上。
 * 关键立场：模型返回 tool_call 只是「请求」，不是授权；执行权完全在这里。
 * 即使模型幻觉出一个未注册工具，也会在这里被拒绝。
 */
export function authorizeTool({ name, args, role }) {
  const meta = TOOL_META[name];
  if (!meta) {
    return { decision: "deny", reason: "tool_not_in_allowlist", safeArgs: args };
  }
  if (meta.risk === "forbidden" || !meta.roles.includes(role)) {
    return { decision: "deny", reason: `role_${role}_not_allowed`, safeArgs: args };
  }

  // JSON Schema 强校验的基线：参数类型与业务边界在这里二次确认。
  if (name === "get_order_status" && !/^ORD-\d{4}$/.test(String(args?.orderId ?? ""))) {
    return { decision: "deny", reason: "invalid_order_id", safeArgs: args };
  }
  if (name === "create_refund_ticket") {
    const amount = Number(args?.amount);
    if (!Number.isFinite(amount) || amount < 0 || amount > 5000) {
      return { decision: "deny", reason: "amount_out_of_range", safeArgs: args };
    }
  }

  if (meta.risk === "high") {
    return { decision: "escalate", reason: "high_risk_requires_human_approval", safeArgs: args };
  }
  return { decision: "allow", reason: "rbac_allow", safeArgs: args };
}

/** L5 规划防护：限制总工具调用轮次，防止无限循环刷接口。 */
export function checkStepBudget(steps, maxSteps) {
  if (steps >= maxSteps) return { ok: false, reason: "max_tool_rounds_exceeded" };
  return { ok: true };
}
