import { appendWorkingMemory, listUndistilled, markDistilled } from "../db/queries.js";

/** 工作记忆 = 会话原始流水：每轮追加，只供审计与离线蒸馏，绝不直接注入 prompt。 */
export async function recordTurn({ traceId, threadId, tenantId, userId, role, messages, finalAnswer, toolCalls = [] }) {
  const rows = [];
  for (const m of messages) {
    const kind = m.getType?.() ?? m._getType?.() ?? "generic";
    if (!["human", "ai", "tool"].includes(kind)) continue;
    rows.push({
      traceId, threadId, tenantId, userId, role,
      kind,
      content: typeof m.content === "string" ? m.content : JSON.stringify(m.content ?? ""),
      toolCalls,
    });
  }
  if (finalAnswer) {
    rows.push({ traceId, threadId, tenantId, userId, role, kind: "ai", content: finalAnswer, toolCalls });
  }
  for (const row of rows) await appendWorkingMemory(row);
  return rows.length;
}

export async function pendingForTenant(tenantId, limit) {
  return listUndistilled(tenantId, limit);
}

export async function commitDistillation(ids) {
  await markDistilled(ids);
}
