import { HumanMessage, ToolMessage, SystemMessage } from "@langchain/core/messages";
import { interrupt } from "@langchain/langgraph";
import { config, SYSTEM_PROMPT } from "../config.js";
import { invokeModel, extractText } from "../llm/model.js";
import { guardInput as guardInputFn, guardOutput as guardOutputFn } from "../security/guard.js";
import { authorizeTool, checkStepBudget } from "../security/rbac.js";
import { maskPII } from "../security/pii.js";
import { recallMemories } from "../memory/recall.js";
import { retrieveEvidence } from "../rag/retrieve.js";
import { assembleContext } from "../context/assemble.js";
import { recordTurn as recordWorkingTurn } from "../memory/working.js";
import { llmCallConfig } from "../observability/tracing.js";

/**
 * 协议完整性：OpenAI 兼容端点要求每个 assistant.tool_calls 都有对应 tool_call_id 的
 * ToolMessage 回应，**且回应必须紧跟在该 assistant 消息之后**。
 *
 * 只「在数组末尾补 ToolMessage」是不够的：如果历史里 AI(tool_calls) 与新 HumanMessage
 * 交错（interrupt 未审批就开新轮、旧脏 checkpoint 等），补出来的 ToolMessage 会排在
 * HumanMessage 后面，端点依然判为 insufficient tool messages。
 *
 * 这里重建消息顺序：按 tool_call_id 把已有 ToolMessage 归位到对应 AIMessage 之后，
 * 真正缺失的才补占位回应。返回 repairs 供 checkpoint 落库，逐步自愈历史。
 */
export function repairMessageSequence(messages, reason = "该工具调用未被执行") {
  const list = Array.isArray(messages) ? messages : [];

  // 先看全量历史，避免把「稍后才回应」的调用误判为悬空而重复补。
  const answers = new Map();
  for (const m of list) {
    if (m?.tool_call_id && !answers.has(m.tool_call_id)) answers.set(m.tool_call_id, m);
  }

  const out = [];
  const repairs = [];
  for (const m of list) {
    // ToolMessage 统一在归属 AIMessage 后面归位，原位置先跳过。
    if (m?.tool_call_id) continue;
    out.push(m);
    for (const c of m?.tool_calls ?? []) {
      if (!c?.id) continue;
      const answer = answers.get(c.id);
      if (answer) {
        out.push(answer);
      } else {
        const fix = new ToolMessage({
          content: `[未执行] ${reason}（tool: ${c.name ?? "unknown"}）`,
          tool_call_id: c.id,
        });
        out.push(fix);
        repairs.push(fix);
      }
    }
  }

  // 找不到归属 AIMessage 的 ToolMessage 原样保留，绝不丢历史。
  const placed = new Set(out);
  for (const m of list) {
    if (m?.tool_call_id && !placed.has(m)) out.push(m);
  }
  return { messages: out, repairs };
}

/** 兼容旧调用点：只返回需要补写的 ToolMessage。 */
export function unansweredToolReplies(messages, reason = "该工具调用未被执行") {
  return repairMessageSequence(messages, reason).repairs;
}

export function createNodes(deps) {
  const { auditor, identity, embedder, model, registry, queue, rateResult, costLedger } = deps;
  const emit = (...args) => auditor.emit(...args);

  return {
    // ---------------- L1 / L2 / L3 接入与输入护栏（fail-close）----------------
    async guardInput(state) {
      const raw = state.input?.raw ?? "";
      const g = guardInputFn(raw);
      await emit("guardInput", "rate_limit_check", "allow", [],
        { backend: rateResult?.backend ?? "unknown", remaining: rateResult?.remaining ?? null });
      await emit("guardInput", "input_guard", g.decision, g.flags,
        { injection: g.injection, piiHits: g.piiHits, truncated: g.truncated });
      return {
        input: { ...g, raw },
        verdict: g.decision === "deny" ? "rejected" : "ok",
        messages: [new HumanMessage(g.text)],
      };
    },

    // ---------------- 记忆召回：情景 / 语义分区，独立失败 ----------------
    async recallMemory(state) {
      try {
        const r = await recallMemories(embedder, {
          tenantId: identity.tenantId, userId: identity.userId, query: state.input.text,
        });
        await emit("recallMemory", "memory_recall", "allow", [],
          { episodes: r.episodes.length, facts: r.facts.length, trace: r.trace });
        return { memories: r };
      } catch (err) {
        await emit("recallMemory", "memory_recall_degraded", "degrade", ["memory_unavailable"], { reason: err.message });
        return { memories: { episodes: [], facts: [], trace: [] } };
      }
    },

    // ---------------- RAG 双路召回 + RRF + 精排（证据不足 fail-close）----------------
    async retrieveRAG(state) {
      try {
        const r = await retrieveEvidence(embedder, state.input.text, {
          tenantId: identity.tenantId, budgetTokens: config.CTX_RAG_TOKENS,
        });
        await emit("retrieveRAG", r.refused ? "rag_refused" : "rag_retrieved",
          r.refused ? "deny" : "allow", r.trace.degradations.map((d) => `${d.stage}:${d.policy}`),
          { stages: r.trace.stages, degradations: r.trace.degradations, evidence: r.evidence.length });
        return { rag: r };
      } catch (err) {
        await emit("retrieveRAG", "rag_error", "deny", ["rag_unavailable"], { reason: err.message });
        return { rag: { evidence: [], refused: true, reason: err.message, trace: { stages: [], degradations: [] }, usedTokens: 0 } };
      }
    },

    // ---------------- 上下文编排：共享窗口 + 软上限 + Headroom ----------------
    async assembleContext(state) {
      const a = assembleContext({
        systemPrompt: SYSTEM_PROMPT,
        agentState: {
          traceId: identity.traceId, tenantId: identity.tenantId, userId: identity.userId,
          role: identity.role, step: state.steps, verdict: state.verdict,
        },
        recentMessages: state.messages.slice(0, -1),
        memories: state.memories,
        ragEvidence: state.rag.evidence,
        remoteEpisodes: state.memories.episodes,
        question: state.input.text,
      });
      if (a.report.overflow) {
        await emit("assembleContext", "context_overflow", "degrade", ["context_overflow"], a.report);
      } else {
        await emit("assembleContext", "context_assembled", "allow", [], a.report);
      }
      return { contextReport: a.report, systemText: a.systemMessage.content };
    },

    // ---------------- 模型推理：只产出 tool_call「请求」，不拥有执行权 ----------------
    async agent(state) {
      // 最后一道协议防线：历史 checkpoint 可能来自修复前的版本（悬空 tool_calls），
      // 或 reject/executeTools 之外的分支。入模前按 tool_call_id 归位 + 补写。
      const healed = repairMessageSequence(state.messages, "该工具调用未被执行，请基于已有信息回答");
      if (healed.repairs.length) {
        await emit("agent", "tool_replies_repaired", "degrade", ["orphan_tool_call"], { count: healed.repairs.length });
      }
      const messages = [new SystemMessage(state.systemText || SYSTEM_PROMPT), ...healed.messages];
      let reply;
      try {
        // 每次 LLM 调用单独打归因标签；child run 不依赖父 run 的 metadata 继承。
        reply = await invokeModel(deps.model, messages, llmCallConfig(identity, { stage: "agent" }));
      } catch (err) {
        await emit("agent", "model_error", "deny", ["model_error"], { reason: err.message });
        return { messages: [], verdict: "model_error", refusalReason: err.message };
      }
      const text = extractText(reply);
      const calls = reply.tool_calls ?? [];
      const steps = state.steps + 1;
      const cost = costLedger.add(reply.usage_metadata, {
        threadId: identity.sessionId, tenantId: identity.tenantId,
        feature: identity.feature, steps,
      });
      await emit("agent", "model_invoke", "allow", [],
        { tools: calls.map((c) => c.name), textChars: text.length, steps, usage: reply.usage_metadata ?? null, cost });

      if (!calls.length) {
        return { messages: [reply, ...healed.repairs], finalAnswer: text, steps, verdict: "answer" };
      }
      const plan = calls.map((c) => ({
        id: c.id, name: c.name, args: c.args ?? {},
        auth: authorizeTool({ name: c.name, args: c.args ?? {}, role: identity.role }),
      }));
      return {
        messages: [reply, ...healed.repairs], toolPlan: plan, steps, verdict: "tool_calls",
        finalAnswer: text,
      };
    },

    // ---------------- L4 / L5 工具网关：策略判定（模型无权绕过）----------------
    async toolGate(state) {
      const plan = state.toolPlan.map((p) => ({
        ...p,
        auth: authorizeTool({ name: p.name, args: p.args, role: identity.role }),
      }));
      for (const p of plan) {
        await emit("toolGate", "tool_authorization", p.auth.decision,
          p.auth.decision === "deny" ? ["unauthorized_tool"] : [],
          { tool: p.name, reason: p.auth.reason, args: p.auth.safeArgs });
      }
      const decisions = plan.map((p) => p.auth.decision);
      const verdict = decisions.includes("escalate")
        ? "needs_approval"
        : decisions.every((d) => d === "deny") ? "rejected" : "gated";
      return { toolPlan: plan, verdict };
    },

    // ---------------- L5 人工审批：interrupt 前无任何副作用 ----------------
    async approval(state) {
      const pending = state.toolPlan.filter((p) => p.auth.decision === "escalate");
      const reply = interrupt({
        type: "human_approval_required",
        traceId: identity.traceId,
        sessionId: identity.sessionId,
        items: pending.map((p) => ({ tool: p.name, args: p.auth.safeArgs, reason: p.auth.reason })),
        expected: "{ decision: 'approved' | 'rejected', approverId, reason }",
      });
      // ↓ 仅在 resume 之后执行
      const approved = reply?.decision === "approved";
      await emit("humanApproval", "human_decision", approved ? "approve" : "reject", [], {
        approverId: reply?.approverId ?? "unknown",
        reason: reply?.reason ?? "",
        tools: pending.map((p) => p.name),
      });
      return {
        toolPlan: state.toolPlan.map((p) =>
          p.auth.decision === "escalate"
            ? { ...p, auth: { ...p.auth, decision: approved ? "allow" : "deny", reason: approved ? "approved_by_human" : "rejected_by_human" } }
            : p
        ),
        approval: { decision: approved ? "approved" : "rejected", approverId: reply?.approverId ?? "unknown", at: Date.now() },
        verdict: "approval_resolved",
      };
    },

    // ---------------- L6 受控执行：参数脱敏 + 输出截断 + 逐次审计 ----------------
    // 协议不变量：模型每发出一个 tool_call_id，都必须收到一条 ToolMessage 回应。
    // 之前只对 allow 的工具回应，deny/escalate-rejected 的调用悬空 → 下一轮重发历史时
    // DeepSeek/OpenAI 兼容端点返回 400 insufficient tool messages。现在所有分支都回应。
    async executeTools(state) {
      const toolMessages = [];
      const results = [];
      for (const p of state.toolPlan) {
        // 拒绝分支也必须回应，但绝不执行工具本身
        if (p.auth.decision !== "allow") {
          const reason = p.auth.reason || "not_authorized";
          const msg = `[网关拒绝] 工具 ${p.name} 未执行：${reason}`;
          results.push({ name: p.name, denied: true, reason });
          toolMessages.push(new ToolMessage({ content: msg, tool_call_id: p.id }));
          await emit("executeTools", "tool_execution_denied", "deny", ["unauthorized_tool"],
            { tool: p.name, reason });
          continue;
        }
        const t = registry.get(p.name);
        if (!t) {
          const msg = `[工具不可用] ${p.name} 未在注册表`;
          results.push({ name: p.name, error: msg });
          toolMessages.push(new ToolMessage({ content: msg, tool_call_id: p.id }));
          await emit("executeTools", "tool_missing", "degrade", ["tool_missing"], { tool: p.name });
          continue;
        }
        try {
          const out = await t.invoke(p.auth.safeArgs ?? p.args);
          const masked = maskPII(String(out)).masked.slice(0, config.MAX_TOOL_OUTPUT_CHARS);
          results.push({ name: p.name, args: p.auth.safeArgs, result: masked });
          toolMessages.push(new ToolMessage({ content: masked, tool_call_id: p.id }));
          await emit("executeTools", "tool_executed", "allow", [], { tool: p.name, outChars: masked.length });
        } catch (err) {
          const msg = `[工具失败] ${p.name}: ${err.message}`.slice(0, 500);
          results.push({ name: p.name, error: msg });
          toolMessages.push(new ToolMessage({ content: msg, tool_call_id: p.id }));
          await emit("executeTools", "tool_failed", "degrade", ["tool_error"], { tool: p.name, reason: err.message });
        }
      }
      // 兜底：toolPlan 为空但仍走出工具路径时，至少给一条占位回应
      if (!toolMessages.length) {
        toolMessages.push(new ToolMessage({ content: "[网关拒绝] 没有工具被授权执行", tool_call_id: state.toolPlan[0]?.id ?? "denied" }));
      }
      return { messages: toolMessages, toolResults: results, verdict: "tools_done" };
    },

    // ---------------- L5 轮次 / 墙钟预算（fail-close）----------------
    async budgetGate(state) {
      const { ok, reason } = checkStepBudget(state.steps, config.MAX_STEPS);
      if (!ok) {
        await emit("budgetGate", "budget_exceeded", "deny", ["max_steps"], { steps: state.steps });
        return { verdict: "budget_exceeded" };
      }
      return { verdict: "ok" };
    },

    // ---------------- L7 输出护栏：脱敏 + 强制引用 + 无证据拒答 ----------------
    async guardOutputNode(state) {
      const g = guardOutputFn(state.finalAnswer, {
        evidence: state.rag.evidence,
        flags: [...state.input.flags, ...(state.rag.trace?.degradations ?? []).map((d) => `${d.stage}:${d.policy}`)],
      });
      await emit("guardOutput", "output_guard", g.decision, g.flags, { notes: g.notes, answerChars: g.answer.length });
      return { finalAnswer: g.answer, verdict: "done", refusalReason: g.flags.join(",") };
    },

    // ---------------- 拒绝分支 ----------------
    async reject(state) {
      const answer = state.verdict === "budget_exceeded"
        ? "已达到最大工具调用轮次，请简化请求后重试。"
        : state.verdict === "model_error"
          ? `模型服务暂时不可用（${state.refusalReason ?? "timeout"}）。请稍后重试或联系人文客服。`
          : "该请求未通过安全策略校验，已被拒绝。请调整表述后重试。";
      await emit("reject", "request_rejected", "deny", [state.verdict], { reason: state.refusalReason ?? state.input.reason });
      // 协议完整性：toolGate 全 deny 时会跳过 executeTools，agent 产出的 tool_calls
      // 若悬空写进 checkpoint，下一轮重发历史就会触发 400。这里在离图前补齐回应。
      const extra = unansweredToolReplies(state.messages, "该请求未通过安全策略校验，工具未执行");
      if (extra.length) {
        await emit("reject", "tool_replies_repaired", "degrade", ["orphan_tool_call"], { count: extra.length });
      }
      return { messages: extra, finalAnswer: answer, verdict: "rejected" };
    },

    // ---------------- 工作记忆落库 + 异步蒸馏入队（不阻塞主请求）----------------
    async recordTurn(state) {
      if (!state.finalAnswer) return {};
      try {
        await recordWorkingTurn({
          traceId: identity.traceId, threadId: identity.sessionId,
          tenantId: identity.tenantId, userId: identity.userId, role: identity.role,
          messages: state.messages, finalAnswer: state.finalAnswer,
          toolCalls: state.toolResults ?? [],
        });
        // jobId = traceId：同一轮重试不会重复入队；真正去重落在 episode 幂等键。
        await queue.enqueue("distill", { tenantId: identity.tenantId }, 10, `distill:${identity.traceId}`);
        await emit("recordTurn", "working_memory_appended", "allow", [], {
          messages: state.messages.length, distillQueued: true,
        });
        return { distillQueued: true };
      } catch (err) {
        await emit("recordTurn", "working_memory_failed", "degrade", ["storage_error"], { reason: err.message });
        return {};
      }
    },
  };
}
