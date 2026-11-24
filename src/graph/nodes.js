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

export function createNodes(deps) {
  const { auditor, identity, embedder, model, registry, queue, rateResult } = deps;
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
      const messages = [new SystemMessage(state.systemText || SYSTEM_PROMPT), ...state.messages];
      let reply;
      try {
        reply = await invokeModel(deps.model, messages);
      } catch (err) {
        await emit("agent", "model_error", "deny", ["model_error"], { reason: err.message });
        return { messages: [], verdict: "model_error", refusalReason: err.message };
      }
      const text = extractText(reply);
      const calls = reply.tool_calls ?? [];
      const steps = state.steps + 1;
      await emit("agent", "model_invoke", "allow", [],
        { tools: calls.map((c) => c.name), textChars: text.length, steps, usage: reply.usage_metadata ?? null });

      if (!calls.length) {
        return { messages: [reply], finalAnswer: text, steps, verdict: "answer" };
      }
      const plan = calls.map((c) => ({
        id: c.id, name: c.name, args: c.args ?? {},
        auth: authorizeTool({ name: c.name, args: c.args ?? {}, role: identity.role }),
      }));
      return {
        messages: [reply], toolPlan: plan, steps, verdict: "tool_calls",
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
    async executeTools(state) {
      const toolMessages = [];
      const results = [];
      for (const p of state.toolPlan) {
        if (p.auth.decision !== "allow") continue;
        const t = registry.get(p.name);
        if (!t) continue;
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
      return { finalAnswer: answer, verdict: "rejected" };
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
        await queue.enqueue("distill", { tenantId: identity.tenantId }, 10);
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
