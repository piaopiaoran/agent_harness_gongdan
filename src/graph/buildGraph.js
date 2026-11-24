import { END, START, StateGraph } from "@langchain/langgraph";
import { AgentState } from "./state.js";
import { createNodes } from "./nodes.js";

export const routeAfterInput = (s) => (s.verdict === "rejected" ? "reject" : "recallMemory");
export const routeAfterAgent = (s) => {
  if (["budget_exceeded", "model_error", "rejected"].includes(s.verdict)) return "reject";
  return s.verdict === "tool_calls" ? "toolGate" : "guardOutputNode";
};
export const routeAfterGate = (s) => {
  if (s.verdict === "needs_approval") return "humanApproval";
  if (s.verdict === "rejected") return "reject";
  return "executeTools";
};
export const routeAfterBudget = (s) => (s.verdict === "budget_exceeded" ? "reject" : "agent");

/**
 * 图结构（每条边都是一次「重新鉴权」的机会）：
 *
 *  START → guardInput ─deny→ reject ────────────────────────┐
 *                └pass→ recallMemory → retrieveRAG → assembleContext → agent
 *                                                              ├answer→ guardOutputNode ┐
 *                                                              └tool_calls→ toolGate    │
 *                                                                  ├deny→ reject         │
 *                                                                  ├escalate→ approval   │
 *                                                                  └allow→ executeTools  │
 *                                                                            ↓           │
 *                                                                       budgetGate ──┐   │
 *                                                                         └ok→ agent│   │
 *  guardOutputNode / reject → recordTurn → END ←───────────────────────────────────┘
 *
 * 面试点：安全不押在 system prompt 上，而是挂在节点与条件边上；
 * 换模型供应商不用改任何一条防线。
 */
export function buildGraph(deps) {
  const n = createNodes(deps);
  return new StateGraph(AgentState)
    .addNode("guardInput", n.guardInput)
    .addNode("recallMemory", n.recallMemory)
    .addNode("retrieveRAG", n.retrieveRAG)
    .addNode("assembleContext", n.assembleContext)
    .addNode("agent", n.agent)
    .addNode("toolGate", n.toolGate)
    .addNode("humanApproval", n.approval)
    .addNode("executeTools", n.executeTools)
    .addNode("budgetGate", n.budgetGate)
    .addNode("guardOutputNode", n.guardOutputNode)
    .addNode("reject", n.reject)
    .addNode("recordTurn", n.recordTurn)

    .addEdge(START, "guardInput")
    .addConditionalEdges("guardInput", routeAfterInput, { reject: "reject", recallMemory: "recallMemory" })
    .addEdge("recallMemory", "retrieveRAG")
    .addEdge("retrieveRAG", "assembleContext")
    .addEdge("assembleContext", "agent")
    .addConditionalEdges("agent", routeAfterAgent, {
      reject: "reject", toolGate: "toolGate", guardOutputNode: "guardOutputNode",
    })
    .addConditionalEdges("toolGate", routeAfterGate, {
      humanApproval: "humanApproval", reject: "reject", executeTools: "executeTools",
    })
    .addEdge("humanApproval", "executeTools")
    .addEdge("executeTools", "budgetGate")
    .addConditionalEdges("budgetGate", routeAfterBudget, { reject: "reject", agent: "agent" })
    .addEdge("guardOutputNode", "recordTurn")
    .addEdge("reject", "recordTurn")
    .addEdge("recordTurn", END)
    .compile({ checkpointer: deps.checkpointer });
}
