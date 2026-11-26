import test from "node:test";
import assert from "node:assert/strict";
import { config } from "../src/config.js";
import { decideOperation, parseDecision, buildDecisionPrompt, FACT_OPS } from "../src/memory/consolidate.js";
import { DistillOutput, parseDistillOutput, episodeIdempotencyKey } from "../src/memory/distiller.js";
import { decayedSimilarity, applyForgetting } from "../src/memory/forget.js";
import { createTokenBudget } from "../src/observability/cost.js";
import { createPriorityQueue } from "../src/queue/priority.js";
import { routeAfterInput } from "../src/graph/buildGraph.js";

test("整合四操作：无相似→ADD，完全相同→NOOP，近似→UPDATE，LLM 决策 DELETE 生效", () => {
  const fact = "用户偏好左侧耳机";
  assert.equal(decideOperation({ fact, similar: [] }).op, "ADD");
  assert.equal(decideOperation({ fact, similar: [{ id: "a", fact, similarity: 0.97 }] }).op, "NOOP");
  const upd = decideOperation({ fact, similar: [{ id: "a", fact: "用户偏好右侧耳机", similarity: 0.9 }] });
  assert.equal(upd.op, "UPDATE");
  assert.equal(upd.targetId, "a");
  const del = decideOperation({
    fact, similar: [{ id: "a", fact: "用户偏好右侧耳机", similarity: 0.85 }],
    decision: { op: "DELETE", targetId: "a", reason: "用户撤回" },
  });
  assert.equal(del.op, "DELETE");
  assert.deepEqual([...FACT_OPS], ["ADD", "UPDATE", "DELETE", "NOOP"]);
});

test("整合决策解析：非法 JSON / 非白名单操作降级为 ADD，不污染库", () => {
  assert.equal(parseDecision("garbage").op, "ADD");
  assert.equal(parseDecision('{"op":"DROP"}').op, "ADD");
  const ok = parseDecision('{"op":"NOOP","targetId":"x","reason":"等价"}');
  assert.equal(ok.op, "NOOP");
  assert.equal(ok.targetId, "x");
  assert.match(buildDecisionPrompt({ fact: "新事实", similar: [{ id: "a", fact: "旧事实", similarity: 0.9 }] }), /ADD|UPDATE|DELETE|NOOP/);
});

test("结构化输出：zod 补全默认值，拒绝越界字段，解析失败可诊断", () => {
  const out = parseDistillOutput('{"goal":"g","summary":"s","outcome":"resolved","failure_points":[],"facts":[{"fact":"用户偏好左侧耳机"}]}');
  assert.equal(out.facts[0].category, "preference");
  assert.equal(out.facts[0].source_kind, "user_said");
  assert.equal(out.facts[0].confidence, 0.75);
  assert.throws(() => DistillOutput.parse({ facts: [{ fact: "ab", category: "unknown" }] }));
  assert.throws(() => parseDistillOutput("not json"), /not_json/);
});

test("episode 幂等键：同批消息稳定，换一条消息就变", () => {
  const rows = [{ id: 2 }, { id: 1 }];
  const a = episodeIdempotencyKey({ tenantId: "t", userId: "u", threadId: "s", rows });
  const b = episodeIdempotencyKey({ tenantId: "t", userId: "u", threadId: "s", rows: [...rows].reverse() });
  const c = episodeIdempotencyKey({ tenantId: "t", userId: "u", threadId: "s", rows: [{ id: 1 }] });
  assert.equal(a, b);
  assert.notEqual(a, c);
  assert.match(a, /^[0-9a-f]{40}$/);
});

test("遗忘：半衰期衰减按年龄折半，过旧项被过滤", () => {
  const now = Date.parse("2026-11-26T00:00:00Z");
  const fresh = { text: "new", similarity: 0.9, createdAt: "2026-11-26T00:00:00Z" };
  const old = { text: "old", similarity: 0.9, createdAt: "2026-11-20T00:00:00Z" };
  // 70 天前恰好折半；30 天按 0.5^(30/70) 衰减。
  assert.ok(Math.abs(decayedSimilarity(0.9, "2026-09-17T00:00:00Z", { now, halfLifeDays: 70 }) - 0.45) < 0.01);
  assert.ok(decayedSimilarity(0.9, "2026-10-27T00:00:00Z", { now, halfLifeDays: 70 }) > 0.45);
  const out = applyForgetting([fresh, old], { now, halfLifeDays: 1, minSimilarity: 0.5 });
  assert.equal(out.length, 1);
  assert.equal(out[0].text, "new");
  assert.equal(out[0].rawSimilarity, 0.9);
});

test("token 预算：按日累计、拒绝超额、次日自动清零", () => {
  let today = Date.parse("2026-11-26T10:00:00Z");
  const b = createTokenBudget({ dailyLimit: 100, now: () => today });
  assert.equal(b.canSpend(60), true);
  b.record({ total_tokens: 60 });
  assert.equal(b.canSpend(50), false);
  assert.equal(b.status().usedTokens, 60);
  today = Date.parse("2026-11-27T10:00:00Z");
  assert.equal(b.canSpend(60), true);
  assert.equal(b.status().usedTokens, 0);
});

test("队列可靠性：ack 移除 in-flight；fail 重试两次后进死信；jobId 去重", async () => {
  const q = createPriorityQueue({ forceMemory: true, startWorker: false, maxAttempts: 2, backoffMs: 1 });
  try {
    await q.enqueue("distill", { n: 1 }, 10, "job-1");
    const dup = await q.enqueue("distill", { n: 1 }, 10, "job-1");
    assert.equal(dup.duplicate, true);
    const item = await q.dequeue();
    assert.equal(item.task.type, "distill");
    assert.equal(await q.fail(item, new Error("boom")), "retry");
    await new Promise((r) => setTimeout(r, 5)); // 等 backoff 到期
    const retry = await q.dequeue();
    assert.equal(retry.task.attempts, 1);
    assert.equal(await q.fail(retry, new Error("boom")), "dead");
    assert.equal((await q.deadLetters()).length, 1);
    assert.equal(await q.size(), 0);
    assert.equal(q.stats().duplicate, 1);
    assert.equal(q.stats().dead, 1);
    await q.enqueue("distill", { n: 2 });
    const ok = await q.dequeue();
    await q.ack(ok);
    assert.equal(q.stats().acked, 1);
  } finally {
    await q.stop();
  }
});

test("召回策略：默认 tool 模式跳过无条件召回节点", async () => {
  assert.equal(config.memory.recallMode, "tool");
  assert.equal(routeAfterInput({ verdict: "ok" }), "retrieveRAG");
  assert.equal(routeAfterInput({ verdict: "rejected" }), "reject");
});
