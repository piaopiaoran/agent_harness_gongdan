# Token 消耗地图 · agent_harness_gongdan

> 说明：本文只做事实梳理，不改代码。
> 数据来源：`src/` 源码逐行核对 + `.env` 实际生效值 + `agent_harness_gongdan_audit_events` 审计表实测记录。
> 核对时间：2026-11-26

---

## 一、一句话结论

**整个系统只有一个真正计费的入口：`src/graph/nodes.js` 的 `agent` 节点里那一次 `ChatOpenAI.invoke`。**
其它所有模块（RAG、记忆、上下文、工具、安全）都不直接花钱，它们只决定这一次 invoke 的 `input_tokens` 有多大。

所以「token 消耗」在本项目里等价于三个变量：

```
input_tokens  = systemText（组装后的系统提示）+ 全部历史 messages + 工具 schema
output_tokens = 回答正文 + reasoning_content
调用次数       = agent 循环轮数（1 ~ MAX_STEPS=14）
```

单轮成本 ≈ `轮数 × input_tokens`，而不是「一次请求多少钱」。这是 agent 系统与普通 RAG 最大的成本结构差异。

---

## 二、实测基线（真实数据，非估算）

最近 6 轮真实调用（deepseek-flash）：

```
轮次   input_tokens   output_tokens   reasoning   cache_read
1       12,894          389             28          0
2       13,273          368             39          0
3       13,637          402             28          0
4       14,027          370             25          0
5       14,387          375             22          0
6       14,754          304             29          0
```

由此得到三个基线指标：

- **输入输出比 ≈ 48.5 : 1**（14,754 / 304），健康 agent 系统通常 10:1 ~ 30:1，本项目明显偏「输入重」。
- **每轮输入增长 ≈ +370 token**，来自未压缩的对话历史线性累积。
- **prompt cache 命中率 = 0%**，14.7K 前缀每轮按全价重算。

而系统自己算出来的上下文报告，和实际账单差了一倍多：

```
审计事件 context_assembled 的 report.total :  6,245
provider 实际 input_tokens               : 14,754
误差                                     : +136%
```

`report.total = {system:203, state:39, recent:4000, memory:603, rag:1026, remote:374}`
这组数字看起来非常健康（6.2K 远低于 usable 20.8K），但真实 prompt 是 14.7K。**也就是说上下文溢出检测永远为 false，等于没有。**

---

## 三、输入侧：配额是怎么定的

参数都在 `src/config.js:47-68`，`.env` 已全部显式覆盖。

**窗口与预留**

- `CONTEXT_WINDOW_TOKENS = 32000`（config.js:47）
- `CONTEXT_HEADROOM_RATIO = 0.35`（config.js:48）
- 由 config.js:174-187 换算出：`headroomTokens = 11200`，`usableTokens = 20800`

**六个区块的软上限（config.js:188-197，合计 10,500）**

- `system 2000`：系统指令，不压缩、不裁剪（assemble.js:44-46）
- `state 500`：Agent State 紧凑 JSON，`stateText.slice(0, B.state * 4)` 按字符折算（assemble.js:49-51）
- `recent 4000`：近程轨迹，**只统计不裁剪**（见第五节缺口 1）
- `memory 1500`：检索记忆，按相似度从低到高丢弃（assemble.js:71-86）
- `rag 1500`：检索证据，最多 6 块（assemble.js:89-101）
- `remote 1000`：远端摘要，有损截断 `slice(0, B.remote * 2)`（assemble.js:104-110）

**token 估算函数（不是真 tokenizer）**

`src/rag/segment.js:68-72`：

```
estimateTokens(text) = ceil(汉字数 + 非汉字字符数 / 4)
```

即「一个汉字算 1 token、4 个英文字符算 1 token」。这是字数近似，与 DeepSeek 真实 tokenizer 有偏差，全项目所有预算判断都建立在这个函数上。

**两个没被配额覆盖的隐性输入（实测约 8.5K）**

- 完整历史 `state.messages`（`nodes.js:125` 全量传入，`nodes.js:149` 原样拼进 messages）
- OpenAI 格式的工具 schema（`tools.js` 的 5 个工具，含 description / parameters / 枚举）
- LangChain 消息封装与 tool_call 的 JSON 结构开销

实测规模参考：`SYSTEM_PROMPT` 全文 203 token（config.js:234）、`TOOL_META` 原始 JSON 154 token，而实际发送时按 OpenAI tools 格式展开后会更大。

---

## 四、逐文件触点清单

### 4.1 上下文组装（最核心）

`src/context/assemble.js`

- 44-46：系统指令区域，`estimateTokens(system)` 记账，不裁剪
- 49-51：Agent State，`slice(0, B.state * 4)` 字符截断
- 53-66：近程轨迹，`slice(0, 4000)` 单条消息截断 + 超限时**只写 trimming 记录、不改数据**
- 70-86：记忆，`if (memTokens + t > B.memory) continue`，按相似度低者先丢
- 88-101：RAG 证据，`if (ragTokens + t > B.rag) continue`
- 104-110：远端摘要，`slice(0, B.remote * 2)`
- 113-118：最终 `contextText` 顺序 = system → state → 证据 → 记忆 → 远端摘要

`src/graph/nodes.js`

- 120-136：`assembleContext` 调用，把 `state.messages.slice(0, -1)` 作为 recentMessages；溢出时 emit `context_overflow`
- 149：`messages = [new SystemMessage(state.systemText), ...healed.messages]` —— **这一行才是真实 prompt 的组装点**，历史完全无上限

### 4.2 RAG

- `src/config.js:56` `RAG_CHUNK_TOKENS = 512`，`RAG_CHUNK_OVERLAP = 0.15`（ingest.js:12-32 按段切块）
- `src/rag/ingest.js:66` 入库字段 `tokens: segment(v.content).length`，是**分词后的词数**，不是模型 token
- `src/config.js:62` `RAG_EVIDENCE_MAX = 6`、`RAG_EVIDENCE_CHUNK_TOKENS = 800`
- `src/rag/retrieve.js:114-136`：单块 `slice(0, 800 * 2)` 字符、总数受 `budgetTokens = CTX_RAG_TOKENS = 1500` 限制
  ⚠️ 不自洽：6 块 × 800 token = 4,800 的名义上限，被 1,500 的总预算实际压在 1–2 块
- `src/rag/retrieve.js:103`：`RAG_RERANK_TOPN = 20` 进精排，精排本身不花模型 token（启发式替身），接真 cross-encoder 后会产生额外调用成本

### 4.3 记忆

- `src/memory/recall.js:45-52`：远端摘要按 `maxChars = CTX_REMOTE_TOKENS * 2` 截字符
- `src/memory/distiller.js:19-31`：独立 `ChatOpenAI`，`maxTokens: 2048`，`temperature 0.1`
- `src/graph/nodes.js:315`：`recordTurn` 里入队 `distill`，`DISTILL_BATCH_SIZE = 8`（config.js:71）
  ⚠️ 这条调用发生在主链之外，**不计入当轮 usage，也不在这条 turn trace 树里**

### 4.4 模型调用

- `src/llm/model.js:94-103`：`ChatOpenAI({ maxTokens: LLM_MAX_TOKENS })`，`.env` 实际 3072
- `src/llm/model.js:107-120`：`invokeModel` 用 `Promise.race` 加硬超时（`LLM_TIMEOUT_MS = 30000`）
- `LLM_MAX_RETRIES = 2`：重试产生的 token **全部重复计费**，且没有 `retry_idx` 记账
- `src/llm/model.js:80`：fake 模型手写 `usage_metadata: { total_tokens: 60 }`，只有总数、无 input/output 拆分

### 4.5 安全截断（间接影响 token）

- `src/security/patterns.js:11`：用户输入 `slice(0, MAX_INPUT_CHARS = 8000)` 字符 ≈ 2,000+ token 上限
- `src/graph/nodes.js:250`：工具结果 `maskPII` 后 `slice(0, MAX_TOOL_OUTPUT_CHARS = 6000)` ≈ 1,500 token 上限
- `src/security/guard.js:31`：最终回答 `slice(0, MAX_OUTPUT_CHARS = 4000)`，只截显示，不截已生成的 token

### 4.6 观测与计量

- `src/observability/tracing.js`：LangSmith 接入，只做自检 / metadata / flush，**不做 token 汇总**
- `src/graph/nodes.js:161`：`emit("agent","model_invoke", ..., { usage: reply.usage_metadata })`
- `src/db/queries.js:224-233`：`appendAudit` 把上面 detail 整体写进 `audit_events.detail` (jsonb)
- `src/security/guard.js:52-72`：`createAuditor`，DB 写入失败 fail-open

也就是说：**每次调用的完整 usage（含 reasoning、cache_read）已经在库里了，只是没有查询、没有聚合、没有计价。**

### 4.7 离线评测（另一笔账）

- `src/eval/judge.js`：7 个 typed question（faithfulness / decision / task_completed / helpfulness / pii_leak / context_precision / context_recall）
- `scripts/eval-smith.mjs:166-180, 236-242`：每个 case 走 Jev judge，日志里会打 `input_tokens`
- 这笔开销不在 `/chat` 链路上，但跑一次全量 eval 就是 `case 数 × 7 次 judge 调用`，做成本预算时要单独列

---

## 五、已经确认的五个缺口

**缺口 1（最严重）：`recent` 配额只记账、不裁剪**

`assemble.js:60-64` 在超限时只是把 `recentTokens` 赋成 `B.recent` 并推一条 trimming 记录，`recentText` 本身没有截断；而 `nodes.js:149` 发送的是 `state.messages` 原文。
结果：报告说 recent = 4000，实际历史一行不少，这是 6,245 vs 14,754 差距的主因。

**缺口 2：`headroom` 是名义预留，不是硬闸**

`assemble.js:120-125` 只计算 `overflow = total + headroom > window`，而 `total` 用的是被低估的估算值，所以 `context_overflow` 几乎不可能触发。真正超窗口时，报错发生在 provider 侧（400 / 截断），系统自身无感知。

**缺口 3：预算闸只有轮次，没有 token**

`src/security/rbac.js:34-37` `checkStepBudget(steps, MAX_STEPS=14)`，`nodes.js:269-272` 的 `budgetGate` 只数步数。
最坏情况：一轮里 14 次 agent 调用，每次 ~15K 输入 ≈ **210K 输入 token / 单轮**，没有任何 token 层面的止损。

**缺口 4：费用口径缺失**

- `cache_read` 恒为 0（前缀不稳定，DeepSeek 上下文缓存没命中）
- 无价目表，无 `cost_usd`，LangSmith 因为模型名 `deepseek-flash` 不在内置价表里也不显示金额
- 蒸馏、embedding、judge 三条支线完全不在账内

**缺口 5：估算与真实值偏差不会被发现**

`estimateTokens` 的 CJK=1 字/token 偏保守，但漏算项（工具 schema、消息封装）更大，净结果是低估 50% 以上。因为没有「估算 vs 账单」的对照校验，这个偏差会一直存在。

---

## 六、优化优先级

**P0 · 让报告和账单对齐（1–2 小时）**

1. `assembleContext` 真正返回裁剪后的 messages，而不是只返回数字；`nodes.js:149` 改用裁剪版。
2. 把工具 schema 的 token 数纳入 `report.total`（`JSON.stringify(tools)` 过一遍 estimateTokens）。
3. 在 `emit` 里同时记录 `estimated` 与 `actual`，差超过 20% 就打 `estimate_drift` 事件。

预期：`context_overflow` 从「永远 false」变成真实可用的护栏。

**P0 · 修 prompt cache（收益最大）**

`cache_read = 0` 意味着 14.7K 每轮全价。把易变内容后置：`system + tools → 检索证据 → 历史`，并且保证前缀逐字节稳定（不要拼时间戳、traceId、随机 id）。
DeepSeek 上下文缓存按前缀命中自动生效，命中后这部分按缓存价计费，成本可降 60%–80%。

**P1 · 加 token 闸（半天）**

在 `budgetGate` 里累加本轮 `usage.input_tokens + output_tokens`，超过 `RUN_TOKEN_BUDGET`（建议先设 60K）就 `deny + reject`，与现有 `max_steps` 并列。这是把「14 步 210K」压回可控区间的唯一硬手段。

**P1 · 修 RAG 预算不自洽**

要么把 `CTX_RAG_TOKENS` 提到 4,800 让 6 块真能进，要么把 `RAG_EVIDENCE_CHUNK_TOKENS` 降到 250 保持 6×250=1500。现在这套是「以为给了 6 块、实际只进 1–2 块」，检索召回被人为砍掉。

**P2 · 补齐账本（1 天）**

- 新建 `llm_calls` 表：`call_id / trace_id / session_id / tenant_id / node / model / input / cache_read / output / reasoning / cost_usd / retry_idx / status`
- 蒸馏调用带上同一个 traceId，或至少打 `stage:distill` tag
- embedding 按 `batch × chunk_tokens` 单独记
- `/health` 输出本会话累计 token 与预估金额

---

## 七、复现本文所有数字

查最近调用明细（无需 psql，用项目自带的 pg）：

```bash
node -e "import('pg').then(async({default:pg})=>{const c=new pg.Client({connectionString:'postgresql://postgres@127.0.0.1:5432/postgres'});await c.connect();const r=await c.query(\"SELECT detail->'usage' u, created_at FROM agent_harness_gongdan_audit_events WHERE event='model_invoke' ORDER BY id DESC LIMIT 10\");console.log(r.rows);await c.end()})"
```

查上下文估算报告：

```bash
node -e "import('pg').then(async({default:pg})=>{const c=new pg.Client({connectionString:'postgresql://postgres@127.0.0.1:5432/postgres'});await c.connect();const r=await c.query(\"SELECT detail->>'total' t, detail->'regions' r FROM agent_harness_gongdan_audit_events WHERE event='context_assembled' ORDER BY id DESC LIMIT 3\");console.log(r.rows);await c.end()})"
```

按节点聚合（验证哪个节点最费钱，当前只有 agent 一个）：

```sql
SELECT node, count(*) calls,
       sum((detail->'usage'->>'input_tokens')::int)  in_tok,
       sum((detail->'usage'->>'output_tokens')::int) out_tok,
       sum((detail->'usage'->'input_token_details'->>'cache_read')::int) cached
FROM agent_harness_gongdan_audit_events
WHERE event = 'model_invoke'
GROUP BY node;
```
