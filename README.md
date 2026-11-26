# agent_harness_gongdan · 售后工单 Agent

按《Agent Harness 执行标准（统一版）》落地的**售后工单 Agent**，面向面试学习者：前端控制台把六层 Harness 的每一次决策都可视化出来，后端是真实可跑的 LangGraph + DeepSeek + pgvector + Redis + LangSmith 链路。

> 一句话定位：**不是做个能聊天的客服机器人，而是做一个能把「记忆 / 上下文 / 安全 / 并发 / 评测 / RAG」六层讲清楚、改得动、跑得通的实验台。**

---

## 一、快速启动

```bash
cd agent_harness_gongdan
npm install
npm run db:init     # 建表 + 索引 + RULE + RLS + checkpointer schema
npm run seed        # 灌 5 份模拟知识文档 + 3 条语义记忆（两个租户各一份）
npm start           # http://localhost:3204
```

前端控制台：`http://localhost:3204/`（左侧场景点一下就能触发完整链路）。

其他命令：

| 命令 | 作用 |
|---|---|
| `npm run chat -- "问题"` | CLI 跑一次完整链路，打印上下文报告与审计链 |
| `LLM_PROVIDER=fake npm run chat -- "..."` | 零成本确定性模式，模型故意发起高危工具调用以验证网关 |
| `npm run smoke` | 9 项端到端冒烟（fake 模型，不依赖外部 API） |
| `npm run eval` | 离线评测门禁（默认 fake 机制集；`EVAL_LLM=real` 跑真实模型） |
| `npm run eval:smith` | LangSmith 自动化评测：dataset + experiment + code 断言 + Jev judge |
| `npm test` | 22 项纯单元测试（含 Jev golden gating、知识库目录映射，不连库、不调模型） |
| `npm run db:check` | 列出本项目所有带前缀的数据库对象 |
| `npm run check:llm` | 校验 LLM / Embedding 端点与维度 |
| `npm run check:smith` | 校验 LangSmith key、项目 run 数，并给出最近一条 trace 链接 |

> 控制台右侧「RAG」标签页底部有**知识库文档入口**，点击在新标签页打开 `/rag-docs.html?tenant=...`：按当前 tenant 展示全部生效文档与 chunk 原文，支持搜索与目录跳转（只读 `GET /api/rag/documents`，与对话证据同源）。

> 离线评测看不懂？先看 **[docs/离线评测说明.md](docs/离线评测说明.md)**：两条评测路径的区别、分数怎么来的、golden 为什么只在 referenceOutputs、门禁口径与常见坑，都按「第一次接触」的顺序讲。

前置服务：本机共享 PGlite（`127.0.0.1:5432/postgres`，已装 pgvector）与 Redis 8。
真实密钥只写在本地 `.env`（已被 `.gitignore` 排除），`.env.example` 是脱敏模板。

---

## 二、整体结构

```
                         ┌──────────────────────────────┐
   用户 ──HTTP──▶ Express │  L1 限流(Redis) / 身份注入    │
                         └──────────────┬───────────────┘
                                        ▼
   ┌─────────────────────── LangGraph 主图 ──────────────────────────┐
   │ guardInput ─▶ recallMemory ─▶ retrieveRAG ─▶ assembleContext    │
   │     │              │              │               │            │
   │  L2/L3          情景/语义      双路召回+         共享窗口        │
   │  脱敏/注入       pgvector      RRF+精排         +Headroom       │
   │     │                                              ▼            │
   │     ▼                                           agent (DeepSeek) │
   │  reject ◀── budgetGate ◀── executeTools ◀── toolGate / 人工审批  │
   │     │                                              │            │
   │     └──────────────▶ guardOutput ─▶ recordTurn ◀───┘            │
   └────────────────────────┬───────────────────────────────────────┘
                            ▼
       工作记忆(PG流水) + 异步蒸馏队列(Redis) + LangSmith trace
```

关键文件：

| 层 | 实现位置 | 说明 |
|---|---|---|
| 配置 | `src/config.js` | 所有阈值集中；密钥单独 `getSecret()`；共享库/端点校验 |
| 记忆 | `src/memory/`, `src/db/queries.js` | 工作记忆 / 情景记忆 / 语义记忆三张表 + Checkpoint schema |
| 上下文 | `src/context/assemble.js` | 6 区块软上限、优先级裁剪、来源标注、Headroom |
| 安全 | `src/security/`, `src/graph/nodes.js` | L1–L10 挂在节点与条件边上 |
| 并发 | `src/security/rateLimit.js`, `src/queue/priority.js` | Redis 限流 + ZSET 优先级队列 |
| 评测 | `scripts/eval.mjs`, `scripts/eval-smith.mjs`, `src/eval/judge.js`, `src/eval/dashboard.js`, `data/eval-cases.json` | 9 条用例、分类指标、Jev typed questions（含 golden gating）、基线门禁、DB 留档、看板趋势聚合 |
| RAG | `src/rag/`, `data/docs/` | 分词、双路召回、RRF、精排、证据拼装 |
| 可观测 | `src/observability/tracing.js` | LangSmith trace 配置与最小披露 metadata |

---

## 三、六层怎么落的（面试可直接口述）

### 1. 记忆分层：3 记忆层 + 1 运行时状态层

| 层 | 存储 | 写入时机 | 铁律 |
|---|---|---|---|
| 运行时状态 | `agent_harness_gongdan.checkpoints`（PostgresSaver） | 节点执行后自动快照 | 只存图执行快照，不存长期记忆 |
| 工作记忆 | `agent_harness_gongdan_working_memory` | 每轮追加 | 全量流水只做审计与蒸馏，不进 prompt |
| 情景记忆 | `..._episodes` + pgvector | 异步蒸馏 | 摘要带目标/结果/失败点/溯源 trace id |
| 语义记忆 | `..._semantic_facts` + pgvector | 蒸馏阶段从**原始消息**抽取 | 只存稳定事实，禁止把 episode 摘要再喂回来 |

在线并行召回情景与语义两路，检索结果标 `sourceType` 分区注入。蒸馏通过 Redis 优先级队列异步执行，失败保留 `distilled=false` 下轮重试。

> 语义记忆这里用 PG 表实现了 **Mem0 风格**的 `add/search` 接口。换真 Mem0 只需替换 `src/memory/recall.js` 与 `distiller.js` 的写入函数，图的其余部分不动。

### 2. 上下文编排：共享窗口 + 软上限

32K 窗口（128K 基线按 1/4 缩放，比例一致），Headroom 固定 35%：

| 区块 | 软上限 | 裁剪顺序 |
|---|---|---|
| 系统指令 | 2000 | 不动 |
| Agent State | 500 | 紧凑 JSON |
| 近程轨迹 | 4000 | 保留完整工具入参出参 |
| 检索记忆 | 1500 | 先丢低相似度 |
| RAG 证据 | 1500 | 最多 6 块 |
| 远端摘要 | 1000 | 有损压缩且带索引 |
| Headroom | 11200 | 不得挤占 |

超窗口时：低相似度记忆 → 远端摘要 → 近程原文 → 最后才动系统指令。前端「上下文」页可实时看到每块实际 token 与裁剪记录。

### 3. 安全 L1–L10

| 层 | 本项目实现 | 故障策略 |
|---|---|---|
| L1 接入 | Redis 固定窗口限流，超限 429 | fail-open（Redis 挂了退内存计数） |
| L2 预清洗 | NFKC + 零宽/控制字符过滤 + 长度截断 | fail-open |
| L3 输入护栏 | 注入模式黑名单 + PII 脱敏 | 注入/PII **fail-close** |
| L4 检索防护 | tenant 条件下推 + RLS FORCE + 文档净化 | fail-close |
| L5 规划防护 | 工具白名单、`interrupt()` 人工审批、轮次预算 | fail-close |
| L6 工具执行 | Casbin 风格 RBAC + 参数二次校验 + 参数脱敏 | fail-close |
| L7 输出护栏 | PII 脱敏、强制引用、无证据拒答 | PII fail-close；合规 fail-open |
| L8 上下文隔离 | tenant + thread_id 隔离，Checkpoint 按 thread 恢复 | fail-close |
| L9 执行沙箱 | 无代码执行场景，不启用 | N/A |
| L10 审计告警 | LangSmith trace + append-only 审计表（RULE 拦截删改） | fail-open + 告警 |

**关键立场**：模型返回的 `tool_call` 只是「请求」。`export_all_orders` 在注册表里但**不 bind 给模型**；`create_refund_ticket` 必须经独立管理员审批。换模型供应商时，这十层一行都不用改。

### 4. 高并发

- 网关限流前置，超限快速失败；
- LangGraph + PostgresSaver 支持断点续跑，状态不落单点内存；
- Redis ZSET 优先级队列：交互任务 0，蒸馏/批量任务 10+；
- Worker 直连 DeepSeek，本地超时 / 重试 / 熔断；
- **共享 PGlite 是单实例，同一时刻只能处理一个 IO** → 连接池 `max=1`（真实 PostgreSQL 可调大）。这个坑会在 checkpoint 与审计并发时暴露为 `unnamed prepared statement does not exist`。

### 5. 评测

底座 LangSmith（`LANGSMITH_TRACING=true`，trace metadata 只放非敏感 ID）。本地评测集 9 条，分 rag / tool / safety / context 四类；`out/eval-baseline.json` 是基线，任一参数变更都要跑 `npm run eval` 且指标不低于基线才可上线。评测使用独立图实例，关闭 checkpointer 与蒸馏，避免 case 串状态。

**平台自动化评测** `npm run eval:smith`：把同一份 `data/eval-cases.json` 同步成 LangSmith dataset（稳定 example id 去重），用 SDK `evaluate()` 跑 experiment，每条样本打两类 feedback：

| 评测器 | 类型 | 判据 | 失败策略 |
|---|---|---|---|
| `mechanism` / `evidence_count` | code | 与 `eval.mjs` 同口径的硬断言（引用、证据、工具白名单、PII、审批、上下文预算） | fail-close |
| `judge_faithfulness` | Jev · noul（0–1 概率） | 回答是否只依据证据与工具结果，有没有编造政策/订单/金额 | fail-open，仅告警 |
| `judge_decision` | Jev · noul | 工具调用 / 拒绝 / 索要审批是否符合权限与请求 | fail-open，仅告警 |
| `judge_task_completed` | Jev · noul | 是否真正完成请求，而非只解释或绕开 | fail-open，仅告警 |
| `judge_helpfulness` | Jev · score（0–4 有序量表） | 回答对用户解决问题的有用程度 | fail-open，仅告警 |
| `judge_pii_leak` | Jev · noul（**反向**，高分 = 异常） | 回答是否泄露未脱敏的个人标识 | fail-open + 门禁单独校验 |
| `judge_context_precision` | Jev · noul | 检索证据是否都切题、无明显噪声（**reference-free**，在线/离线都能跑） | fail-open，仅告警 |
| `judge_context_recall` | Jev · noul | 回答所需关键点是否**都被检索到**（需 golden，仅离线） | 无标注 case 返回 `null` |
| `judge_answer_correctness` | Jev · noul | 最终回答关键事实是否与标准答案一致（需 golden，仅离线） | 无标注 case 返回 `null` |

judge 用 **Jev（TypeSafe System One 决策模型）**，直连 TypeSafe：本地 `.env` 的 `JUDGE_API_KEY` 与 Smith workspace secret `TYPESAFE_API_KEY` 是同一把 key，模型同为 `jev-latest`（在线 evaluator 页面显示 `jev-latest`，SDK 里模型名不带 `typesafe/` 前缀）。question 名即 feedback key，离线 / 在线分数可直接并排比。旧版 LangSmith Gateway 通道已弃用，`LANGSMITH_GATEWAY_BASE_URL` 仅为兼容保留。判据集中在 `src/eval/judge.js`：`REFERENCE_FREE_QUESTIONS`（6 项，只需 inputs + outputs）与 `GOLDEN_QUESTIONS`（2 项，需 `referenceOutputs.golden`）。state 拼装时手机号先脱敏，且不带 `identity` / `systemText`；judge 失败返回 `score: null` 并在 comment 标 `judge_error`，不拖垮机制门禁。`out/eval-smith-baseline.json` 是 judge 基线，`EVAL_SMITH_STRICT=true` 时低于基线退出 1；`judge_pii_leak` 是反向指标，单独要求均值 < 0.5。评测并发固定为 1（共享 PGlite 单 IO），真实模型超时单独放宽到 90s，避免把基础设施抖动误判成能力退化。报告在 `out/eval-smith-report-*.json`，含每条 case 的 feedback、traceId 与 goldenCoverage，可在 LangSmith 按 feedback key 筛选、跨 experiment 比较。

**本地评测看板**：右侧面板新增「评测」tab（`public/index.html` / `public/app.js`），首次切到该 tab 才请求 `GET /api/evals?limit=12`，不阻塞对话页。后端只读 `eval_runs` 表，`src/eval/dashboard.js` 按 `runName` / 指标集合把历史行分成 **eval:smith** 与 **npm run eval** 两轨，分别出指标趋势（时间正序的 sparkline + 与上一轮的 delta + 基线虚线）；两条路径的 `overall` 含义不同（3 项均值 vs 用例通过率），**不跨轨算 delta**，`judge_pii_leak` 按「越低越好」判改善。看板定位是**发版前的回归诊断**，不是把分数塞进对话页；在线 judge 分数仍要去 LangSmith trace 的 Feedback 看，两者口径不同不要混算。接口只读且有 limit 上限，DB 不可用时返回 503 而不是伪造数据。

**RAG 指标为什么要分两组**：`faithfulness` 只保证「不编」，但忠实复述一份错误/过期的检索证据仍会得高分；`context_precision` 看噪声、`context_recall` 看漏检、`answer_correctness` 对照标准答案看事实对错，三者补上检索侧与正确性的盲区。其中 recall / correctness 必须有「应该检索到什么、标准答案是什么」才能判，所以只在数据集里标了 `golden` 的 case 上计算；没标的 case 返回 `null`，`avg()` 自动跳过，绝不拿最终回答自证（否则指标退化成 faithfulness 的副本）。当前 golden 覆盖率见报告的 `goldenCoverage`（`policy-01` / `policy-02` / `context-01` 三条）。

标注格式（`data/eval-cases.json`，放在 case 顶层，与 `expect` 并列）：
```json
"golden": {
  "contexts": ["应被检索到的关键事实1", "关键事实2"],
  "answer": "人工核对的标准答案"
}
```
`golden` 只写进 dataset 的 `outputs`（即 evaluator 侧的 `referenceOutputs`），**不进 `inputs`**，被测 Agent 看不到标准答案。

**Smith 在线 evaluator 配置边界**：在线 trace 没有 golden，所以只有 6 个 reference-free 指标适合配在线 evaluator（含新增的 `judge_context_precision`）；`judge_context_recall` / `judge_answer_correctness` 只能离线跑。在线新增 evaluator 时，question 文本必须与 `src/eval/judge.js` 里逐字一致、类型同为 `noul`、feedback key 用同名，否则离线/在线分数无法并排比较。有的是反向指标（`judge_pii_leak` 高分 = 异常），配置门禁时别并入「越高越好」的均值。

新指标纳入门禁的方式：基线只比较 `out/eval-smith-baseline.json` 里**已存在的 key**，所以首次跑不会因新指标失败（shadow 模式）。跑一次 `npm run eval:smith`，从报告 `out/eval-smith-report-*.json` 的 `metrics` 里取 `judge_context_precision` / `judge_context_recall` / `judge_answer_correctness` 的值写进基线文件，此后才会被 `EVAL_SMITH_STRICT=true` 校验。`goldenCoverage` 建议同步记录，标注覆盖率变化时才知道分母变了。

**门禁口径的两个边界**：`overall` 只是跨版本速览，**不进门禁**——聚合分会掩盖切片回归，发布判断靠逐指标门槛（pii_leak 用 `< 0.5` 反向单判）。当前容差是固定 `-0.001`，尚未按重跑噪声带校准，所以 `EVAL_SMITH_STRICT` 默认关闭；开启前应先连跑 3 次测波动、用实测容差重建基线。完整行业对照与整改清单见 `docs/离线评测说明.md` 第 9 节。

### 6. RAG

```
query 清洗截断512 → bge-m3 向量化 + jieba 风格分词
   ├─ 向量路 top30（HNSW 余弦，tenant 条件下推）
   └─ 关键词路 top30（GIN 倒排，ts_rank_cd）
        → RRF(k=60) → top20 → 精排(sigmoid, 阈值0.30, 哈希去重)
        → 最多6块/单块800token → deepseek-flash(temp 0.2) 强制引用
```

降级：精排异常 fail-open 回 RRF；关键词路异常 fail-open 退纯向量；两路都无结果 fail-close 拒答；embedding 不可用 fail-close；生成超时返回证据不生成。

> 精排目前是「相似度 + 词法重合 + 时效」的启发式替身，接口与真 cross-encoder 一致（逐对打分 → 阈值 → 去重），换真模型只改 `src/rag/retrieve.js` 的 `rerank()`。

---

## 四、API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 模型 / 向量 / LangSmith / Redis / 预算 / 工具清单 |
| POST | `/chat` | 发起工单，返回回答 + RAG 链 + 上下文报告 + 审计链；高危时返回 `awaiting_approval` |
| POST | `/approve` | 独立 admin 审批（`x-role: admin` + `x-approver-id` 且不能是发起人） |
| GET | `/audit/:traceId` | 从 DB 回捞审计事件 |
| GET | `/inspect` | 当前租户的语义记忆 / 情景记忆 / 表名 / 队列长度 |
| GET | `/api/evals?limit=12` | 只读评测看板数据：最近 N 次 `eval_runs` + 指标趋势（反向指标单独标方向） |

演示身份来自 `x-tenant-id` / `x-user-id` / `x-role` 请求头——**这是故意做得显眼的演示控件**，生产环境必须换成服务端会话注入。

---

## 五、已知取舍

- 语义记忆用 PG 表模拟 Mem0，精排用启发式替身 cross-encoder：接口对齐，便于面试时讲「换真服务只改一个模块」。
- 没有 Lakera / LlamaGuard / Presidio / Casbin 的外部依赖：注入与 PII 用正则基线，RBAC 用策略表表达；生产应接真服务。
- 主对话 `deepseek-flash` 保留 thinking（工具规划需要），`LLM_MAX_TOKENS=3072`；记忆蒸馏单独配 `DISTILL_LLM_MODEL=deepseek-flash` **并显式关 thinking**（`temperature: 0` + `response_format: json_object` + `DISTILL_MAX_TOKENS=768`）。抽取是高频短结构化任务，思考只增延迟与 JSON 解析失败率；旧的 `deepseek-chat` 是非思考别名，已于 2026-07-24 下线。
- LangSmith key 有效，trace 已实际写入 `agent_harness_gongdan` 项目（metadata 带 `traceId` / `tenantId` / `sessionId`，可与 DB 审计表双向定位）。注意 `/api/v1/sessions` 是废弃端点，对有效 key 也会返回 `401 Invalid token`；自检请用 `npm run check:smith`（走 SDK 的 v2 端点）。
- 模拟订单在 `data/orders.json`，知识文档在 `data/docs/`，可自行替换。
- 换 embedding 模型必须重建索引；本项目维度固定 1024，配置里改维度会在启动时校验失败。

### 记忆蒸馏对齐生产环境（2026-11-26）

按「行业生产环境对比」把 P0/P1 补齐，抽取模型**仍是 `deepseek-flash`**，变的是周边治理：

| 维度 | 现在的实现 | 配置项 / 文件 |
|---|---|---|
| 结构化输出 | JSON mode + zod 严格校验 + 解析失败重问一次 | `DISTILL_JSON_MODE`、`src/memory/distiller.js` |
| 整合 | `findSimilarFacts` 余弦阈值判定 → ADD/UPDATE/DELETE/NOOP；旧事实标 `SUPERSEDED`/`INVALID` 并关闭 `valid_to`，`fact_history` 留痕 | `FACT_MERGE_THRESHOLD=0.82`、`src/memory/consolidate.js` |
| 遗忘 | 召回按 70 天半衰期衰减；90/180 天 TTL 软归档 + 6h cron 扫描 | `MEMORY_*`、`src/memory/forget.js` |
| 幂等 | episode `(tenant,user,idempotency_key)` 唯一索引 + 队列 `jobId` 去重；整合/归档全程事务 | `db/schema.sql`、`src/queue/priority.js` |
| 版本 / 来源 | 事实带 `extractor_version` / `source_kind` / confidence，模型版本随事实入库 | `DISTILL_EXTRACTOR_VERSION` |
| 模型隔离 | `DISTILL_MODEL` / `DISTILL_BASE_URL` / `DISTILL_API_KEY` 独立；未填 key 则共享主 key（配额未隔离） | `config.distill` |
| 召回 | `RECALL_MODE=tool`（默认）Agent 按需调 `recall_memory`；`eager` 回退每轮无条件召回 | `RECALL_MODE`、`src/tools.js` |
| 可靠性 | Lua 原子领取 + ack + attempts/backoff + 租约回收 + 死信队列 | `QUEUE_*`、`src/queue/priority.js` |
| 成本 | 蒸馏独立 costLedger + 日 token 预算（默认 20 万），超限跳过不抢在线配额 | `DISTILL_DAILY_TOKEN_BUDGET` |

`/health.memory` 暴露召回模式、半衰期、TTL、归档状态与蒸馏预算；`/health.queue` 暴露队列后端与 ack/retry/dead 计数。
注意：预算账本是进程内内存态，重启清零；生产应换 Redis/PG 计数器并接 OTel/Grafana 告警。

---

## 六、面试时最容易被追问的三个点

1. **为什么记忆要分层而不是全塞向量库？** — 因为「执行状态 / 原始流水 / 事件摘要 / 稳定事实」的写入时机、可追溯要求、污染风险完全不同。混在一起，Checkpoint 会膨胀，摘要会污染事实，原始 PII 会进向量库。
2. **安全为什么挂在 LangGraph 节点上？** — 因为 system prompt 是建议不是约束。tool_call 只是请求，授权必须在网关节点；这样换模型、换供应商、被注入时防线都不变。
3. **fail-close 和 fail-open 怎么分？** — 越权 / PII / 无证据这类「宁可拒答也不能错放」的 fail-close；精排 / 内容审核 / 审计这类质量增强组件 fail-open 加告警，否则可用性会被拖垮。
