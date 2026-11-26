# agent_harness_gongdan · Harness 技术说明文档

> 用途：把本项目的 **模型 / 库 / 工具 / 基础设施 / 模块划分** 一次讲清，便于迁移或复用到另一个项目。
> 配套阅读：`README.md`（六层架构与面试口述版）、`docs/离线评测说明.md`（评测细节）、`.env.example`（全量配置项）。
> 时效：以 2026-11-26 的代码为准。

---

## 0. 一页速览

| 维度 | 结论 |
|---|---|
| 项目 | 售后工单 Agent（Agent Harness 统一执行标准的可观测实验台） |
| 运行时 | Node.js ≥ 20，纯 ESM（`"type": "module"`），无 TypeScript、无构建步骤 |
| 编排 | LangGraph 1.x 主图 + PostgresSaver 断点续跑 |
| 模型 | DeepSeek `deepseek-flash`（对话 + 蒸馏）、`BAAI/bge-m3`（向量）、`BAAI/bge-reranker-v2-m3`（精排）、`jev-latest`（评测 judge）、`fake` 离线兜底 |
| 存储 | PostgreSQL + pgvector（共享 PGlite `:5432/postgres`）、Redis 8 |
| 观测 | LangSmith trace + TypeSafe 评测 |
| 服务 | Express 5，端口 3204 |
| 六层 | 记忆 / 上下文 / 安全 / 并发 / 评测 / RAG |

---

## 1. 模型清单

所有模型走 **OpenAI 兼容 `/chat/completions`** 协议，换供应商只改 `LLM_BASE_URL` + `LLM_MODEL`。

| 用途 | 模型 | 端点 | 关键参数 | 说明 |
|---|---|---|---|---|
| 主对话 / 工具规划 | `deepseek-flash` | `https://api.deepseek.com/v1` | temp 0.2，max_tokens 3072，timeout 30s，retries 2 | 保留 thinking（工具规划需要）；`deepseek-chat` 非思考别名已于 2026-07-24 下线 |
| 记忆蒸馏 / 事实抽取 | `deepseek-flash`（参数隔离） | 同上，可用 `DISTILL_BASE_URL` / `DISTILL_API_KEY` 独立 | temp 0，max_tokens 768，`response_format: json_object`，**显式关 thinking** | 高频短结构化任务；抽取器版本 `DISTILL_EXTRACTOR_VERSION` 随事实入库 |
| Embedding | `BAAI/bge-m3`（1024 维） | `https://api.siliconflow.cn/v1` | batch 32 | 换模型必须重建 pgvector 索引；无 key 自动降级 fake hash 向量（仍 1024 维） |
| 精排 Rerank | `BAAI/bge-reranker-v2-m3` | `https://api.siliconflow.cn/v1` | timeout 8s，API 阈值 0.05 | cross-encoder 逐对打分；`heuristic` 为启发式降级替身，两者阈值独立标定（0.05 vs 0.30） |
| 评测 Judge | `jev-latest`（Jev · TypeSafe System One） | `https://api.typesafe.ai` | timeout 30s | 直连 TypeSafe，与 Smith 在线 evaluator 同 key / 同模型 / 同 question 名；旧 Gateway 通道已弃用 |
| 离线兜底 | `fake` 确定性应答 | 本地 | `LLM_PROVIDER=fake` / `EMBEDDING_PROVIDER=fake` | 零成本、可断网；会故意发起高危工具调用以验证网关 |

单位成本口径：`MODEL_PRICING_JSON`（USD / 1M tokens），未配置时用内置参考价。

---

## 2. 库清单（`package.json` dependencies，共 11 个）

| 库 | 版本 | 作用 |
|---|---|---|
| `@langchain/core` | ^1.4.2 | 消息、工具、prompt 基础抽象 |
| `@langchain/langgraph` | ^1.7.5 | 主图编排、条件边、`interrupt()` 人工审批、状态通道 |
| `@langchain/langgraph-checkpoint-postgres` | ^1.1.4 | `PostgresSaver`：图执行快照落 PG，按 `thread_id` 断点恢复 |
| `@langchain/openai` | ^1.8.4 | 对话 / 蒸馏模型的接入层（OpenAI 兼容） |
| `@typesafe-ai/sdk` | ^0.7.0 | Jev judge 的 SDK（9 个 judge 指标） |
| `langsmith` | ^0.14.0 | trace 上报 + dataset 同步 + `evaluate()` + feedback |
| `express` | ^5.3.0 | HTTP 网关（注意 5.x 的路由语法与 async 错误自动捕获） |
| `ioredis` | ^5.4.1 | L1 限流 + ZSET 优先级队列（Lua 原子领取 / ack / 死信） |
| `pg` | ^8.13.1 | Postgres 驱动；连接池 `max=1`（共享 PGlite 单 IO 限制） |
| `zod` | ^3.24.1 | 启动期配置校验 + 蒸馏结构化输出的严格校验 |
| `dotenv` | ^16.4.5 | `.env` 加载，密钥不入库 |

标准库：`node:test`（22 项单测）、`node --watch`（dev）。无 TypeScript / 无打包器 / 无 ORM（手写 SQL）。

---

## 3. 基础设施与外部服务

| 组件 | 地址 / 版本 | 承担什么 |
|---|---|---|
| PostgreSQL + pgvector | `127.0.0.1:5432/postgres`（共享 PGlite，已装 pgvector） | 工作记忆 / 情景 / 语义事实 / fact_history / RAG 文档与 chunk / 审计表（append-only，RULE 拦截删改）/ eval_runs；HNSW 余弦索引 + GIN 倒排 + RLS FORCE |
| Redis 8 | `127.0.0.1:6379` | 限流固定窗口（fail-open 退内存计数）、蒸馏优先级队列（交互 0 / 批量 10+） |
| LangSmith | `https://api.smith.langchain.com`，项目 `agent_harness_gongdan` | 全链路 trace（metadata 仅非敏感 ID）+ 自动化评测 |
| DeepSeek | `https://api.deepseek.com/v1` | 主对话 + 蒸馏 |
| SiliconFlow | `https://api.siliconflow.cn/v1` | Embedding + Rerank（共用一把 key） |
| TypeSafe | `https://api.typesafe.ai` | Jev judge（本地 `JUDGE_API_KEY` = Smith secret `TYPESAFE_API_KEY`） |

命名空间约定：所有 DB 对象统一前缀 `agent_harness_gongdan`，checkpointer 用独立 schema，便于共享库里多项目共存。

---

## 4. 工具清单（`src/tools.js`）

| 工具 | 是否 bind 给模型 | 说明 |
|---|---|---|
| `get_order_status` | ✅ | 查订单状态（tenant 条件下推） |
| `get_refund_policy` | ✅ | 查退款政策（依赖 RAG 证据） |
| `send_email` | ✅ | 发送通知邮件 |
| `recall_memory` | ✅ | 按需召回语义/情景记忆（`RECALL_MODE=tool` 默认，`eager` 可回退） |
| `create_refund_ticket` | ⚠️ bind + `interrupt()` 人工审批 | 资金类动作，需独立 admin 审批且不能是发起人 |
| `export_all_orders` | ❌ 注册但**绝不 bind** | 高危演示：验证"tool_call 只是请求，授权在网关" |

工具侧防护：白名单 + RBAC 策略表 + 参数二次校验 + 参数脱敏，均 fail-close。

---

## 5. 源码模块地图

```
src/
├── config.js              # 阈值集中 + zod 校验 + getSecret() + SYSTEM_PROMPT
├── server.js              # Express 路由：/chat /approve /audit/:traceId /inspect /api/evals /health
├── runtime.js             # 图实例装配、依赖注入、预算
├── tools.js               # 6 个工具定义与注册表
├── graph/
│   ├── buildGraph.js      # 主图与条件边
│   ├── nodes.js           # guardInput / recallMemory / retrieveRAG / assembleContext / agent / executeTools / guardOutput / recordTurn
│   └── state.js           # 状态 schema
├── memory/
│   ├── working.js         # 工作记忆追加
│   ├── recall.js          # 情景 + 语义并行召回（Mem0 风格接口）
│   ├── distiller.js       # 异步蒸馏（JSON mode + zod + 重问一次）
│   ├── consolidate.js     # ADD/UPDATE/DELETE/NOOP 整合，旧事实 SUPERSEDED
│   └── forget.js          # 半衰期衰减 + TTL 归档 + 6h cron 扫描
├── context/
│   ├── assemble.js        # 6 区块拼装、优先级裁剪、来源标注
│   └── budget.js          # token 计数与 Headroom
├── security/
│   ├── guard.js           # 输入/输出护栏编排（L2/L3/L7）
│   ├── patterns.js        # 注入模式黑名单
│   ├── pii.js             # PII 识别与脱敏
│   ├── rbac.js            # Casbin 风格策略表（L5/L6）
│   └── rateLimit.js       # L1 Redis 固定窗口限流
├── rag/
│   ├── ingest.js / segment.js / catalog.js  # 入库、切分、目录映射
│   ├── retrieve.js        # 双路召回 + RRF + 精排入口（降级链）
│   └── rerank.js          # cross-encoder 主路 + heuristic 兜底 + shadow
├── queue/priority.js      # ZSET 队列：优先级、attempts/backoff/lease/dead-letter
├── llm/model.js           # 对话模型工厂（fake / openai）
├── llm/embedding.js       # 向量模型（siliconflow / fake）
├── observability/
│   ├── tracing.js         # LangSmith 配置与最小披露 metadata
│   └── cost.js            # costLedger（主链路 / 蒸馏分离）
├── eval/
│   ├── judge.js           # 9 个 judge 指标（6 reference-free + 2 golden + pii 反向）
│   └── dashboard.js       # 评测看板双轨聚合
└── db/
    ├── pool.js            # 连接池（max=1）
    └── queries.js         # 手写 SQL 访问层

scripts/    # init-db / seed / smoke / chat / eval / eval-smith / check-* / show-usage
tests/      # harness.test.js、memory-alignment.test.js（node:test）
data/       # orders.json、eval-cases.json、docs/ 知识库原文
public/     # 控制台（index.html / app.js / styles.css / rag-docs.*）
db/schema.sql
```

---

## 6. HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 模型 / 向量 / LangSmith / Redis / 预算 / 工具清单；`/health.memory`、`/health.queue` 另给记忆与队列细节 |
| POST | `/chat` | 返回回答 + RAG 链 + 上下文报告 + 审计链；高危时 `awaiting_approval` |
| POST | `/approve` | admin 审批（`x-role: admin` + `x-approver-id`，不能是发起人） |
| GET | `/audit/:traceId` | 从 DB 回捞审计事件 |
| GET | `/inspect` | 当前租户语义/情景记忆、表名、队列长度 |
| GET | `/api/evals?limit=12` | 评测看板：最近 N 次 `eval_runs` + 趋势（只读，DB 不可用返回 503） |

演示身份来自 `x-tenant-id` / `x-user-id` / `x-role`——**故意显眼的演示控件**，生产必须改为服务端会话注入。

---

## 7. 六层 Harness 落地要点

1. **记忆**：运行时状态（checkpoints）/ 工作记忆 / 情景记忆 / 语义记忆四层分离；PG 表实现 Mem0 风格 `add/search`。整合阈值 0.82，召回 70 天半衰期，事实 180 天 / 情景 90 天 TTL，蒸馏走 Redis 队列异步执行。
2. **上下文**：32K 共享窗口 + 35% Headroom；6 区块软上限（系统 2000 / State 500 / 近程 4000 / 记忆 1500 / RAG 1500 / 远端 1000 / Headroom 11200）；超窗按"低相似度记忆 → 远端摘要 → 近程原文 → 最后系统指令"裁剪。
3. **安全 L1–L10**：全部挂在 LangGraph 节点与条件边上，不依赖 system prompt。fail-close：注入、PII、越权、无证据、tenant 隔离；fail-open：精排、审核、审计（加告警）。
4. **并发**：限流前置快速失败 + 断点续跑 + ZSET 优先级队列 + Worker 本地超时/重试/熔断。
5. **评测**：本地 9 条用例（rag/tool/safety/context）+ LangSmith 自动化评测（机制断言 fail-close、9 个 Jev judge fail-open）+ 双基线门禁 + 双轨看板。
6. **RAG**：清洗截断 512 → bge-m3 + 分词 → 向量 top30（HNSW）+ 关键词 top30（GIN, ts_rank_cd）→ RRF(k=60) → 精排 top20（阈值 0.05）→ 最多 6 块 / 单块 800 token → 强制 `[证据n]` 引用。降级链：精排异常回 RRF，关键词异常退纯向量，无结果拒答，向量不可用拒答，生成超时只返证据。

---

## 8. 工程工具链（npm scripts）

| 命令 | 作用 |
|---|---|
| `npm start` / `npm run dev` | 生产启动 / `node --watch` 热重启（:3204） |
| `npm run db:init` / `db:check` | 建表 + 索引 + RULE + RLS + checkpoint schema / 列出带前缀对象 |
| `npm run seed` | 灌 5 份知识文档 + 3 条语义记忆（两租户） |
| `npm run chat -- "问题"` | CLI 全链路，打印上下文报告与审计链 |
| `npm run smoke` | 9 项端到端冒烟（fake 模型，不依赖外部 API） |
| `npm run eval` / `npm run eval:smith` | 本地离线门禁 / LangSmith 自动化评测 |
| `npm test` | 22 项纯单测（不连库不调模型） |
| `npm run check:llm` / `check:smith` / `check:eval` | 端点与维度 / LangSmith 连通性 / 评测健康 |
| `npm run usage` | token 消耗与成本 |

---

## 9. 可迁移的工程约定

- **配置即策略**：所有阈值集中在 `src/config.js`，启动期 zod 校验；密钥单独 `getSecret()`，不进 config 对象、不进日志与 trace metadata。
- **端点可信边界**：`assertSafeEndpoint()` 强制 HTTPS（仅本机可 http）、禁止 URL 夹带凭据；`assertSharedDatabase()` 锁定共享库地址，防止误改端口/库名。
- **模型可插拔**：OpenAI 兼容 + `fake` 确定性兜底，测试/冒烟/评测默认零成本离线可跑。
- **评测双基线**：`out/eval-baseline.json`（机制）与 `out/eval-smith-baseline.json`（judge）分开；机制 fail-close、judge fail-open，`judge_pii_leak` 反向单判（< 0.5）。`overall` 不进门禁，防聚合分掩盖切片回归。
- **知识标注只进 referenceOutputs**：`golden` 写进 dataset 的 outputs，不写 inputs，被测 Agent 看不到标准答案。
- **DB 命名空间**：统一前缀 + 独立 checkpoint schema，支持共享库多项目共存。
- **已知取舍**（生产需替换）：正则注入/PII 基线（非 Lakera/Presidio/LlamaGuard）、策略表 RBAC（非 Casbin）、heuristic 精排兜底、预算账本为进程内存态（应换 Redis/PG 计数 + OTel 告警）。
