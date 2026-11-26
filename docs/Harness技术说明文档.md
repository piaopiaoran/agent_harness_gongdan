# agent_harness_gongdan · Harness 技术说明文档

> 面向"迁移 / 复用到另一个项目"的技术说明：把本项目的**模型、库、工具、基础设施**一次讲清，配合 `README.md`（六层架构详解）与 `.env.example`（完整配置项）使用。所有内容以 2026-11-26 的代码为准。

---

## 1. 项目定位

**售后工单 Agent**：按《Agent Harness 执行标准（统一版）》落地的可观测实验台。前端控制台把六层 Harness（记忆 / 上下文 / 安全 / 并发 / 评测 / RAG）的每一次决策可视化；后端是真实可跑的 LangGraph + DeepSeek + pgvector + Redis + LangSmith 链路。

技术栈一句话：**Node.js ≥ 20 · ESM · Express 5 · LangGraph 1.x · Postgres(pgvector) · Redis 8 · LangSmith**。

---

## 2. 模型清单（全部可独立替换，端点均为 OpenAI 兼容）

| 用途 | 模型 | 端点 / 供应商 | 关键配置 | 备注 |
|---|---|---|---|---|
| 主对话 / 工具规划 | `deepseek-flash` | `https://api.deepseek.com/v1` | temp 0.2，max_tokens 3072，timeout 30s，retries 2 | **保留 thinking**（工具规划需要）；旧的 `deepseek-chat` 非思考别名已下线 |
| 记忆蒸馏 / 事实抽取 | `deepseek-flash`（与主模型同名但参数隔离） | 同上，可经 `DISTILL_BASE_URL/DISTILL_API_KEY` 换端点 | temp 0（显式**关 thinking**），max_tokens 768，`response_format: json_object` | 高频短结构化任务，思考只增延迟与 JSON 解析失败率；抽取器版本号随事实入库 |
| Embedding | `BAAI/bge-m3`（1024 维） | `https://api.siliconflow.cn/v1` | batch 32，维度 1024 | 换模型必须重建 pgvector 索引；未填 key 自动降级 fake hash 向量（仍 1024 维） |
| 精排 Rerank | `BAAI/bge-reranker-v2-m3`（cross-encoder） | `https://api.siliconflow.cn/v1` | timeout 8s，API 阈值 0.05（heuristic 是 0.30，分开标定） | 支持 shadow 双跑；heuristic 启发式为降级兜底，接口与真 cross-encoder 一致 |
| 评测 Judge | `jev-latest`（Jev · TypeSafe System One 决策模型） | `https://api.typesafe.ai` | timeout 30s；直连 TypeSafe，非 LangSmith Gateway | 离线 `eval:smith` 与 Smith 在线 evaluator 共用同一模型、同一把 key、同一套 question 名 |
| 确定性兜底 | `fake`（本地确定性应答） | 本地 | `LLM_PROVIDER=fake` / `EMBEDDING_PROVIDER=fake` | 零成本、可离线；故意发起高危工具调用以验证安全网关 |

**模型接入方式**：统一走 `@langchain/openai` 的 OpenAI 兼容 `/chat/completions`，`LLM_BASE_URL` + `LLM_MODEL` 两项即可换供应商；换模型供应商时安全十层一行不用改。

---

## 3. 核心库清单（package.json dependencies）

| 库 | 版本 | 在本项目中的角色 |
|---|---|---|
| `@langchain/core` | ^1.4.2 | 消息、工具定义、prompt 组装的基础抽象 |
| `@langchain/langgraph` | ^1.7.5 | **主图编排**：guardInput → recallMemory → retrieveRAG → assembleContext → agent → executeTools → guardOutput → recordTurn；`interrupt()` 人工审批 |
| `@langchain/langgraph-checkpoint-postgres` | ^1.1.4 | PostgresSaver：图执行快照落 PG，断点续跑，按 thread_id 恢复（独立 schema `agent_harness_gongdan`） |
| `@langchain/openai` | ^1.8.4 | 所有模型（对话 / 蒸馏）的接入层，OpenAI 兼容协议 |
| `@typesafe-ai/sdk` | ^0.7.0 | Jev judge 的 SDK：离线评测 9 个 judge 指标直连 TypeSafe |
| `langsmith` | ^0.14.0 | trace 上报 + 自动化评测（dataset 同步、SDK `evaluate()`、feedback） |
| `express` | ^5.3.0 | HTTP 网关：`/chat` `/approve` `/audit/:traceId` `/inspect` `/api/evals` `/health` |
| `ioredis` | ^5.4.1 | L1 固定窗口限流 + ZSET 优先级队列（含 Lua 原子领取 / ack / 死信） |
| `pg` | ^8.13.1 | Postgres 驱动；连接池 `max=1`（共享 PGlite 单 IO 的约束，真实 PG 可调大） |
| `zod` | ^3.24.1 | 启动期配置校验（config.js 的 schema）+ 蒸馏结构化输出严格校验 |
| `dotenv` | ^16.4.5 | `.env` 加载；真实密钥不进仓库 |

**标准库**：Node 20+ 原生 `node:test`（22 项单元测试）、`node --watch`（dev 模式），无额外测试框架。

---

## 4. 基础设施与外部服务

| 组件 | 地址 / 版本 | 用途 |
|---|---|---|
| PostgreSQL + pgvector | 本机共享 `127.0.0.1:5432/postgres`（底层为共享 PGlite，装了 pgvector） | 工作记忆 / 情景 / 语义事实三张表 + fact_history + RAG 文档/chunk + 审计表（RULE 拦截删改，append-only）+ eval_runs + HNSW 向量索引 + GIN 倒排 + RLS FORCE |
| Redis 8 | `127.0.0.1:6379` | L1 限流（fail-open 退内存计数）、蒸馏优先级队列（交互 0 / 批量 10+）、原子任务领取 |
| LangSmith | `https://api.smith.langchain.com`，项目 `agent_harness_gongdan` | 全链路 trace（metadata 只放非敏感 ID）+ 自动化评测 dataset/experiment/feedback |
| DeepSeek API | `https://api.deepseek.com/v1` | 主对话 + 蒸馏模型 |
| SiliconFlow | `https://api.siliconflow.cn/v1` | Embedding（bge-m3）+ Rerank（bge-reranker-v2-m3），两者共用一把 key |
| TypeSafe | `https://api.typesafe.ai` | Jev judge 直连；本地 `JUDGE_API_KEY` 与 Smith workspace secret `TYPESAFE_API_KEY` 同一把 |

---

## 5. Agent 工具清单（`src/tools.js` 注册表）

| 工具 | 是否 bind 给模型 | 说明 |
|---|---|---|
| `get_order_status` | ✅ | 查订单状态（tenant 条件下推） |
| `get_refund_policy` | ✅ | 查退款政策（走 RAG 证据） |
| `send_email` | ✅ | 发送通知邮件 |
| `recall_memory` | ✅ | 按需语义/情景记忆召回（`RECALL_MODE=tool` 默认） |
| `create_refund_ticket` | ⚠️ bind 但需 `interrupt()` 人工审批 | 资金类动作，独立 admin 审批（不能是发起人） |
| `export_all_orders` | ❌ **注册但绝不 bind** | 高危演示工具：在注册表里但不暴露给模型，验证"tool_call 只是请求"的立场 |

安全约束：白名单 + RBAC（Casbin 风格策略表）+ 参数二次校验 + 参数脱敏，全部 fail-close。

---

## 6. 六层 Harness 落地摘要（细节见 README.md）

1. **记忆**：运行时状态（PostgresSaver checkpoints）/ 工作记忆 / 情景记忆 / 语义记忆四层分离；PG 表实现 Mem0 风格 `add/search`，换真 Mem0 只改 `src/memory/recall.js` 与 `distiller.js`。整合用余弦阈值 0.82 做 ADD/UPDATE/DELETE/NOOP，遗忘按 70 天半衰期 + 90/180 天 TTL。
2. **上下文**：32K 共享窗口 + 35% Headroom，6 区块软上限，超窗按"低相似度记忆 → 远端摘要 → 近程原文 → 最后系统指令"裁剪。
3. **安全 L1–L10**：挂在 LangGraph 节点与条件边上，不依赖 system prompt；fail-close（越权/PII/无证据）与 fail-open（精排/审核/审计）分开。
4. **并发**：Redis 限流前置 + ZSET 优先级队列 + 断点续跑 + 熔断重试。
5. **评测**：本地 9 条用例（rag/tool/safety/context 四类）+ LangSmith 自动化评测（机制 code 断言 fail-close + 9 个 Jev judge fail-open）+ 双基线门禁 + 本地看板。
6. **RAG**：双路召回（pgvector HNSW top30 + GIN 倒排 top30）→ RRF(k=60) → cross-encoder 精排 → 最多 6 块证据 → 强制引用。

---

## 7. 工程工具链（npm scripts）

| 命令 | 作用 |
|---|---|
| `npm start` / `npm run dev` | 启动服务（http://localhost:3204）/ watch 模式 |
| `npm run db:init` / `db:check` | 建表 + 索引 + RULE + RLS + checkpoint schema / 列出带前缀的 DB 对象 |
| `npm run seed` | 灌 5 份知识文档 + 3 条语义记忆（两租户） |
| `npm run chat -- "问题"` | CLI 跑完整链路，打印上下文报告与审计链 |
| `npm run smoke` | 9 项端到端冒烟（fake 模型，不依赖外部 API） |
| `npm run eval` / `eval:smith` | 本地离线评测门禁 / LangSmith 自动化评测 |
| `npm test` | 22 项纯单元测试（node --test，不连库不调模型） |
| `npm run check:llm` / `check:smith` / `check:eval` | 校验模型端点与维度 / LangSmith 连通性 / 评测健康 |
| `npm run usage` | 查看 token 消耗与成本 |

---

## 8. 可迁移的工程约定（换项目时值得带走）

- **配置即策略**：所有阈值集中在 `src/config.js`，启动期 zod 校验；密钥单独 `getSecret()`，不进 config 对象、不进日志。
- **端点可信边界**：`assertSafeEndpoint()` 强制 HTTPS（仅本机可 http）、禁止 URL 夹带凭据。
- **模型可插拔**：OpenAI 兼容协议 + `fake` 确定性兜底，测试 / 冒烟 / 评测默认零成本离线可跑。
- **评测双基线**：`out/eval-baseline.json`（机制门禁）与 `out/eval-smith-baseline.json`（judge 门禁），参数变更必须过基线才能上线；judge fail-open、机制 fail-close。
- **DB 命名空间**：所有表 / 索引 / RULE / RLS 统一 `agent_harness_gongdan` 前缀，checkpointer 用独立 schema，方便共享库里多项目共存。
- **已知取舍**（生产需替换）：注入/PII 用正则基线（非 Lakera/Presidio）、RBAC 用策略表（非 Casbin）、精排有 cross-encoder 主路但 heuristic 兜底、预算账本为进程内存态（应换 Redis/PG 计数器 + OTel 告警）。
