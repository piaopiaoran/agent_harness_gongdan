import { createEmbedder } from "../src/llm/embedding.js";
import { ingestDocs } from "../src/rag/ingest.js";
import { upsertFact } from "../src/db/queries.js";
import { closePool } from "../src/db/pool.js";

const embedder = createEmbedder();
console.log(`[seed] embedding=${embedder.kind} dims=${embedder.dims}`);

try {
  for (const tenantId of ["tenant-a", "tenant-b"]) {
    const stats = await ingestDocs(embedder, { tenantId });
    console.log(`[seed] ${tenantId}:`, stats);
  }

  // 预置几条语义记忆，演示「用户偏好 / 业务规则」与 RAG 文档来源分离。
  const seeds = [
    { tenantId: "tenant-a", userId: "u-1", fact: "用户 u-1 偏好原路退回，不接受店铺余额退款", category: "preference" },
    { tenantId: "tenant-a", userId: "u-1", fact: "用户 u-1 曾反馈静音降噪耳机 Pro 左耳无声音", category: "order" },
    { tenantId: "tenant-a", userId: "u-2", fact: "用户 u-2 关注物流时效，常询问订单进度", category: "preference" },
  ];
  for (const s of seeds) {
    const vec = await embedder.embedQuery(s.fact);
    await upsertFact({ ...s, embedding: vec, confidence: 0.9, sourceTraceId: "seed" });
  }
  console.log(`[seed] 预置 ${seeds.length} 条语义记忆（Mem0 风格事实表）`);
} catch (err) {
  console.error("[seed] 失败:", err.message);
  process.exitCode = 1;
} finally {
  await closePool();
}
