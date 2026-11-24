import { config, getSecret } from "../src/config.js";

/** 启动期外部依赖自检：只打印端点与状态，绝不打印密钥。 */
async function check(label, url, options) {
  const started = Date.now();
  try {
    const res = await fetch(url, { ...options, signal: AbortSignal.timeout(15000) });
    const text = await res.text();
    let parsed = null;
    try { parsed = JSON.parse(text); } catch { /* 非 JSON 也照常报告状态码 */ }
    console.log(`[check] ${label}: HTTP ${res.status} · ${Date.now() - started}ms`);
    return { ok: res.ok, parsed };
  } catch (err) {
    console.error(`[check] ${label}: 失败 · ${err.message}`);
    return { ok: false, parsed: null };
  }
}

const llm = await check("LLM /chat/completions", `${config.LLM_BASE_URL}/chat/completions`, {
  method: "POST",
  headers: { "Content-Type": "application/json", Authorization: `Bearer ${getSecret("llm")}` },
  body: JSON.stringify({ model: config.LLM_MODEL, messages: [{ role: "user", content: "ping" }], max_tokens: 8 }),
});

const dim = config.EMBEDDING_DIMS;
const emb = config.embeddingMode === "siliconflow"
  ? await check("Embedding /embeddings", `${config.EMBEDDING_BASE_URL}/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${getSecret("embedding")}` },
      // bge-m3 原生 1024 维， SiliconFlow 不接受 dimensions 参数
      body: JSON.stringify({ model: config.EMBEDDING_MODEL, input: "售后退款政策" }),
    })
  : { ok: true, parsed: null };

const vectorLen = emb.parsed?.data?.[0]?.embedding?.length ?? null;
if (vectorLen != null) {
  console.log(`[check] embedding dims: ${vectorLen}（配置 ${dim}）${vectorLen === dim ? " ✓" : " ✗ 必须重建索引"}`);
}

const llmOk = llm.ok || config.modelMode === "fake";
const embOk = emb.ok && (vectorLen == null || vectorLen === dim);
process.exit(llmOk && embOk ? 0 : 1);
