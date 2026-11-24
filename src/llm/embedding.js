import { OpenAIEmbeddings } from "@langchain/openai";
import { config, getSecret } from "../config.js";

/**
 * 确定性哈希向量：零密钥零成本，只保证链路可跑通，不代表语义相似度。
 * 维度必须与 real embedding（bge-m3 = 1024）和 SQL vector(1024) 严格一致，
 * 否则换模型必须重建索引 —— Harness 标准明令禁止「换 embedding 不重建索引」。
 */
export function hashEmbedding(text, dims = config.EMBEDDING_DIMS) {
  const vector = new Array(dims).fill(0);
  const normalized = String(text ?? "")
    .toLowerCase()
    .replace(/[\s，。！？、；：,.!?;:()"'`~《》【】\[\]{}]/g, " ")
    .trim();

  const tokens = [];
  for (const segment of normalized.split(/\s+/)) {
    if (!segment) continue;
    if (/^[a-z0-9]+$/.test(segment)) {
      tokens.push(segment);
      continue;
    }
    if (segment.length === 1) tokens.push(segment);
    else for (let i = 0; i < segment.length - 1; i += 1) tokens.push(segment.slice(i, i + 2));
  }

  for (const token of tokens) {
    let hash = 2166136261;
    for (let i = 0; i < token.length; i += 1) {
      hash ^= token.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    const index = (hash >>> 0) % dims;
    const sign = ((hash >>> 16) & 1) === 1 ? 1 : -1;
    vector[index] += sign;
  }
  const norm = Math.sqrt(vector.reduce((s, v) => s + v * v, 0)) || 1;
  return vector.map((v) => Number((v / norm).toFixed(6)));
}

export function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let score = 0;
  for (let i = 0; i < a.length; i += 1) score += a[i] * b[i];
  return score;
}

export function createEmbedder() {
  if (config.embeddingMode === "siliconflow") {
    // SiliconFlow /v1/embeddings 不接受 dimensions 参数：bge-m3 原生就是 1024 维。
    const model = new OpenAIEmbeddings({
      model: config.EMBEDDING_MODEL,
      apiKey: getSecret("embedding"),
      configuration: { baseURL: config.EMBEDDING_BASE_URL },
      batchSize: config.EMBEDDING_BATCH_SIZE,
      timeout: config.LLM_TIMEOUT_MS,
      maxRetries: config.LLM_MAX_RETRIES,
    });
    return {
      kind: "siliconflow",
      dims: config.EMBEDDING_DIMS,
      embedQuery: async (text) => normalize(await model.embedQuery(text)),
      embedDocuments: async (texts) => (await model.embedDocuments(texts)).map(normalize),
    };
  }

  return {
    kind: "fake",
    dims: config.EMBEDDING_DIMS,
    embedQuery: async (text) => hashEmbedding(text),
    embedDocuments: async (texts) => texts.map((t) => hashEmbedding(t)),
  };
}

function normalize(vec) {
  const out = Array.from(vec ?? []);
  if (out.length !== config.EMBEDDING_DIMS) {
    throw new Error(`[embedding] 维度不匹配：期望 ${config.EMBEDDING_DIMS}，实际 ${out.length}；换模型必须重建索引`);
  }
  const norm = Math.sqrt(out.reduce((s, v) => s + v * v, 0)) || 1;
  return out.map((v) => Number((v / norm).toFixed(6)));
}
