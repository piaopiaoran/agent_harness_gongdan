import { config } from "../config.js";

/**
 * token → 成本折算（USD / 1M tokens）。
 *
 * ⚠️ 内置价格表是**参考值**，不是实时账单：供应商会调价，渠道别名（如 deepseek-flash）
 * 的实际单价也可能不同。接生产账单必须用 MODEL_PRICING_JSON 覆盖，或对接供应商
 * billing API。未命中的模型返回 costUsd=null，绝不猜价。
 */
export const PRICING_NOTE = "内置价格为参考值（USD/1M tokens），生产请用 MODEL_PRICING_JSON 覆盖或对接供应商账单";

const BUILTIN_PRICING = Object.freeze({
  "gpt-4o-mini":     { input: 0.15, output: 0.60, cachedInput: 0.075 },
  "gpt-4o":          { input: 2.50, output: 10.0, cachedInput: 1.25 },
  "gpt-4.1-mini":    { input: 0.40, output: 1.60, cachedInput: 0.10 },
  "gpt-4.1":         { input: 2.00, output: 8.00, cachedInput: 0.50 },
  "deepseek-chat":   { input: 0.27, output: 1.10 },
  "deepseek-reasoner": { input: 0.55, output: 2.19 },
  // 项目 .env 使用的硅基流动别名；按实际套餐用 MODEL_PRICING_JSON 覆盖。
  "deepseek-flash":  { input: 0.27, output: 1.10 },
});

function num(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

function normalizeRate(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const input = num(raw.input ?? raw.inputPerM);
  const output = num(raw.output ?? raw.outputPerM);
  if (input === null || output === null) return null;
  const cachedInput = num(raw.cachedInput ?? raw.cachedInputPerM) ?? input;
  return { input, output, cachedInput };
}

/** 优先精确匹配；否则按最长前缀匹配（兼容 -2024-xx-xx 之类的日期后缀）。 */
export function resolveRate(model) {
  const name = String(model ?? "").trim();
  if (!name) return null;
  const sources = [
    ...Object.entries(config.pricing.overrides).map(([key, raw]) => ({ key, rate: normalizeRate(raw), source: "override" })),
    ...Object.entries(BUILTIN_PRICING).map(([key, raw]) => ({ key, rate: normalizeRate(raw), source: "builtin" })),
  ];
  const exact = sources.find((s) => s.key === name && s.rate);
  if (exact) return { ...exact.rate, model: exact.key, source: exact.source };
  const prefix = sources
    .filter((s) => s.rate && (name.startsWith(s.key) || s.key.startsWith(name)))
    .sort((a, b) => b.key.length - a.key.length)[0];
  return prefix ? { ...prefix.rate, model: prefix.key, source: prefix.source } : null;
}

function round(value) {
  return Math.round(value * 1e8) / 1e8;
}

/**
 * 用 LangChain usage_metadata 计价。
 *
 * - input/output 来自 provider usage，不用本地 estimateTokens。
 * - cached_read 按 cachedInput 单价，避免高估。
 * - 只有 total_tokens（fake 模型）不猜价：fake 记 0，real 记 unpriced。
 */
export function priceUsage(usage, model = config.LLM_MODEL, meta = {}) {
  const u = usage && typeof usage === "object" ? usage : {};
  const inputTokens = num(u.input_tokens) ?? 0;
  const outputTokens = num(u.output_tokens) ?? 0;
  const totalTokens = num(u.total_tokens) ?? inputTokens + outputTokens;
  const cachedTokens = Math.min(inputTokens, num(
    u.input_token_details?.cache_read ?? u.input_token_details?.cached_read
      ?? u.cached_tokens ?? u.prompt_tokens_details?.cached_tokens
  ) ?? 0);
  const unattributedTokens = Math.max(0, totalTokens - inputTokens - outputTokens);

  const base = {
    at: new Date().toISOString(),
    ...meta,
    model,
    tokens: { inputTokens, outputTokens, cachedTokens, unattributedTokens, totalTokens },
    currency: "USD",
    note: PRICING_NOTE,
  };

  if (model === "fake" || config.modelMode === "fake") {
    return { ...base, matchedModel: "fake", rates: null, costUsd: 0, source: "fake" };
  }
  if (inputTokens === 0 && outputTokens === 0) {
    return { ...base, matchedModel: null, rates: null, costUsd: null, source: "unpriced", reason: "usage_has_only_total_tokens" };
  }
  const rate = resolveRate(model);
  if (!rate) {
    return { ...base, matchedModel: null, rates: null, costUsd: null, source: "unpriced", reason: "model_not_in_pricing_table" };
  }
  const billableInput = inputTokens - cachedTokens;
  const costUsd = round(
    (billableInput * rate.input + cachedTokens * rate.cachedInput + outputTokens * rate.output) / 1e6
  );
  return {
    ...base,
    matchedModel: rate.model,
    rates: { inputPerM: rate.input, outputPerM: rate.output, cachedInputPerM: rate.cachedInput },
    costUsd,
    source: rate.source,
  };
}

/**
 * 进程级 token 预算：给蒸馏这类后台批处理单独设上限，避免上量后抢在线配额。
 * 只按自然日累计（内存态，重启清零），真实生产应换成 Redis / PG 计数器。
 */
export function createTokenBudget({ dailyLimit = config.distill.dailyTokenBudget, now = () => Date.now() } = {}) {
  let day = "";
  let used = 0;
  let records = 0;
  const sync = () => {
    const key = new Date(now()).toISOString().slice(0, 10);
    if (key !== day) { day = key; used = 0; records = 0; }
  };
  sync();
  return {
    canSpend(estimated = 0) {
      sync();
      return used + Math.max(0, Number(estimated) || 0) <= dailyLimit;
    },
    record(usage) {
      sync();
      const total = Number(usage?.total_tokens)
        ?? (Number(usage?.input_tokens) || 0) + (Number(usage?.output_tokens) || 0);
      used += Math.max(0, total);
      records += 1;
      return this.status();
    },
    status() {
      sync();
      return {
        date: day,
        dailyLimit,
        usedTokens: used,
        remainingTokens: Math.max(0, dailyLimit - used),
        calls: records,
      };
    },
  };
}

/** 每次请求的账本：逐次明细 + 会话级汇总。只在内存，不跨进程持久化。 */
export function createCostLedger(model = config.modelMode === "fake" ? "fake" : config.LLM_MODEL) {
  const invocations = [];
  return {
    model,
    add(usage, meta = {}) {
      const item = priceUsage(usage, meta.model ?? model, meta);
      invocations.push(item);
      return item;
    },
    summary() {
      const tokens = invocations.reduce((acc, i) => {
        acc.inputTokens += i.tokens.inputTokens;
        acc.outputTokens += i.tokens.outputTokens;
        acc.cachedTokens += i.tokens.cachedTokens;
        acc.totalTokens += i.tokens.totalTokens;
        return acc;
      }, { inputTokens: 0, outputTokens: 0, cachedTokens: 0, totalTokens: 0 });
      const priced = invocations.filter((i) => typeof i.costUsd === "number");
      return {
        currency: "USD",
        model,
        calls: invocations.length,
        pricedCalls: priced.length,
        unpricedCalls: invocations.length - priced.length,
        tokens,
        costUsd: round(priced.reduce((sum, i) => sum + i.costUsd, 0)),
        note: PRICING_NOTE,
        invocations,
      };
    },
  };
}
