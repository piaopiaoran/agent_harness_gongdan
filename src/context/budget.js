/**
 * 上下文预算：有效上下文预算 = 「比例 + 硬顶」双约束。
 *
 * 决策背景（2026-11-26 修订）：
 * - deepseek-flash 对应 DeepSeek-V4.1-Flash，标称窗口 1,048,576、最大输出 384K；
 *   1M 是容量上限，不是可用容量，Chroma 的对照实验显示所有模型都在窗口远未填满时开始掉准确率。
 * - 因此窗口只作不可越过的上限，实际分配由「比例 × A」与「分区硬顶」取小值决定：
 *   窗口变大只抬上限，不会让前缀、记忆、轨迹无限膨胀。
 * - A = W − O − S（输出预留 + 10% 安全余量）。O 随档位不同：思考 token 计入输出。
 *
 * 一档一策：不为每个请求按最贵档付费。
 * - interactive：默认档，装配量 30K 上下，首字延迟最优。
 * - analysis：复杂多步任务，总量约 96K，各分区取硬顶。
 * - longdoc：读长 PDF / 大知识库才用，允许 200K 以上。
 * 长文档档只能显式升档，不靠启发式自动进入，避免成本失控。
 */

/** 分区比例：六个区共享同一窗口，总和 < 1，剩余部分不显式分配（留白是结果，不是目标）。 */
const RATIOS = Object.freeze({
  system: 0.15,   // 固定前缀区
  state: 0.035,   // Agent State 信封
  memory: 0.12,   // 记忆召回（情景 + 语义）
  rag: 0.10,      // RAG 证据（与记忆合计约 0.22）
  recent: 0.32,   // 最近原文轨迹
  remote: 0.09,   // 远端压缩摘要
});

/** 各档硬顶（token）。硬顶不随窗口变化，是"有效上下文"的经验上界，需按实测衰减曲线校准。 */
const CAPS = Object.freeze({
  interactive: Object.freeze({
    system: 8192, state: 2048, memory: 4096, rag: 4096,
    recent: 10240, remote: 2048, headroom: 8192, outputReserve: 16384,
  }),
  analysis: Object.freeze({
    system: 8192, state: 4096, memory: 12288, rag: 12288,
    recent: 49152, remote: 8192, headroom: 16384, outputReserve: 65536,
  }),
  longdoc: Object.freeze({
    system: 8192, state: 4096, memory: 24576, rag: 24576,
    recent: 122880, remote: 16384, headroom: 32768, outputReserve: 65536,
  }),
});

export const CONTEXT_PROFILES = Object.freeze(["interactive", "analysis", "longdoc"]);

/**
 * 升档关键词：只在用户明确要求"读全文 / 整份 / 长文档"时才进入 analysis。
 * longdoc 不在启发式里出现——最贵档只能显式指定，自动升档会让成本不可控。
 */
const ANALYSIS_HINTS = /(全文|整份|整篇|长文档|完整报告|通读|逐页|pdf|手册|规范全文|代码库)/i;

/**
 * 选择档位。requested 来自 CONTEXT_PROFILE（auto 时走启发式）。
 * longdoc 只能显式指定：自动升到最贵档会让成本不可控。
 */
export function selectContextProfile(question = "", requested = "auto") {
  if (requested && requested !== "auto") return requested;
  const q = String(question ?? "");
  if (q.length > 2000 || ANALYSIS_HINTS.test(q)) return "analysis";
  return "interactive";
}

/**
 * 计算某档的实际预算。
 * @param {object} p
 * @param {number} p.window 模型可用输入窗口（不是宣传窗口）
 * @param {number} p.safetyMargin 安全余量比例，默认 0.10
 * @param {string} p.profile interactive | analysis | longdoc
 * @param {object} [p.capOverrides] 覆盖分区硬顶，用于实测后收敛
 */
export function computeBudget({ window, safetyMargin = 0.10, profile = "interactive", capOverrides = {} }) {
  if (!CONTEXT_PROFILES.includes(profile)) throw new Error(`unknown context profile: ${profile}`);
  const caps = { ...CAPS[profile], ...capOverrides };
  const outputReserve = caps.outputReserve;
  const usable = Math.max(0, Math.floor(window - outputReserve));
  const available = Math.floor(usable * (1 - safetyMargin)); // A
  const region = (name) => Math.max(0, Math.min(caps[name], Math.floor(available * RATIOS[name])));
  const budget = {
    profile,
    window,
    outputReserve,
    safetyMargin,
    usable,
    available,
    ratios: { ...RATIOS },
    caps: { ...caps },
    system: region("system"),
    state: region("state"),
    memory: region("memory"),
    rag: region("rag"),
    recent: region("recent"),
    remote: region("remote"),
    headroom: caps.headroom,
    // 装配量上限：六个区合计，用于判定"这一档到底能装多少"
    assemblyCap: ["system", "state", "memory", "rag", "recent", "remote"]
      .reduce((sum, k) => sum + region(k), 0),
    // 与窗口的关系：总量 = 装配 + 缓冲，必须留有余量
    overflowAt: caps.headroom,
  };
  return Object.freeze(budget);
}

/** 一次性算好三档，供 /health 与审计展示。 */
export function computeAllProfiles(opts) {
  return Object.freeze(Object.fromEntries(
    CONTEXT_PROFILES.map((p) => [p, computeBudget({ ...opts, profile: p })])
  ));
}
