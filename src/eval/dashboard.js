/**
 * 评测看板的纯数据层：把 eval_runs 的历史行按「评测路径」分组整理成指标趋势。
 *
 * 两条路径都会写 overall，但含义完全不同：
 *   eval:smith → (mechanism + faithfulness + decision) / 3
 *   npm run eval → 9 条用例的整体通过率
 * 因此 delta 与基线必须在各自轨道内计算，禁止跨轨合并成一条折线。
 *
 * 不碰数据库、不碰 LangSmith，便于单测；反向指标（泄露率）单独标注方向。
 */

// 高分=有问题的指标：数值上升是回归，不是进步。
export const REVERSE_KEYS = new Set(["judge_pii_leak"]);

// 展示顺序：先总览与机制分，再 RAG 指标，其余按名字排。
const SMITH_ORDER = [
  "overall", "mechanism",
  "judge_faithfulness", "judge_decision", "judge_task_completed",
  "judge_helpfulness", "judge_context_precision",
  "judge_context_recall", "judge_answer_correctness",
  "judge_pii_leak",
];
const MECHANISM_ORDER = ["overall", "rag", "tool", "safety", "context"];

export const TRACKS = [
  { id: "smith", label: "语义评测 · eval:smith", hint: "真实模型 + Jev judge；overall = (mechanism + faithfulness + decision) / 3" },
  { id: "mechanism", label: "机制评测 · npm run eval", hint: "代码断言（fake / real 模型）；overall = 9 条用例通过率" },
  { id: "other", label: "其他历史记录", hint: "无法识别评测路径的行，单独展示以免污染上面两轨" },
];

const finite = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});

/** 按 runName 前缀优先、指标集合兜底，判断这条记录属于哪条评测路径。 */
function trackOf(run) {
  const name = String(run.run_name ?? run.runName ?? "");
  const keys = Object.keys(obj(run.metrics));
  if (name.startsWith("smith-judge-") || keys.some((k) => k.startsWith("judge_")) || keys.includes("mechanism")) {
    return "smith";
  }
  if (name.startsWith("harness-eval-") || ["rag", "tool", "safety", "context"].some((k) => keys.includes(k))) {
    return "mechanism";
  }
  return "other";
}

function sortKeys(keys, order) {
  const rank = (k) => {
    const i = order.indexOf(k);
    return i === -1 ? order.length : i;
  };
  return [...keys].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

/** 同一轨道内的指标趋势；baseline 取该轨最新一条记录写库时的基线。 */
function seriesFor(runs, order) {
  const keys = sortKeys(new Set(runs.flatMap((r) => Object.keys(r.metrics))), order);
  // 图表从左到右按时间正序；runs 本身保持最新在前，便于列表展示。
  const timeline = [...runs].reverse();
  const baseline = obj(runs[0]?.baseline);

  return keys.map((key) => {
    const points = timeline
      .map((r) => ({
        runId: r.id,
        runName: r.runName,
        createdAt: r.createdAt,
        value: finite(r.metrics[key]),
      }))
      // null（judge 失败 / 无 golden 标注 / 该轨没有这个 key）不画点，也不参与 delta。
      .filter((p) => p.value !== null);

    const latest = points.at(-1)?.value ?? null;
    const previous = points.at(-2)?.value ?? null;
    const delta = latest !== null && previous !== null
      ? Number((latest - previous).toFixed(4))
      : null;
    const reverse = REVERSE_KEYS.has(key);
    const improved = delta === null ? null : (reverse ? delta < 0 : delta > 0);

    return {
      key,
      reverse,
      latest,
      previous,
      delta,
      improved,
      baseline: finite(baseline[key]),
      points,
    };
  });
}

/**
 * @param {Array<{id, run_name, dataset, metrics, baseline, passed, created_at}>} rows
 *        按 created_at DESC 返回的数据库行。
 * @returns {{runs, keys, series, groups, latest, count}}
 */
export function buildEvalDashboard(rows = []) {
  const runs = (Array.isArray(rows) ? rows : []).map((r) => {
    const metrics = obj(r.metrics);
    const normalized = {
      id: r.id ?? "",
      runName: String(r.run_name ?? r.runName ?? ""),
      dataset: String(r.dataset ?? ""),
      metrics,
      baseline: obj(r.baseline),
      passed: Boolean(r.passed),
      createdAt: r.created_at ?? r.createdAt ?? null,
    };
    return { ...normalized, track: trackOf(normalized) };
  });

  const byTrack = new Map(TRACKS.map((t) => [t.id, []]));
  for (const r of runs) byTrack.get(r.track).push(r);

  const groups = TRACKS
    .map((t) => {
      const trackRuns = byTrack.get(t.id);
      return {
        id: t.id,
        label: t.label,
        hint: t.hint,
        runs: trackRuns,
        count: trackRuns.length,
        latest: trackRuns[0] ?? null,
        series: seriesFor(trackRuns, t.id === "mechanism" ? MECHANISM_ORDER : SMITH_ORDER),
      };
    })
    .filter((g) => g.count > 0);

  // 顶层 series 保留给兼容调用方：key 加轨道前缀，避免 overall 两套口径再被合并。
  // 前端看板直接使用 groups，按轨渲染。
  const series = groups.flatMap((g) => g.series.map((s) => ({
    ...s,
    track: g.id,
    key: `${g.id}:${s.key}`,
    displayKey: s.key,
  })));

  return {
    runs,
    keys: series.map((s) => s.key),
    series,
    groups,
    latest: runs[0] ?? null,
    count: runs.length,
  };
}
