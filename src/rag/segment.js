/**
 * jieba 风格轻量分词：自定义词典最长匹配 + 未登录词 bigram + 英文数字连续 token。
 * 对应 Harness 风险台账「jieba 专名切分不准 → 自定义词典，保留英文数字连续 token」。
 * 这里不引 native 依赖，保证面试演示零编译成本；生产可无痛替换为 nodejieba。
 */
const DICT = [
  "七天无理由", "七天", "无理由退货", "退换货", "退款", "退货", "换货", "保修", "维修",
  "质量问题", "物流", "签收", "运输中", "订单号", "订单", "发票", "支付", "补偿",
  "耳机", "音箱", "键盘", "左耳", "右耳", "蓝牙", "降噪", "人工客服", "售后",
  "政策", "规定", "时效", "运费", "原路退回", "审核", "工单", "赔付",
];

export function segment(text) {
  const input = String(text ?? "").toLowerCase();
  const tokens = [];
  let buffer = "";

  const flush = () => {
    if (!buffer) return;
    // 先按词典最长匹配切
    let i = 0;
    while (i < buffer.length) {
      const hit = DICT.find((w) => buffer.startsWith(w, i));
      if (hit) {
        tokens.push(hit);
        i += hit.length;
        continue;
      }
      const ch = buffer[i];
      if (/[a-z0-9]/.test(ch)) {
        let j = i + 1;
        while (j < buffer.length && /[a-z0-9._-]/.test(buffer[j])) j += 1;
        tokens.push(buffer.slice(i, j));
        i = j;
        continue;
      }
      if (/[\u4e00-\u9fa5]/.test(ch)) {
        tokens.push(buffer.slice(i, i + 2)); // 未登录词：bigram 兜底
        i += 1; // 步进 1 保留重叠召回
      } else {
        i += 1;
      }
    }
    buffer = "";
  };

  for (const ch of input) {
    if (/[\u4e00-\u9fa5a-z0-9._-]/.test(ch)) buffer += ch;
    else flush();
  }
  flush();
  return [...new Set(tokens.filter(Boolean))];
}

export function searchText(text) {
  return segment(text).join(" ");
}

/** 词法重合度：给精排阶段用的廉价特征（cross-encoder 的本地基线替身）。 */
export function lexicalOverlap(queryTokens, content) {
  if (!queryTokens.length) return 0;
  const set = new Set(segment(content));
  const hit = queryTokens.filter((t) => set.has(t)).length;
  return hit / queryTokens.length;
}

/** 粗估 token 数：中文按 1 字 ≈ 1 token，英文按 4 字符 ≈ 1 token。 */
export function estimateTokens(text) {
  const s = String(text ?? "");
  const cjk = (s.match(/[\u4e00-\u9fa5]/g) ?? []).length;
  const rest = s.length - cjk;
  return Math.ceil(cjk + rest / 4);
}
