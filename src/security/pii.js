/**
 * Presidio 的本地轻量替身：正则识别 + 可逆性不可恢复的脱敏。
 * 面试点：脱敏必须发生在「入模前」和「出模后」两处，只在入口做一次不够 ——
 * 工具返回值和 RAG 文档同样可能带 PII。
 */
const PATTERNS = [
  { type: "phone", re: /1[3-9]\d{9}/g, mask: (s) => s.replace(/(\d{3})\d{4}(\d{4})/, "$1****$2") },
  { type: "id_card", re: /\b\d{17}[\dXx]\b/g, mask: (s) => s.replace(/(\d{6})\d{8}(\d{3}[\dXx])/, "$1********$2") },
  { type: "email", re: /[\w.+-]+@[\w-]+\.[\w.]+/g, mask: (s) => s.replace(/^(.)(.*)(@.*)$/, "$1***$3") },
  { type: "bank_card", re: /\b\d{16,19}\b/g, mask: (s) => s.replace(/\d{12}(\d{4})/, "************$1") },
];

export function maskPII(text) {
  let masked = String(text ?? "");
  const hits = [];
  for (const p of PATTERNS) {
    const found = masked.match(p.re) ?? [];
    if (found.length) {
      hits.push({ type: p.type, count: found.length });
      masked = masked.replace(p.re, (m) => p.mask(m));
    }
  }
  // 详细地址：保留城市/区，隐藏门牌
  const addr = masked.match(/([\u4e00-\u9fa5]{2,8}(?:省|市|区|县)[\u4e00-\u9fa5\d]{2,20}(?:号|栋|室|楼))/g) ?? [];
  if (addr.length) {
    hits.push({ type: "address", count: addr.length });
    masked = masked.replace(/([\u4e00-\u9fa5]{2,8}(?:省|市|区|县))[\u4e00-\u9fa5\d]{2,20}(号|栋|室|楼)/g, "$1***$2");
  }
  return { masked, hits };
}

/** 工具参数脱敏：递归处理字符串，避免把手机号原样发给下游或写进 trace。 */
export function maskToolArgs(value) {
  if (typeof value === "string") return maskPII(value).masked;
  if (Array.isArray(value)) return value.map(maskToolArgs);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, maskToolArgs(v)]));
  }
  return value;
}

export function hasPII(text) {
  return maskPII(text).hits.length > 0;
}
