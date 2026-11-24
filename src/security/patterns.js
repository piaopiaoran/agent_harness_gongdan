import { config } from "../config.js";

/** L2 预清洗：零宽字符 / 双向控制符是常见的 prompt 注入隐藏手段。 */
export function normalizeInput(raw) {
  let text = String(raw ?? "");
  text = text.normalize("NFKC");
  text = text.replace(/[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]/g, "");
  text = text.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, " ");
  text = text.replace(/[ \t]{2,}/g, " ").trim();
  const truncated = text.length > config.MAX_INPUT_CHARS;
  return { text: text.slice(0, config.MAX_INPUT_CHARS), truncated };
}

/** L3 注入检测：Lakera 的本地基线替身（关键词 + 结构特征），命中即 fail-close。 */
const INJECTION_PATTERNS = [
  { name: "instruction_override", re: /忽略(之前|上面|以上|所有)|ignore (all )?(previous|above)|forget (all )?instructions/i },
  { name: "role_switch", re: /你现在是|切换到.*模式|developer mode|act as (an? )?(admin|system|developer)/i },
  { name: "prompt_exfiltration", re: /输出(你的)?系统提示|重复.*系统消息|reveal (your )?system prompt|print.*prompt/i },
  { name: "jailbreak", re: /\bDAN\b|越狱|绕过(安全|限制|规则)|bypass (safety|restriction)/i },
  { name: "privilege_escalation", re: /我是管理员|以管理员身份|grant me admin|提升权限/i },
  { name: "data_exfiltration", re: /导出全部|所有订单|泄露.*(用户|客户)|dump (all|every)/i },
];

export function detectInjection(text) {
  const flags = [];
  for (const p of INJECTION_PATTERNS) {
    if (p.re.test(text)) flags.push(p.name);
  }
  return flags;
}

/** 把检索文档当数据而非指令：去除文档里伪造的「系统指令」结构。 */
export function sanitizeDocument(text) {
  return String(text ?? "")
    .replace(/<\s*\/?\s*(system|assistant|instructions?)\s*>/gi, "[文档标记]")
    .replace(/^\s*(system|assistant)\s*:/gim, "[文档标记]");
}
