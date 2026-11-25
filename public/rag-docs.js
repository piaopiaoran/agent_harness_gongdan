const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
$("tenantId").value = params.get("tenant") || "tenant-a";
let cache = null;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function render(data) {
  cache = data;
  const toc = $("toc");
  const docsBox = $("docs");
  toc.replaceChildren();
  docsBox.replaceChildren();

  const all = data?.documents ?? [];
  const kw = $("filter").value.trim().toLowerCase();
  const docs = all.filter((d) => !kw ||
    `${d.title} ${d.source} ${(d.chunks ?? []).map((c) => c.content).join(" ")}`.toLowerCase().includes(kw));

  $("count").textContent = `${docs.length}/${all.length}`;
  const bar = $("runtime");
  bar.replaceChildren(
    el("span", "badge ok", `tenant: ${data?.tenantId ?? "—"}`),
    el("span", "badge", `${all.length} 份文档`),
    el("span", "badge", `${data?.chunkCount ?? 0} 个 chunk`),
  );

  if (!all.length) {
    docsBox.append(el("p", "empty", "知识库为空 · 运行 npm run seed 灌入文档后刷新"));
    return;
  }
  if (!docs.length) {
    docsBox.append(el("p", "empty", "没有匹配的文档 · 清空搜索词重试"));
    return;
  }

  docs.forEach((d, i) => {
    const nav = el("button", "toc-item");
    nav.type = "button";
    nav.append(el("b", "", d.title), el("small", "", `${d.chunks.length} 块 · v${d.version}`));
    nav.onclick = () => docsBox.querySelectorAll(".docs-doc")[i]?.scrollIntoView({ behavior: "smooth", block: "start" });
    toc.append(nav);

    const article = el("article", "docs-doc");
    article.id = `doc-${i}`;
    const head = el("div", "docs-doc-head");
    head.append(el("h2", "", d.title),
      el("div", "docs-doc-meta", `${d.source} · v${d.version} · ${d.chunks.length} 块 · ${d.acl?.length ? `ACL ${d.acl.join(", ")}` : "无 ACL 限制"}`));
    article.append(head);

    for (const c of d.chunks ?? []) {
      const chunk = el("section", "docs-chunk");
      chunk.append(el("div", "docs-chunk-head", `chunk #${c.chunkIdx} · ${c.tokens} tok`));
      chunk.append(el("pre", "docs-chunk-text", c.content));
      article.append(chunk);
    }
    docsBox.append(article);
  });
}

async function load() {
  const box = $("docs");
  box.replaceChildren(el("p", "empty", "加载中…"));
  const tenantId = $("tenantId").value.trim() || "tenant-a";
  history.replaceState(null, "", `?tenant=${encodeURIComponent(tenantId)}`);
  try {
    const res = await fetch("/api/rag/documents", { headers: { "x-tenant-id": tenantId } });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message ?? data.error ?? `HTTP ${res.status}`);
    render(data);
  } catch (err) {
    $("runtime").replaceChildren(el("span", "badge warn", "加载失败"));
    $("count").textContent = "";
    $("toc").replaceChildren();
    box.replaceChildren(el("p", "empty", `知识库不可用：${err.message}`));
  }
}

$("refresh").onclick = load;
$("filter").addEventListener("input", () => render(cache));
$("tenantId").addEventListener("change", load);
load();
