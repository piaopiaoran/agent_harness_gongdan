/**
 * 知识库文档目录的纯映射层：把 DB 行整理成前端可直接渲染的结构。
 * tenant 过滤与 active 版本过滤在 SQL 层完成，这里不重复做权限判断。
 */
export function buildDocumentCatalog(documents = [], chunks = []) {
  const byId = new Map();
  const out = [];

  for (const d of documents) {
    const doc = {
      id: d.id,
      source: d.source,
      title: d.title,
      version: d.version,
      acl: d.acl ?? [],
      createdAt: d.created_at ?? null,
      chunks: [],
    };
    byId.set(doc.id, doc);
    out.push(doc);
  }

  let chunkCount = 0;
  for (const c of chunks) {
    const doc = byId.get(c.doc_id);
    if (!doc) continue; // 孤立 chunk 不进目录（正常只会出现在清理中途）
    doc.chunks.push({
      id: c.id,
      chunkIdx: c.chunk_idx,
      content: c.content,
      tokens: c.tokens ?? 0,
      createdAt: c.created_at ?? null,
    });
    chunkCount += 1;
  }

  for (const doc of out) doc.chunks.sort((a, b) => a.chunkIdx - b.chunkIdx);
  out.sort((a, b) => String(a.source).localeCompare(String(b.source)) || b.version - a.version);

  return { count: out.length, chunkCount, documents: out };
}
