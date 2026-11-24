import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { config } from "../config.js";
import { upsertDocument, insertChunks } from "../db/queries.js";
import { searchText, segment } from "./segment.js";

const here = dirname(fileURLToPath(import.meta.url));
const docsDir = join(here, "..", "..", "data", "docs");

/** 按段落切块，chunk 512 token / overlap 15%；不在表格与代码块中间切断。 */
export function chunkDocument(body, { maxTokens = config.RAG_CHUNK_TOKENS, overlap = config.RAG_CHUNK_OVERLAP } = {}) {
  const lines = body.split(/\n+/).map((s) => s.trim()).filter(Boolean);
  const blocks = [];
  let current = "";
  for (const line of lines) {
    const candidate = current ? `${current}\n${line}` : line;
    if (segment(candidate).length * 1.6 > maxTokens && current) {
      blocks.push(current);
      current = line;
    } else {
      current = candidate;
    }
  }
  if (current) blocks.push(current);

  const overlapCount = Math.max(1, Math.floor(blocks.length * overlap));
  const chunks = [];
  for (let i = 0; i < blocks.length; i += 1) {
    const start = Math.max(0, i - overlapCount);
    chunks.push(blocks.slice(start, i + 1).join("\n"));
  }
  return chunks.map((content, i) => ({ content, chunkIdx: i }));
}

export async function ingestDocs(embedder, { tenantId = "tenant-a" } = {}) {
  const files = (await readdir(docsDir)).filter((f) => f.endsWith(".md")).sort();
  const vectors = [];
  const stats = { files: 0, chunks: 0, skipped: 0, version: 0 };

  for (const file of files) {
    const body = await readFile(join(docsDir, file), "utf8");
    const hash = createHash("sha256").update(body).digest("hex");
    const doc = await upsertDocument({
      tenantId, source: `data/docs/${file}`, title: file.replace(/^\d+-|\.md$/g, ""), contentHash: hash,
    });
    if (!doc.changed) {
      stats.skipped += 1;
      continue;
    }
    stats.files += 1;
    stats.version = Math.max(stats.version, doc.version);
    const chunks = chunkDocument(body);
    vectors.push(...chunks.map((c) => ({ ...c, docId: doc.id, tenantId, body })));
  }

  if (vectors.length) {
    const texts = vectors.map((v) => v.content);
    const embeddings = await embedder.embedDocuments(texts);
    const rows = vectors.map((v, i) => ({
      docId: v.docId,
      tenantId: v.tenantId,
      chunkIdx: v.chunkIdx,
      content: v.content,
      tokens: segment(v.content).length,
      searchText: searchText(v.content),
      embedding: embeddings[i],
      contentHash: createHash("sha256").update(v.content).digest("hex"),
    }));
    await insertChunks(rows);
    stats.chunks = rows.length;
  }
  return stats;
}
