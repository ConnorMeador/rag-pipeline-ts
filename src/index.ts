import { extname, join } from 'node:path';
import { readdir, readFile } from 'node:fs/promises';

import { chunkText } from './chunk.js';
import type { Embedder } from './embed.js';
import { rerank as rerankCandidates } from './rerank.js';
import type { RerankOptions } from './rerank.js';
import { VectorStore } from './store.js';
import type { SearchResult, StoredChunk } from './store.js';

export { chunkText } from './chunk.js';
export type { ChunkOptions, TextChunk } from './chunk.js';
export { OllamaEmbedder, OpenAICompatEmbedder } from './embed.js';
export type { Embedder } from './embed.js';
export { cosineSimilarity, VectorStore } from './store.js';
export type { SearchResult, StoredChunk } from './store.js';
export { rerank } from './rerank.js';
export type { RerankOptions } from './rerank.js';

const TEXT_EXTENSIONS = new Set(['.md', '.mdx', '.txt']);

export interface IngestOptions {
  embedder: Embedder;
  maxChars?: number;
  overlapChars?: number;
}

/**
 * Read every `.md` / `.mdx` / `.txt` file in `dir`, chunk it, embed every
 * chunk, and return an in-memory VectorStore. Call `store.save(path)`
 * yourself if you want to persist the index to disk.
 */
export async function ingest(dir: string, options: IngestOptions): Promise<VectorStore> {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = entries.filter((e) => e.isFile() && TEXT_EXTENSIONS.has(extname(e.name)));

  const chunks: StoredChunk[] = [];
  for (const file of files) {
    const raw = await readFile(join(dir, file.name), 'utf8');
    const pieces = chunkText(raw, { maxChars: options.maxChars, overlapChars: options.overlapChars });
    for (const piece of pieces) {
      chunks.push({
        id: `${file.name}#${piece.index}`,
        text: piece.text,
        embedding: [],
        metadata: { source: file.name, charStart: piece.charStart, charEnd: piece.charEnd },
      });
    }
  }

  if (chunks.length === 0) return VectorStore.fromChunks([]);

  const embeddings = await options.embedder.embedBatch(chunks.map((c) => c.text));
  chunks.forEach((chunk, i) => {
    chunk.embedding = embeddings[i]!;
  });

  return VectorStore.fromChunks(chunks);
}

export interface AskOptions {
  embedder: Embedder;
  store: VectorStore;
  /** Number of results to return. Default: 5. */
  k?: number;
  /** Opt-in second-stage LLM rerank (see rerank.ts). Default: false. */
  rerank?: boolean;
  rerankOptions?: RerankOptions;
  /** Cosine candidate pool size fed into the reranker. Default: max(k * 4, 20). */
  candidatePoolSize?: number;
}

export interface AskResult {
  query: string;
  results: SearchResult[];
}

/**
 * Embed `question`, retrieve the top candidates by cosine similarity, and —
 * only if `rerank: true` and `rerankOptions` are both supplied — run an LLM
 * second pass over the candidate pool. Rerank failures never surface here;
 * see rerank.ts's fail-open behavior.
 */
export async function ask(question: string, options: AskOptions): Promise<AskResult> {
  const k = options.k ?? 5;
  const poolSize = options.candidatePoolSize ?? Math.max(k * 4, 20);

  const queryEmbedding = await options.embedder.embed(question);
  const candidates = options.store.search(queryEmbedding, poolSize);

  const results =
    options.rerank && options.rerankOptions
      ? await rerankCandidates(question, candidates, k, options.rerankOptions)
      : candidates.slice(0, k);

  return { query: question, results };
}
