import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import { Bm25Index } from './bm25.js';

export interface StoredChunk {
  id: string;
  text: string;
  embedding: number[];
  metadata?: Record<string, unknown>;
}

export interface SearchResult {
  chunk: StoredChunk;
  score: number;
}

interface StoreFile {
  version: 1;
  builtAt: string;
  chunks: StoredChunk[];
}

/** Cosine similarity between two vectors. Returns 0 (not NaN) for a zero vector. */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * A JSON-file vector store: load the whole array into memory, brute-force
 * cosine over it on every query. See the README for why this beats standing
 * up a vector database at this scale (under ~10k chunks).
 */
export class VectorStore {
  private chunks: StoredChunk[] = [];
  private builtAt: string = new Date(0).toISOString();

  static async load(path: string): Promise<VectorStore> {
    const store = new VectorStore();
    const raw = await readFile(path, 'utf8');
    const parsed = JSON.parse(raw) as StoreFile;
    store.chunks = parsed.chunks ?? [];
    store.builtAt = parsed.builtAt ?? store.builtAt;
    return store;
  }

  static fromChunks(chunks: StoredChunk[]): VectorStore {
    const store = new VectorStore();
    store.chunks = chunks;
    store.builtAt = new Date().toISOString();
    return store;
  }

  async save(path: string): Promise<void> {
    const payload: StoreFile = { version: 1, builtAt: this.builtAt, chunks: this.chunks };
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, JSON.stringify(payload), 'utf8');
  }

  get size(): number {
    return this.chunks.length;
  }

  /** Brute-force cosine search over every stored chunk. O(n) per query — fine below ~10k chunks. */
  search(queryEmbedding: readonly number[], k: number): SearchResult[] {
    if (k <= 0) return [];
    const scored = this.chunks.map((chunk) => ({
      chunk,
      score: cosineSimilarity(queryEmbedding, chunk.embedding),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  private keywordIndex?: Bm25Index;
  private chunksById?: Map<string, StoredChunk>;

  /**
   * BM25 keyword search over the same chunks. The index is built lazily on
   * the first call (chunks are fixed once a store is created or loaded), so
   * dense-only callers never pay for it. Scores are raw BM25, not cosine.
   */
  keywordSearch(query: string, k: number): SearchResult[] {
    if (k <= 0) return [];
    if (!this.keywordIndex || !this.chunksById) {
      this.keywordIndex = new Bm25Index(this.chunks);
      this.chunksById = new Map(this.chunks.map((c) => [c.id, c]));
    }
    const byId = this.chunksById;
    return this.keywordIndex.search(query, k).map((hit) => ({ chunk: byId.get(hit.id)!, score: hit.score }));
  }
}
