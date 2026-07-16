import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { cosineSimilarity, VectorStore } from '../src/store.js';
import type { StoredChunk } from '../src/store.js';

function unitVector(dims: number, hot: number): number[] {
  const v = new Array(dims).fill(0);
  v[hot] = 1;
  return v;
}

describe('cosineSimilarity', () => {
  it('is 1 for identical vectors', () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1);
  });

  it('is 0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
  });

  it('is -1 for opposite vectors', () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1);
  });

  it('returns 0 for a zero vector instead of NaN', () => {
    expect(cosineSimilarity([0, 0], [1, 1])).toBe(0);
  });
});

describe('VectorStore', () => {
  const chunks: StoredChunk[] = [
    { id: 'a', text: 'about cats', embedding: unitVector(4, 0) },
    { id: 'b', text: 'about dogs', embedding: unitVector(4, 1) },
    { id: 'c', text: 'about birds', embedding: unitVector(4, 2) },
    { id: 'd', text: 'near cats', embedding: [0.9, 0.1, 0, 0] },
  ];

  it('ranks the most similar chunk first', () => {
    const store = VectorStore.fromChunks(chunks);
    const results = store.search(unitVector(4, 0), 2);
    expect(results[0]!.chunk.id).toBe('a');
    expect(results[1]!.chunk.id).toBe('d');
    expect(results[0]!.score).toBeGreaterThan(results[1]!.score);
  });

  it('returns at most k results', () => {
    const store = VectorStore.fromChunks(chunks);
    expect(store.search(unitVector(4, 0), 2)).toHaveLength(2);
    expect(store.search(unitVector(4, 0), 100)).toHaveLength(chunks.length);
  });

  it('returns an empty array for k <= 0', () => {
    const store = VectorStore.fromChunks(chunks);
    expect(store.search(unitVector(4, 0), 0)).toEqual([]);
  });

  it('round-trips through save/load on disk', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'rag-store-'));
    const path = join(dir, 'index.json');
    try {
      const store = VectorStore.fromChunks(chunks);
      await store.save(path);
      const loaded = await VectorStore.load(path);
      expect(loaded.size).toBe(chunks.length);
      const results = loaded.search(unitVector(4, 0), 1);
      expect(results[0]!.chunk.id).toBe('a');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
