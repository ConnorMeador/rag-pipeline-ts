import { describe, expect, it } from 'vitest';
import type { Embedder } from '../src/embed.js';
import { reciprocalRankFusion } from '../src/hybrid.js';
import { ask } from '../src/index.js';
import { VectorStore } from '../src/store.js';

describe('reciprocalRankFusion', () => {
  it('rewards ids that rank well in both lists', () => {
    const fused = reciprocalRankFusion([
      [{ id: 'a' }, { id: 'b' }, { id: 'c' }],
      [{ id: 'b' }, { id: 'd' }, { id: 'a' }],
    ]);
    expect(fused.map((f) => f.id).slice(0, 2)).toEqual(['b', 'a']);
  });

  it('uses the 1 / (k + rank) formula with k = 60 by default', () => {
    const [only] = reciprocalRankFusion([[{ id: 'x' }]]);
    expect(only!.score).toBeCloseTo(1 / 61);
  });

  it('counts a duplicated id within one list once', () => {
    const [only] = reciprocalRankFusion([[{ id: 'x' }, { id: 'x' }]]);
    expect(only!.score).toBeCloseTo(1 / 61);
  });

  it('applies per-list weights', () => {
    const fused = reciprocalRankFusion([[{ id: 'a' }], [{ id: 'b' }]], { weights: [1, 3] });
    expect(fused[0]!.id).toBe('b');
  });
});

describe('ask() with hybrid retrieval', () => {
  // A fake embedder that only "understands" animals: every text lands on the
  // same axis unless it mentions cats or dogs. It can't tell error codes
  // apart, which is exactly the blind spot keyword search covers.
  const embedder: Embedder = {
    async embed(text) {
      return vectorFor(text);
    },
    async embedBatch(texts) {
      return texts.map(vectorFor);
    },
  };
  function vectorFor(text: string): number[] {
    const t = text.toLowerCase();
    return [t.includes('cat') ? 1 : 0, t.includes('dog') ? 1 : 0, 0.1];
  }

  const texts = {
    cats: 'Cats purr and nap.',
    dogs: 'Dogs bark at the mail carrier.',
    e1234: 'Error E1234 means the build cache is corrupt.',
    e9999: 'Error E9999 means the network is down.',
  };
  const store = VectorStore.fromChunks(
    Object.entries(texts).map(([id, text]) => ({ id, text, embedding: vectorFor(text) }))
  );

  it('dense retrieval alone cannot separate the two error docs', async () => {
    const { results } = await ask('what is E9999', { embedder, store, k: 4 });
    const e9999Score = results.find((r) => r.chunk.id === 'e9999')!.score;
    const e1234Score = results.find((r) => r.chunk.id === 'e1234')!.score;
    expect(e9999Score).toBeCloseTo(e1234Score);
  });

  it('hybrid retrieval puts the exact-token match first', async () => {
    const { results } = await ask('what is E9999', { embedder, store, k: 2, retrieval: 'hybrid' });
    expect(results[0]!.chunk.id).toBe('e9999');
    expect(results).toHaveLength(2);
  });
});
