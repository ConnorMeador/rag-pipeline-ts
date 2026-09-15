import { describe, expect, it } from 'vitest';
import { mean, ndcgAtK, recallAtK, reciprocalRankAtK } from '../src/metrics.js';

const judged = new Map([
  ['d1', 1],
  ['d2', 1],
]);

describe('recallAtK', () => {
  it('is the share of relevant docs found in the top k', () => {
    expect(recallAtK(['d1', 'x', 'y'], judged, 3)).toBe(0.5);
    expect(recallAtK(['x', 'd2', 'd1'], judged, 3)).toBe(1);
    expect(recallAtK(['x', 'd2', 'd1'], judged, 1)).toBe(0);
  });

  it('is 0 when nothing is judged relevant', () => {
    expect(recallAtK(['d1'], new Map(), 10)).toBe(0);
  });
});

describe('reciprocalRankAtK', () => {
  it('is 1 / rank of the first relevant doc', () => {
    expect(reciprocalRankAtK(['x', 'y', 'd2'], judged, 10)).toBeCloseTo(1 / 3);
  });

  it('is 0 when the first relevant doc falls outside k', () => {
    expect(reciprocalRankAtK(['x', 'y', 'd2'], judged, 2)).toBe(0);
  });
});

describe('ndcgAtK', () => {
  it('is 1 for a perfect ranking', () => {
    expect(ndcgAtK(['d1', 'd2', 'x'], judged, 10)).toBeCloseTo(1);
  });

  it('matches a hand-computed value', () => {
    // DCG = 1/log2(3) (d1 at rank 2); IDCG = 1 + 1/log2(3)
    const expected = 1 / Math.log2(3) / (1 + 1 / Math.log2(3));
    expect(ndcgAtK(['x', 'd1'], judged, 10)).toBeCloseTo(expected);
  });

  it('uses graded gains', () => {
    const graded = new Map([
      ['hi', 2],
      ['lo', 1],
    ]);
    expect(ndcgAtK(['hi', 'lo'], graded, 2)).toBeGreaterThan(ndcgAtK(['lo', 'hi'], graded, 2));
  });
});

describe('mean', () => {
  it('averages and handles an empty list', () => {
    expect(mean([1, 2, 3])).toBe(2);
    expect(mean([])).toBe(0);
  });
});
