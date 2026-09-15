/**
 * Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).
 *
 * Merges ranked lists from retrievers whose scores live on different scales
 * — cosine similarity is in [-1, 1], BM25 is unbounded — by ignoring the
 * scores entirely and using only rank: score(d) = Σ 1 / (k + rank_i(d)).
 * No per-corpus score normalization to tune, which is why it's the default
 * fusion method in most hybrid search engines.
 */

export interface RankedId {
  id: string;
}

export interface FusedResult {
  id: string;
  score: number;
}

export interface RrfOptions {
  /** Rank-smoothing constant. Default: 60, the value from the original paper. */
  k?: number;
  /** Optional per-list weights, same order as `lists`. Default: all 1. */
  weights?: readonly number[];
}

export function reciprocalRankFusion(
  lists: ReadonlyArray<readonly RankedId[]>,
  options: RrfOptions = {}
): FusedResult[] {
  const k = options.k ?? 60;
  const scores = new Map<string, number>();

  lists.forEach((list, listIndex) => {
    const weight = options.weights?.[listIndex] ?? 1;
    const seen = new Set<string>();
    list.forEach((item, rank) => {
      // A retriever listing the same id twice shouldn't count it twice.
      if (seen.has(item.id)) return;
      seen.add(item.id);
      scores.set(item.id, (scores.get(item.id) ?? 0) + weight / (k + rank + 1));
    });
  });

  return [...scores.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score);
}
