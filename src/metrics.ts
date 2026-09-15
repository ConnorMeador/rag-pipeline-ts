/**
 * Standard IR metrics over a ranked list of ids and a set of relevance
 * judgments (qrels). Pure functions with no I/O, so the benchmark numbers in
 * the README come from code that is itself unit-tested.
 */

/** id -> graded relevance (> 0 means relevant). */
export type Judgments = ReadonlyMap<string, number>;

/** Fraction of all relevant ids that appear in the top k. */
export function recallAtK(ranked: readonly string[], relevant: Judgments, k: number): number {
  const totalRelevant = [...relevant.values()].filter((g) => g > 0).length;
  if (totalRelevant === 0) return 0;
  const hits = ranked.slice(0, k).filter((id) => (relevant.get(id) ?? 0) > 0).length;
  return hits / totalRelevant;
}

/** 1 / rank of the first relevant id within the top k, else 0. */
export function reciprocalRankAtK(ranked: readonly string[], relevant: Judgments, k: number): number {
  const index = ranked.slice(0, k).findIndex((id) => (relevant.get(id) ?? 0) > 0);
  return index === -1 ? 0 : 1 / (index + 1);
}

/** nDCG@k with graded gains, the headline metric BEIR reports. */
export function ndcgAtK(ranked: readonly string[], relevant: Judgments, k: number): number {
  const dcg = ranked
    .slice(0, k)
    .reduce((sum, id, i) => sum + (relevant.get(id) ?? 0) / Math.log2(i + 2), 0);
  const ideal = [...relevant.values()]
    .filter((g) => g > 0)
    .sort((a, b) => b - a)
    .slice(0, k)
    .reduce((sum, gain, i) => sum + gain / Math.log2(i + 2), 0);
  return ideal === 0 ? 0 : dcg / ideal;
}

export function mean(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length;
}
