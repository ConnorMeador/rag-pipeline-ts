/**
 * Okapi BM25 keyword index.
 *
 * The lexical half of hybrid retrieval. Dense embeddings are good at
 * paraphrase ("heart attack" ~ "myocardial infarction") and bad at exact
 * rare tokens — gene names, error codes, part numbers, version strings —
 * which is precisely where BM25 is strongest. Everything lives in memory,
 * same as `VectorStore`: an inverted index of term -> (doc, tf) postings.
 */

export interface Bm25Options {
  /** Term-frequency saturation. Default: 1.2 (the usual Lucene/Anserini value). */
  k1?: number;
  /** Length normalization, 0 = none, 1 = full. Default: 0.75. */
  b?: number;
}

export interface Bm25Doc {
  id: string;
  text: string;
}

export interface Bm25Result {
  id: string;
  score: number;
}

// A deliberately short stopword list: only words so common they carry no
// ranking signal. Aggressive lists hurt recall on technical text.
const STOPWORDS = new Set(
  'a an and are as at be but by for from has have in into is it its of on or that the their this to was were which with'.split(
    ' '
  )
);

/** Lowercase, split on anything that isn't a letter or digit, drop stopwords. */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length > 0 && !STOPWORDS.has(t));
}

interface Posting {
  doc: number;
  tf: number;
}

export class Bm25Index {
  private readonly k1: number;
  private readonly b: number;
  private readonly ids: string[] = [];
  private readonly docLengths: number[] = [];
  private readonly postings = new Map<string, Posting[]>();
  private totalLength = 0;

  constructor(docs: readonly Bm25Doc[], options: Bm25Options = {}) {
    this.k1 = options.k1 ?? 1.2;
    this.b = options.b ?? 0.75;

    for (const doc of docs) {
      const docIndex = this.ids.length;
      const tokens = tokenize(doc.text);
      this.ids.push(doc.id);
      this.docLengths.push(tokens.length);
      this.totalLength += tokens.length;

      const counts = new Map<string, number>();
      for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1);
      for (const [term, tf] of counts) {
        let list = this.postings.get(term);
        if (!list) {
          list = [];
          this.postings.set(term, list);
        }
        list.push({ doc: docIndex, tf });
      }
    }
  }

  get size(): number {
    return this.ids.length;
  }

  /** Top-k documents by BM25 score. Documents sharing no query term are never returned. */
  search(query: string, k: number): Bm25Result[] {
    if (k <= 0 || this.ids.length === 0) return [];

    const n = this.ids.length;
    const avgLength = this.totalLength / n || 1;
    const scores = new Map<number, number>();

    // Each distinct query term counts once; repeating a word in the query
    // shouldn't let it dominate the ranking.
    for (const term of new Set(tokenize(query))) {
      const list = this.postings.get(term);
      if (!list) continue;
      // BM25+ style IDF floor: log(1 + ...) is never negative, so a term that
      // appears in most documents contributes little instead of subtracting.
      const idf = Math.log(1 + (n - list.length + 0.5) / (list.length + 0.5));
      for (const { doc, tf } of list) {
        const norm = this.k1 * (1 - this.b + (this.b * this.docLengths[doc]!) / avgLength);
        const termScore = (idf * (tf * (this.k1 + 1))) / (tf + norm);
        scores.set(doc, (scores.get(doc) ?? 0) + termScore);
      }
    }

    return [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, k)
      .map(([doc, score]) => ({ id: this.ids[doc]!, score }));
  }
}
