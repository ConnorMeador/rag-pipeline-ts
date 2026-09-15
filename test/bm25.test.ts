import { describe, expect, it } from 'vitest';
import { Bm25Index, tokenize } from '../src/bm25.js';

describe('tokenize', () => {
  it('lowercases, splits on punctuation, and drops stopwords', () => {
    expect(tokenize('The BRCA1 gene, and its role in DNA-repair.')).toEqual(['brca1', 'gene', 'role', 'dna', 'repair']);
  });

  it('keeps non-ASCII letters intact', () => {
    expect(tokenize('Café naïve')).toEqual(['café', 'naïve']);
  });
});

describe('Bm25Index', () => {
  const docs = [
    { id: 'cats', text: 'Cats are small domesticated carnivores that purr.' },
    { id: 'dogs', text: 'Dogs are loyal domesticated animals that bark.' },
    { id: 'error', text: 'The deploy failed with error code E1234 after the build step.' },
    { id: 'long', text: `cats ${'filler words about nothing in particular '.repeat(40)}` },
  ];

  it('finds the document containing a rare exact token', () => {
    const index = new Bm25Index(docs);
    expect(index.search('what does E1234 mean', 1)[0]!.id).toBe('error');
  });

  it('ranks a short focused document above a long one with the same term', () => {
    const index = new Bm25Index(docs);
    const ids = index.search('cats', 2).map((r) => r.id);
    expect(ids).toEqual(['cats', 'long']);
  });

  it('never returns documents that share no query term', () => {
    const index = new Bm25Index(docs);
    expect(index.search('zebra', 10)).toEqual([]);
  });

  it('respects k and handles k <= 0 and an empty index', () => {
    const index = new Bm25Index(docs);
    expect(index.search('domesticated', 1)).toHaveLength(1);
    expect(index.search('domesticated', 0)).toEqual([]);
    expect(new Bm25Index([]).search('cats', 5)).toEqual([]);
  });

  it('sums evidence across query terms', () => {
    const index = new Bm25Index(docs);
    const [top] = index.search('domesticated purr', 1);
    expect(top!.id).toBe('cats');
  });
});
