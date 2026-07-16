import { afterEach, describe, expect, it, vi } from 'vitest';
import { rerank } from '../src/rerank.js';
import type { SearchResult } from '../src/store.js';

function candidate(id: string, score: number): SearchResult {
  return { chunk: { id, text: `content for ${id}`, embedding: [] }, score };
}

const candidates: SearchResult[] = [
  candidate('a', 0.9),
  candidate('b', 0.8),
  candidate('c', 0.7),
  candidate('d', 0.6),
  candidate('e', 0.5),
  candidate('f', 0.4),
];

const baseOptions = { baseUrl: 'http://localhost:9999/v1', model: 'test-model' };

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('rerank', () => {
  it('skips the network call entirely when there are not enough candidates to rerank', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const result = await rerank('q', candidates.slice(0, 3), 5, baseOptions);
    expect(result).toEqual(candidates.slice(0, 3));
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('reorders candidates per the model reply on the happy path', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ choices: [{ message: { content: '[3, 0, 1]' } }] }),
      })
    );
    const result = await rerank('q', candidates, 3, baseOptions);
    expect(result.map((r) => r.chunk.id)).toEqual(['d', 'a', 'b']);
  });

  it('tops up from cosine order when the model returns fewer than k valid indices', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ choices: [{ message: { content: '[2]' } }] }),
      })
    );
    const result = await rerank('q', candidates, 3, baseOptions);
    expect(result.map((r) => r.chunk.id)).toEqual(['c', 'a', 'b']);
  });

  it('fails open to cosine order on a non-OK HTTP response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }));
    const result = await rerank('q', candidates, 3, baseOptions);
    expect(result).toEqual(candidates.slice(0, 3));
  });

  it('fails open to cosine order on a network error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('ECONNREFUSED')));
    const result = await rerank('q', candidates, 3, baseOptions);
    expect(result).toEqual(candidates.slice(0, 3));
  });

  it('fails open to cosine order on a timeout', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(Object.assign(new Error('The operation was aborted'), { name: 'TimeoutError' }))
    );
    const result = await rerank('q', candidates, 3, baseOptions);
    expect(result).toEqual(candidates.slice(0, 3));
  });

  it('fails open to cosine order on a garbled, non-JSON model reply', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ choices: [{ message: { content: 'Sure! Here you go: not actually json' } }] }),
      })
    );
    const result = await rerank('q', candidates, 3, baseOptions);
    expect(result).toEqual(candidates.slice(0, 3));
  });

  it('fails open to cosine order when the response body is missing expected fields', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: async () => ({ unexpected: 'shape' }) })
    );
    const result = await rerank('q', candidates, 3, baseOptions);
    expect(result).toEqual(candidates.slice(0, 3));
  });
});
