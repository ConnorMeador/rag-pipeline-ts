import type { SearchResult } from './store.js';

export interface RerankOptions {
  /** e.g. http://localhost:11434/v1 or https://api.openai.com/v1 */
  baseUrl: string;
  /** Not required for local endpoints. */
  apiKey?: string;
  model?: string;
  timeoutMs?: number;
}

/**
 * Re-order the cosine top-N candidates using an LLM's judgment of relevance.
 *
 * This is opt-in and off by default in the pipeline (see index.ts) — it adds
 * a network round trip and a second model call to every query, for a
 * relevance gain that often isn't worth it once the cosine ranking is
 * already decent.
 *
 * When it IS enabled, it fails open: any HTTP error, timeout, or response
 * that isn't a clean JSON array of indices falls back to the original
 * cosine order, unchanged. This is the important production lesson encoded
 * here — a reranker call is an optimization, not a dependency, and a broken
 * or slow reranker must degrade search quality, never break it.
 */
export async function rerank(
  query: string,
  candidates: SearchResult[],
  k: number,
  options: RerankOptions
): Promise<SearchResult[]> {
  if (candidates.length <= k) return candidates.slice(0, k);

  const fallback = () => candidates.slice(0, k);
  const timeoutMs = options.timeoutMs ?? 8_000;

  try {
    const numbered = candidates.map((c, i) => `[${i}] ${c.chunk.text.slice(0, 800)}`).join('\n\n');

    const prompt =
      `Query: ${query}\n\nCandidates:\n${numbered}\n\n` +
      `Return ONLY a JSON array of the candidate indices (the numbers in [brackets] above), ` +
      `ordered from most to least relevant to the query. Include all ${candidates.length} indices ` +
      `exactly once. Respond with the JSON array only — no prose, no markdown fences.`;

    const res = await fetch(`${options.baseUrl.replace(/\/+$/, '')}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}),
      },
      body: JSON.stringify({
        model: options.model ?? 'gpt-4o-mini',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0,
      }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return fallback();

    const json = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
    const content = json.choices?.[0]?.message?.content;
    if (typeof content !== 'string') return fallback();

    const match = content.match(/\[[\d,\s]+\]/);
    if (!match) return fallback();

    const order: unknown = JSON.parse(match[0]);
    if (!Array.isArray(order) || order.length === 0) return fallback();

    const seen = new Set<number>();
    const reordered: SearchResult[] = [];
    for (const raw of order) {
      const idx = typeof raw === 'number' ? raw : NaN;
      if (Number.isInteger(idx) && idx >= 0 && idx < candidates.length && !seen.has(idx)) {
        seen.add(idx);
        reordered.push(candidates[idx]!);
        if (reordered.length >= k) break;
      }
    }
    // The model returned fewer than k valid/unique indices (it dropped one,
    // repeated one, or hallucinated an out-of-range value). Top up from the
    // original cosine order rather than silently returning a short result.
    if (reordered.length < k) {
      for (let i = 0; i < candidates.length && reordered.length < k; i++) {
        if (!seen.has(i)) {
          seen.add(i);
          reordered.push(candidates[i]!);
        }
      }
    }
    return reordered;
  } catch {
    // Network error, timeout, or JSON.parse failure — fail open by design.
    return fallback();
  }
}
