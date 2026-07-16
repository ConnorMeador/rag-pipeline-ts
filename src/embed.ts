/**
 * Embedding backends.
 *
 * Two implementations of a common `Embedder` interface:
 *  - OllamaEmbedder: local, free, private (nomic-embed-text via Ollama's HTTP API)
 *  - OpenAICompatEmbedder: any OpenAI-compatible /v1/embeddings endpoint
 *
 * Neither implementation does anything at construction time — a network call
 * only happens when you call embed()/embedBatch() — so importing this module
 * never requires a running service or an API key. See the README for the
 * cost/privacy-vs-quality tradeoff between the two.
 */

export interface Embedder {
  /** Embed a single string, returning its vector. */
  embed(text: string): Promise<number[]>;
  /** Embed many strings in one call where the backend supports it. */
  embedBatch(texts: string[]): Promise<number[][]>;
}

export interface OllamaEmbedderOptions {
  /** Default: http://localhost:11434 */
  baseUrl?: string;
  /** Default: nomic-embed-text */
  model?: string;
  timeoutMs?: number;
}

/**
 * Local embeddings via Ollama's `/api/embed` endpoint. Zero marginal cost, no
 * API key, and the text never leaves the machine.
 */
export class OllamaEmbedder implements Embedder {
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly timeoutMs: number;

  constructor(options: OllamaEmbedderOptions = {}) {
    this.baseUrl = (options.baseUrl ?? 'http://localhost:11434').replace(/\/+$/, '');
    this.model = options.model ?? 'nomic-embed-text';
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  async embed(text: string): Promise<number[]> {
    const [vector] = await this.embedBatch([text]);
    if (!vector) throw new Error('ollama embed response contained no vectors');
    return vector;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const res = await fetch(`${this.baseUrl}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, input: texts }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`ollama embed failed (${res.status}): ${body.slice(0, 200)}`);
    }
    const json = (await res.json()) as { embeddings?: number[][] };
    if (!Array.isArray(json.embeddings)) {
      throw new Error('ollama embed response missing "embeddings"');
    }
    return json.embeddings;
  }
}

export interface OpenAICompatEmbedderOptions {
  /** e.g. https://api.openai.com/v1 or a self-hosted OpenAI-compatible server */
  baseUrl: string;
  /** Read this from an env var at call time — never hardcode it. */
  apiKey: string;
  /** Default: text-embedding-3-small */
  model?: string;
  timeoutMs?: number;
}

/** Any OpenAI-compatible `/v1/embeddings` endpoint. */
export class OpenAICompatEmbedder implements Embedder {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly model: string;
  private readonly timeoutMs: number;

  constructor(options: OpenAICompatEmbedderOptions) {
    if (!options.apiKey) {
      throw new Error('OpenAICompatEmbedder requires an apiKey');
    }
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
    this.apiKey = options.apiKey;
    this.model = options.model ?? 'text-embedding-3-small';
    this.timeoutMs = options.timeoutMs ?? 30_000;
  }

  async embed(text: string): Promise<number[]> {
    const [vector] = await this.embedBatch([text]);
    if (!vector) throw new Error('embeddings response contained no vectors');
    return vector;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const res = await fetch(`${this.baseUrl}/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ model: this.model, input: texts }),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`embeddings request failed (${res.status}): ${body.slice(0, 200)}`);
    }
    const json = (await res.json()) as { data?: Array<{ embedding: number[] }> };
    if (!Array.isArray(json.data)) {
      throw new Error('embeddings response missing "data"');
    }
    return json.data.map((d) => d.embedding);
  }
}
