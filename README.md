# rag-pipeline-ts

## What this is

A small TypeScript retrieval-augmented generation (RAG) pipeline, extracted
and adapted from a production system that powers chat assistants on 10 live
sites. The original serves real user questions against per-site knowledge
bases; this repo strips out the site-specific wiring and keeps the parts that
generalize — chunking, embeddings, a vector store, and an opt-in reranker —
as a standalone, readable reference implementation.

It's small on purpose. Every file is short enough to read end to end, and the
test suite runs fully offline (no services, no API keys) so you can clone it
and verify the claims below yourself in under a minute.

## Architecture

```
                    ┌─────────────┐
   docs/*.md  ───▶  │  chunk.ts   │  paragraph-aware, overlapping windows
                    └──────┬──────┘
                           │ text chunks
                           ▼
                    ┌─────────────┐
                    │  embed.ts   │  Ollama (local) or any OpenAI-compatible API
                    └──────┬──────┘
                           │ vectors
                           ▼
                    ┌─────────────┐
                    │  store.ts   │  JSON file + brute-force cosine search
                    └──────┬──────┘
                           │ top-N candidates (cosine order)
                           ▼
                    ┌─────────────┐
   question   ───▶  │ rerank.ts   │  OPT-IN — LLM re-orders top-N, fails
                    │ (optional)  │  open to cosine order on any error
                    └──────┬──────┘
                           │ top-k results
                           ▼
                       index.ts
                    ingest() / ask()
                           │
                           ▼
                        cli.ts
                  pnpm ask "<question>" <dir>
```

`ingest(dir)` reads a folder of `.md`/`.mdx`/`.txt` files, chunks each one,
embeds every chunk, and returns an in-memory `VectorStore`. `ask(question)`
embeds the question, does a cosine search over the store, and — only if you
ask for it — runs a second LLM pass to re-rank the candidates.

## Three decisions and their tradeoffs

### 1. Local Ollama embeddings by default, hosted as an opt-in

`OllamaEmbedder` talks to a local `nomic-embed-text` model over Ollama's HTTP
API. It's free per call, keeps the source text on the machine, and has no
rate limit to plan around. The tradeoff is quality and portability: a local
768-dim model won't match a frontier hosted embedding model on some
retrieval benchmarks, and it requires Ollama running wherever the pipeline
runs, which is a real deployment constraint. `OpenAICompatEmbedder` exists
for exactly the cases where that tradeoff flips — a shared multi-tenant
service where per-call cost is negligible next to the ops cost of running a
local model everywhere. Same `Embedder` interface either way, so a caller
picks the backend without touching `chunk.ts`, `store.ts`, or `rerank.ts`.

### 2. Reranking is opt-in, and it fails open

`rerank.ts` takes the cosine top-N and asks an LLM to re-order them. It's off
by default in `ask()`. Turning it on adds a full model round trip to every
query — in practice that's the difference between a search that feels
instant and one with a noticeable pause, for a relevance gain that's often
marginal once the embedding model is already decent. It's worth it when
result ordering actually matters for what happens next (e.g. only the first
result gets shown, or gets fed into a prompt with no room for near-misses);
it's not worth it for a "show me the top 5" browsing experience.

The more important property is what happens when the reranker breaks: it
fails open, silently falling back to the original cosine order. Every
failure mode is covered — non-200 response, timeout, a reply that isn't
valid JSON, a reply that's valid JSON but not an array of indices, an array
that's missing some indices. This exists because of a real incident: a
reranker's malformed reply was once allowed to propagate as a broken result
set instead of degrading gracefully. A second-stage relevance pass is an
optimization; it should never be a single point of failure for search itself.

### 3. A JSON file, not a vector database

`store.ts` is deliberately unglamorous: load the array into memory, compute
cosine similarity against every row, sort, slice. No index structure, no
approximate nearest neighbor, no separate service to run or pay for. This is
the right size for anything under roughly 10k chunks — a full linear scan
over that many 768-dim vectors is single-digit milliseconds in Node, and the
whole store fits comfortably in memory. Past that point (or once you need
metadata filtering at scale, multi-tenant isolation, or horizontal scaling)
a real vector database earns its keep. Below it, a vector DB is mostly
operational overhead: another service to deploy, monitor, and pay for, in
exchange for performance headroom you're not using yet.

## Usage

```bash
pnpm install
pnpm build   # type-check + compile src/ to dist/
pnpm test    # runs fully offline, no services required

# Walkthrough: ingest a folder of docs, then ask one question against it.
# Requires a local Ollama with `nomic-embed-text` pulled.
pnpm ask "What does this project do?" ./docs
```

Programmatic use:

```ts
import { OllamaEmbedder, ingest, ask } from './src/index.js';

const embedder = new OllamaEmbedder(); // http://localhost:11434, nomic-embed-text
const store = await ingest('./docs', { embedder });

const { results } = await ask('How does chunking handle overlap?', {
  embedder,
  store,
  k: 5,
});
```

Enabling the reranker:

```ts
const { results } = await ask('...', {
  embedder,
  store,
  rerank: true,
  rerankOptions: {
    baseUrl: 'http://localhost:11434/v1', // or any OpenAI-compatible /v1
    model: 'qwen2.5-coder:7b',
  },
});
```

### Environment variables (CLI only)

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Embedding backend |
| `RERANK` | unset | Set to `1` to enable reranking |
| `RERANK_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible chat endpoint |
| `RERANK_API_KEY` | unset | Only needed for hosted rerank endpoints |
| `RERANK_MODEL` | `gpt-4o-mini` | Rerank model name |

## Limitations

- **No persistence layer wired into the CLI.** `VectorStore.save()`/`.load()`
  exist, but `cli.ts` re-ingests from scratch on every run for simplicity.
  A real deployment would build the index once and load it on startup.
- **Brute-force search only.** Fine below ~10k chunks (see above); it will
  not scale to a large corpus without swapping in an ANN index or a vector
  database.
- **No incremental re-indexing.** Adding one new document means either
  re-embedding everything or writing your own append-and-merge logic on top
  of `VectorStore`.
- **No streaming.** `ask()` returns a complete result set; there's no
  token-by-token generation step here at all — this pipeline stops at
  retrieval, not generation.
- **The reranker prompt is a single, unstructured instruction.** It works,
  but a production system with real relevance requirements would want a
  more carefully evaluated prompt (or a dedicated cross-encoder model)
  rather than a general-purpose chat model asked to return JSON.

## What I'd do differently

- Add a small CLI subcommand to build and persist an index once
  (`rag-pipeline build ./docs ./index.json`), separate from asking questions
  against it — the current CLI conflates the two for the sake of a
  one-line demo.
- Swap the reranker's regex-based JSON extraction for a provider that
  supports structured output / JSON mode natively, so "fails open on
  malformed JSON" becomes a much rarer path instead of the default
  assumption.
- Add a lightweight retrieval eval (a fixed set of query → expected-source
  pairs) so changes to chunking or embedding parameters have a
  regression check beyond "the unit tests still pass."
