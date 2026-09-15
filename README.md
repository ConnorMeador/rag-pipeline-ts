# rag-pipeline-ts

## What this is

A small TypeScript retrieval-augmented generation (RAG) pipeline, extracted
and adapted from a production system that powers chat assistants on 10 live
sites. The original serves real user questions against per-site knowledge
bases; this repo strips out the site-specific wiring and keeps the parts that
generalize — chunking, embeddings, a vector store, BM25 hybrid retrieval, and
an opt-in reranker — as a standalone, readable reference implementation.

It's small on purpose. Every file is short enough to read end to end, and the
test suite runs fully offline (no services, no API keys) so you can clone it
and verify the claims below yourself in under a minute.

Retrieval quality is measured, not asserted: `pnpm bench` runs the pipeline
against the BEIR SciFact benchmark and writes the numbers in
[Measured retrieval quality](#measured-retrieval-quality) below.

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
                    │  bm25.ts    │  + in-memory BM25 keyword index (lazy)
                    │  hybrid.ts  │  + reciprocal rank fusion (opt-in)
                    └──────┬──────┘
                           │ top-N candidates
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
embeds the question, does a cosine search over the store (or, with
`retrieval: 'hybrid'`, a cosine search and a BM25 keyword search merged by
reciprocal rank fusion), and — only if you ask for it — runs a second LLM pass
to re-rank the candidates.

## Measured retrieval quality

[BEIR SciFact](https://github.com/beir-cellar/beir) test split: 300 scientific
claims searched against 5,183 paper abstracts, scored against human relevance
judgments. The corpus goes through this repo's own `chunkText` (defaults: 1,200
chars, 150 overlap → 12,092 chunks), `OllamaEmbedder`, `VectorStore` and
`Bm25Index`; a document ranks at its best chunk. Run it yourself with
`pnpm bench`; raw output is in [`bench/results/scifact.json`](bench/results/scifact.json).

| Method | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| BM25 only | 0.648 | 0.617 | 0.764 | 0.880 |
| Dense: nomic-embed-text (no task prefix) | 0.687 | 0.650 | 0.824 | 0.950 |
| Dense: nomic-embed-text (task prefixes) | 0.696 | 0.657 | 0.839 | 0.940 |
| **Hybrid RRF: BM25 + nomic-embed-text (no task prefix)** | **0.712** | **0.678** | **0.846** | **0.960** |
| Hybrid RRF: BM25 + nomic-embed-text (task prefixes) | 0.706 | 0.669 | 0.845 | 0.955 |
| Dense: mxbai-embed-large (no task prefix) | 0.726 | 0.690 | 0.860 | 0.968 |
| Dense: mxbai-embed-large (task prefix) | 0.730 | 0.692 | 0.872 | 0.975 |
| Hybrid RRF: BM25 + mxbai-embed-large (no task prefix) | 0.731 | 0.698 | 0.861 | 0.980 |
| **Hybrid RRF: BM25 + mxbai-embed-large (task prefix)** | **0.737** | **0.705** | **0.864** | **0.977** |

What the numbers say:

- **The harness is calibrated.** The BM25 row lands at 0.648 nDCG@10 against
  the 0.665 the BEIR paper reports for BM25 on SciFact. The small gap is
  expected: this is a simple tokenizer with no stemming, scoring chunks rather
  than whole abstracts.
- **Hybrid retrieval pays off most on a weaker embedder.** Fusing BM25 into
  the default `nomic-embed-text` setup lifts nDCG@10 by +0.025 (0.687 → 0.712)
  and cuts the misses at Recall@100 from 5.0% to 4.0%. On the stronger
  `mxbai-embed-large` the lift shrinks to +0.005–0.007. My read, not a
  measured breakdown: keyword search recovers exact-term matches (gene and
  drug names) that the weaker model misses and the stronger one already finds.
- **Model choice beats prompt details.** Switching embedders moved nDCG@10
  more (+0.039) than adding the documented task prefixes (+0.004 to +0.009).
  With nomic, prefixes helped dense-only search but slightly *hurt* the hybrid
  (0.712 → 0.706), so "use the documented prefixes" isn't automatically right
  once another retriever is in the mix. That's why it's measured.
- **Search cost is small at this size.** A brute-force cosine query over
  12,092 chunks took 10–19 ms in Node on a Ryzen 9 9950X; building the BM25
  index took 289 ms. Embedding the corpus took 1.5–2.5 min on an RTX 5070 Ti
  that other jobs were also using, so treat that as a rough figure.

Not measured here: the LLM reranker. Its relevance gain on this benchmark is
unknown, so the tradeoff discussion below is judgment, not a number.

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
instant and one with a noticeable pause, for a relevance gain I haven't
benchmarked yet (see above). It's worth it when
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
the right size for corpora in the low tens of thousands of chunks — the
benchmark above scans 12,092 vectors in 10–19 ms per query in Node, and the
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
RETRIEVAL=hybrid pnpm ask "What does this project do?" ./docs

# Benchmark on BEIR SciFact (downloads ~8 MB once, caches embeddings).
# Requires the listed embedding models pulled in Ollama.
EMBED_MODELS=nomic-embed-text,mxbai-embed-large pnpm bench
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
  retrieval: 'hybrid', // optional: cosine + BM25 merged with reciprocal rank fusion
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
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `RETRIEVAL` | `dense` | Set to `hybrid` for cosine + BM25 fusion |
| `RERANK` | unset | Set to `1` to enable reranking |
| `RERANK_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible chat endpoint |
| `RERANK_API_KEY` | unset | Only needed for hosted rerank endpoints |
| `RERANK_MODEL` | `gpt-4o-mini` | Rerank model name |

## Limitations

- **No persistence layer wired into the CLI.** `VectorStore.save()`/`.load()`
  exist, but `cli.ts` re-ingests from scratch on every run for simplicity.
  A real deployment would build the index once and load it on startup.
- **Brute-force search only.** Fine at the benchmark's ~12k chunks (see
  above); it will not scale to a large corpus without swapping in an ANN
  index or a vector database.
- **A deliberately simple BM25 tokenizer.** Lowercase, split on
  non-alphanumerics, a short stopword list, no stemming. It's why BM25 here
  scores a little under the published baseline.
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
- Put the reranker through the same benchmark, so its latency cost can be
  weighed against a measured nDCG change instead of intuition.
- Sweep chunk size and overlap on the benchmark. The 1,200/150 defaults
  came from the production system and haven't been tuned against numbers yet.
