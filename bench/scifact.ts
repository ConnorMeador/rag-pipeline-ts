#!/usr/bin/env node
/**
 * Retrieval benchmark on BEIR SciFact (5,183 abstracts, 300 test claims with
 * human relevance judgments), run through this repo's own chunker, embedder,
 * VectorStore and BM25 index — not a separate evaluation stack.
 *
 * Usage:
 *   pnpm bench                                  # nomic-embed-text, all variants
 *   EMBED_MODELS=nomic-embed-text,mxbai-embed-large pnpm bench
 *
 * Needs a local Ollama with the listed embedding models pulled. The dataset
 * (~8 MB, from the MTEB mirror on Hugging Face) and embeddings are cached in
 * bench/.data/, which is gitignored: nothing from the dataset is committed.
 * Results land in bench/results/.
 */

import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { Bm25Index } from '../src/bm25.js';
import { chunkText } from '../src/chunk.js';
import { OllamaEmbedder } from '../src/embed.js';
import { reciprocalRankFusion } from '../src/hybrid.js';
import { mean, ndcgAtK, recallAtK, reciprocalRankAtK } from '../src/metrics.js';
import type { Judgments } from '../src/metrics.js';
import { VectorStore } from '../src/store.js';
import type { StoredChunk } from '../src/store.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = join(HERE, '.data', 'scifact');
const RESULTS_DIR = join(HERE, 'results');
const DATASET_BASE = 'https://huggingface.co/datasets/mteb/scifact/resolve/main';

/** Chunks retrieved per query before collapsing to documents. */
const CHUNK_POOL = 300;
const EMBED_BATCH = 64;

/**
 * Task prefixes the model authors document. nomic-embed-text in particular
 * is trained with them, so "no prefix" is a realistic misconfiguration worth
 * measuring rather than assuming away.
 */
const PREFIXES: Record<string, { query: string; document: string }> = {
  'nomic-embed-text': { query: 'search_query: ', document: 'search_document: ' },
  'mxbai-embed-large': { query: 'Represent this sentence for searching relevant passages: ', document: '' },
};

interface Row {
  _id: string;
  title?: string;
  text: string;
}

async function fetchCached(name: string): Promise<string> {
  const path = join(DATA_DIR, name);
  if (existsSync(path)) return readFile(path, 'utf8');
  const res = await fetch(`${DATASET_BASE}/${name}`);
  if (!res.ok) throw new Error(`download ${name} failed (${res.status})`);
  const body = await res.text();
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, body, 'utf8');
  return body;
}

function parseJsonl(raw: string): Row[] {
  return raw
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as Row);
}

function parseQrels(raw: string): Map<string, Map<string, number>> {
  const qrels = new Map<string, Map<string, number>>();
  for (const line of raw.split(/\r?\n/).slice(1)) {
    const [queryId, docId, score] = line.split('\t');
    if (!queryId || !docId || score === undefined) continue;
    const judged = qrels.get(queryId) ?? new Map<string, number>();
    judged.set(docId, Number(score));
    qrels.set(queryId, judged);
  }
  return qrels;
}

/** Embeddings cached as raw Float32 so a re-run doesn't re-embed 8k chunks. */
async function embedCached(
  embedder: OllamaEmbedder,
  cacheName: string,
  texts: string[]
): Promise<{ vectors: number[][]; seconds: number; cached: boolean }> {
  const path = join(DATA_DIR, cacheName);
  if (existsSync(path)) {
    const buf = await readFile(path);
    const dims = buf.readUInt32LE(0);
    const floats = new Float32Array(buf.buffer, buf.byteOffset + 4, (buf.byteLength - 4) / 4);
    const vectors: number[][] = [];
    for (let i = 0; i < floats.length; i += dims) vectors.push(Array.from(floats.subarray(i, i + dims)));
    if (vectors.length === texts.length) return { vectors, seconds: 0, cached: true };
  }

  const started = performance.now();
  const vectors: number[][] = [];
  for (let i = 0; i < texts.length; i += EMBED_BATCH) {
    vectors.push(...(await embedder.embedBatch(texts.slice(i, i + EMBED_BATCH))));
    process.stderr.write(`\r  embedded ${Math.min(i + EMBED_BATCH, texts.length)}/${texts.length}`);
  }
  process.stderr.write('\n');
  const seconds = (performance.now() - started) / 1000;

  const dims = vectors[0]?.length ?? 0;
  const buf = Buffer.alloc(4 + vectors.length * dims * 4);
  buf.writeUInt32LE(dims, 0);
  vectors.forEach((v, row) => v.forEach((x, col) => buf.writeFloatLE(x, 4 + (row * dims + col) * 4)));
  await mkdir(DATA_DIR, { recursive: true });
  await writeFile(path, buf);
  return { vectors, seconds, cached: false };
}

/** Collapse a chunk ranking into a document ranking (a document ranks at its best chunk). */
function toDocRanking(chunkIds: readonly string[]): string[] {
  const docs: string[] = [];
  const seen = new Set<string>();
  for (const chunkId of chunkIds) {
    const docId = chunkId.slice(0, chunkId.lastIndexOf('#'));
    if (!seen.has(docId)) {
      seen.add(docId);
      docs.push(docId);
    }
  }
  return docs;
}

interface Scores {
  method: string;
  ndcg10: number;
  mrr10: number;
  recall10: number;
  recall100: number;
}

function score(method: string, rankings: Map<string, string[]>, qrels: Map<string, Judgments>): Scores {
  const ids = [...qrels.keys()];
  const per = (fn: (ranked: string[], judged: Judgments) => number) =>
    mean(ids.map((id) => fn(rankings.get(id) ?? [], qrels.get(id)!)));
  return {
    method,
    ndcg10: per((r, j) => ndcgAtK(r, j, 10)),
    mrr10: per((r, j) => reciprocalRankAtK(r, j, 10)),
    recall10: per((r, j) => recallAtK(r, j, 10)),
    recall100: per((r, j) => recallAtK(r, j, 100)),
  };
}

async function main() {
  const models = (process.env.EMBED_MODELS ?? 'nomic-embed-text').split(',').map((m) => m.trim());
  const ollamaUrl = process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434';

  console.error('Loading SciFact (cached after first run) ...');
  const corpus = parseJsonl(await fetchCached('corpus.jsonl'));
  const allQueries = parseJsonl(await fetchCached('queries.jsonl'));
  const qrels = parseQrels(await fetchCached('qrels/test.tsv'));
  const queries = allQueries.filter((q) => qrels.has(q._id));

  const chunks: StoredChunk[] = [];
  for (const doc of corpus) {
    const body = doc.title ? `${doc.title}\n\n${doc.text}` : doc.text;
    for (const piece of chunkText(body)) {
      chunks.push({ id: `${doc._id}#${piece.index}`, text: piece.text, embedding: [] });
    }
  }
  console.error(`${corpus.length} documents -> ${chunks.length} chunks; ${queries.length} test queries.`);

  const results: Scores[] = [];
  const timings: Record<string, unknown> = {};

  const bm25Started = performance.now();
  const bm25 = new Bm25Index(chunks);
  timings.bm25IndexMs = Math.round(performance.now() - bm25Started);
  const bm25Chunks = new Map(queries.map((q) => [q._id, bm25.search(q.text, CHUNK_POOL).map((h) => h.id)]));
  results.push(score('BM25', mapValues(bm25Chunks, toDocRanking), qrels));

  for (const model of models) {
    const embedder = new OllamaEmbedder({ baseUrl: ollamaUrl, model, timeoutMs: 120_000 });
    const variants = PREFIXES[model]
      ? [
          { label: `${model} (no task prefix)`, key: 'raw', prefix: { query: '', document: '' } },
          { label: `${model} (task prefixes)`, key: 'prefixed', prefix: PREFIXES[model]! },
        ]
      : [{ label: model, key: 'raw', prefix: { query: '', document: '' } }];

    for (const variant of variants) {
      console.error(`Embedding with ${variant.label} ...`);
      const cacheSlug = `${model.replace(/[^a-z0-9.-]+/gi, '_')}-${variant.key}`;
      const docEmb = await embedCached(
        embedder,
        `emb-docs-${cacheSlug}.f32`,
        chunks.map((c) => variant.prefix.document + c.text)
      );
      const queryEmb = await embedCached(
        embedder,
        `emb-queries-${cacheSlug}.f32`,
        queries.map((q) => variant.prefix.query + q.text)
      );
      const store = VectorStore.fromChunks(chunks.map((c, i) => ({ ...c, embedding: docEmb.vectors[i]! })));

      const searchStarted = performance.now();
      const denseChunks = new Map(
        queries.map((q, i) => [q._id, store.search(queryEmb.vectors[i]!, CHUNK_POOL).map((r) => r.chunk.id)])
      );
      timings[`${variant.label} search ms/query`] = +((performance.now() - searchStarted) / queries.length).toFixed(2);
      if (!docEmb.cached) timings[`${variant.label} corpus embed s`] = +docEmb.seconds.toFixed(1);

      results.push(score(`Dense: ${variant.label}`, mapValues(denseChunks, toDocRanking), qrels));

      const fused = new Map(
        queries.map((q) => [
          q._id,
          reciprocalRankFusion([
            (denseChunks.get(q._id) ?? []).map((id) => ({ id })),
            (bm25Chunks.get(q._id) ?? []).map((id) => ({ id })),
          ]).map((f) => f.id),
        ])
      );
      results.push(score(`Hybrid RRF: BM25 + ${variant.label}`, mapValues(fused, toDocRanking), qrels));
    }
  }

  const table = [
    '| Method | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |',
    '|---|---:|---:|---:|---:|',
    ...results.map(
      (r) =>
        `| ${r.method} | ${r.ndcg10.toFixed(3)} | ${r.mrr10.toFixed(3)} | ${r.recall10.toFixed(3)} | ${r.recall100.toFixed(3)} |`
    ),
  ].join('\n');

  console.log(`\nBEIR SciFact test split — ${queries.length} queries, ${corpus.length} docs, ${chunks.length} chunks\n`);
  console.log(table);
  console.log('\nTimings:', JSON.stringify(timings, null, 2));

  await mkdir(RESULTS_DIR, { recursive: true });
  const record = {
    dataset: 'BEIR SciFact (test)',
    source: `${DATASET_BASE}`,
    queries: queries.length,
    documents: corpus.length,
    chunks: chunks.length,
    chunking: 'chunkText defaults (maxChars 1200, overlap 150); document score = best chunk',
    ranAt: new Date().toISOString(),
    node: process.version,
    results,
    timings,
  };
  await writeFile(join(RESULTS_DIR, 'scifact.json'), `${JSON.stringify(record, null, 2)}\n`, 'utf8');
  await writeFile(join(RESULTS_DIR, 'scifact.md'), `${table}\n`, 'utf8');
}

function mapValues<K, V, W>(map: Map<K, V>, fn: (value: V) => W): Map<K, W> {
  return new Map([...map.entries()].map(([k, v]) => [k, fn(v)]));
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exitCode = 1;
});
