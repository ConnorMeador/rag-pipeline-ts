#!/usr/bin/env node
/**
 * Walkthrough CLI: ingest a directory of docs, then answer one question
 * against it. This is a readable demonstration of the ingest() -> ask()
 * flow with sensible defaults — not meant to be a production entry point.
 *
 * Usage:
 *   pnpm ask "What does this project do?" ./docs
 *
 * Env:
 *   OLLAMA_BASE_URL   default http://localhost:11434 (embeddings)
 *   RERANK            set to "1" to enable the opt-in reranker
 *   RERANK_BASE_URL   OpenAI-compatible chat/completions base, e.g. an Ollama /v1 endpoint
 *   RERANK_API_KEY    only needed for hosted rerank endpoints
 *   RERANK_MODEL      default gpt-4o-mini
 */

import { OllamaEmbedder } from './embed.js';
import { ask, ingest } from './index.js';

async function main() {
  const [question, dir] = process.argv.slice(2);
  if (!question || !dir) {
    console.error('Usage: pnpm ask "<question>" <docs-dir>');
    process.exitCode = 1;
    return;
  }

  const embedder = new OllamaEmbedder({
    baseUrl: process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434',
  });

  console.error(`Ingesting ${dir} ...`);
  const store = await ingest(dir, { embedder });
  console.error(`Indexed ${store.size} chunk(s).`);

  const useRerank = process.env.RERANK === '1';
  const result = await ask(question, {
    embedder,
    store,
    rerank: useRerank,
    rerankOptions: useRerank
      ? {
          baseUrl: process.env.RERANK_BASE_URL ?? 'http://localhost:11434/v1',
          apiKey: process.env.RERANK_API_KEY,
          model: process.env.RERANK_MODEL ?? 'gpt-4o-mini',
        }
      : undefined,
  });

  console.log(`\nQ: ${result.query}\n`);
  result.results.forEach((r, i) => {
    const source = (r.chunk.metadata as { source?: string } | undefined)?.source ?? r.chunk.id;
    console.log(`${i + 1}. [${r.score.toFixed(3)}] ${source}`);
    console.log(`   ${r.chunk.text.slice(0, 200).replace(/\s+/g, ' ')}...\n`);
  });
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exitCode = 1;
});
