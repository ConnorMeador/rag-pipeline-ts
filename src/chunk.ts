/**
 * Paragraph-aware text chunker.
 *
 * Splits a document into overlapping chunks sized for embedding models.
 * Blank-line boundaries (paragraphs) are the preferred split points; a
 * single paragraph larger than `maxChars` is hard-split on whitespace so no
 * chunk silently exceeds the target size.
 */

export interface ChunkOptions {
  /** Target maximum characters per chunk. Default: 1200 (roughly 300 tokens). */
  maxChars?: number;
  /** Characters of trailing context carried into the next chunk. Default: 150. */
  overlapChars?: number;
}

export interface TextChunk {
  /** 0-based position of this chunk among the document's chunks. */
  index: number;
  /** Chunk text, including any overlap carried over from the previous chunk. */
  text: string;
  /** Offset in the source document where this chunk's own (non-overlap) content starts. */
  charStart: number;
  /** Offset in the source document where this chunk's own (non-overlap) content ends. */
  charEnd: number;
}

const DEFAULT_MAX_CHARS = 1200;
const DEFAULT_OVERLAP_CHARS = 150;

interface Span {
  text: string;
  start: number;
  end: number;
}

/** Split on blank-line boundaries, tracking each paragraph's offset in the source doc. */
function splitParagraphs(doc: string): Span[] {
  const spans: Span[] = [];
  const parts = doc.split(/\n{2,}/);
  let cursor = 0;
  for (const part of parts) {
    const start = doc.indexOf(part, cursor);
    const trimmed = part.trim();
    if (trimmed.length > 0) {
      const leading = part.length - part.trimStart().length;
      const actualStart = start + leading;
      spans.push({ text: trimmed, start: actualStart, end: actualStart + trimmed.length });
    }
    cursor = start + part.length;
  }
  return spans;
}

/**
 * Hard-split a single oversized paragraph into pieces of at most `maxChars`,
 * preferring to break on a whitespace boundary rather than mid-word.
 */
function hardSplit(text: string, start: number, maxChars: number, overlapChars: number): Span[] {
  const pieces: Span[] = [];
  let pos = 0;
  while (pos < text.length) {
    let end = Math.min(pos + maxChars, text.length);
    if (end < text.length) {
      const lastSpace = text.lastIndexOf(' ', end);
      if (lastSpace > pos + maxChars * 0.5) {
        end = lastSpace;
      }
    }
    const slice = text.slice(pos, end).trim();
    if (slice.length > 0) {
      pieces.push({ text: slice, start: start + pos, end: start + end });
    }
    if (end >= text.length) break;
    // Always make forward progress even if overlapChars would otherwise stall us.
    pos = Math.max(end - overlapChars, pos + 1);
  }
  return pieces;
}

function tailOverlap(text: string, overlapChars: number): string {
  if (overlapChars === 0 || text.length === 0) return '';
  return text.slice(-overlapChars).trimStart();
}

/**
 * Chunk a document into overlapping, paragraph-aware pieces.
 *
 * Edge cases handled explicitly:
 *  - empty document -> []
 *  - whitespace-only document -> []
 *  - a single paragraph larger than `maxChars` (no blank lines at all) ->
 *    hard-split at whitespace boundaries instead of silently exceeding the
 *    target size or throwing.
 */
export function chunkText(doc: string, options: ChunkOptions = {}): TextChunk[] {
  const maxChars = options.maxChars ?? DEFAULT_MAX_CHARS;
  const overlapChars = options.overlapChars ?? DEFAULT_OVERLAP_CHARS;

  if (maxChars <= 0) {
    throw new Error('maxChars must be a positive number');
  }
  if (overlapChars < 0 || overlapChars >= maxChars) {
    throw new Error('overlapChars must be >= 0 and less than maxChars');
  }

  const rawParagraphs = splitParagraphs(doc);
  if (rawParagraphs.length === 0) return [];

  // Expand any paragraph that alone exceeds maxChars into hard-split pieces
  // so every span downstream is guaranteed to fit in a single chunk.
  const paragraphs: Span[] = [];
  for (const p of rawParagraphs) {
    if (p.text.length > maxChars) {
      paragraphs.push(...hardSplit(p.text, p.start, maxChars, overlapChars));
    } else {
      paragraphs.push(p);
    }
  }

  const chunks: TextChunk[] = [];
  let buffer: Span[] = [];
  let bufferLen = 0;

  const flush = () => {
    if (buffer.length === 0) return;
    const first = buffer[0]!;
    const last = buffer[buffer.length - 1]!;
    const coreText = buffer.map((p) => p.text).join('\n\n');
    const prevChunk = chunks[chunks.length - 1];
    const prevOverlap = prevChunk ? tailOverlap(prevChunk.text, overlapChars) : '';
    const text = prevOverlap ? `${prevOverlap}\n\n${coreText}` : coreText;

    chunks.push({ index: chunks.length, text, charStart: first.start, charEnd: last.end });
    buffer = [];
    bufferLen = 0;
  };

  for (const p of paragraphs) {
    const additionalLen = bufferLen === 0 ? p.text.length : bufferLen + 2 + p.text.length;
    if (additionalLen > maxChars && buffer.length > 0) {
      flush();
    }
    buffer.push(p);
    bufferLen = buffer.reduce((sum, para, i) => sum + para.text.length + (i > 0 ? 2 : 0), 0);
  }
  flush();

  return chunks;
}
