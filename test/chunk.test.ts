import { describe, expect, it } from 'vitest';
import { chunkText } from '../src/chunk.js';

describe('chunkText', () => {
  it('returns no chunks for an empty document', () => {
    expect(chunkText('')).toEqual([]);
  });

  it('returns no chunks for a whitespace-only document', () => {
    expect(chunkText('   \n\n\t  \n  ')).toEqual([]);
  });

  it('keeps a short document as a single chunk', () => {
    const doc = 'This is a short paragraph.\n\nAnd a second one.';
    const chunks = chunkText(doc, { maxChars: 1000, overlapChars: 50 });
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.text).toContain('short paragraph');
    expect(chunks[0]!.text).toContain('second one');
  });

  it('splits a single huge paragraph (no blank lines) into multiple chunks', () => {
    const sentence = 'The quick brown fox jumps over the lazy dog. ';
    const doc = sentence.repeat(200); // ~9,400 chars, zero paragraph breaks
    const chunks = chunkText(doc, { maxChars: 500, overlapChars: 50 });
    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) {
      // core text is capped at maxChars; a little slack for the carried overlap + separator
      expect(chunk.text.length).toBeLessThanOrEqual(500 + 60);
    }
  });

  it('produces monotonically increasing chunk indices and offsets', () => {
    const doc = Array.from({ length: 10 }, (_, i) => `Paragraph number ${i}. `.repeat(20)).join('\n\n');
    const chunks = chunkText(doc, { maxChars: 400, overlapChars: 40 });
    expect(chunks.length).toBeGreaterThan(1);
    chunks.forEach((c, i) => expect(c.index).toBe(i));
    for (let i = 1; i < chunks.length; i++) {
      expect(chunks[i]!.charStart).toBeGreaterThanOrEqual(chunks[i - 1]!.charStart);
    }
  });

  it('carries overlap text from the previous chunk into the next one', () => {
    const doc = Array.from({ length: 6 }, (_, i) => `Section ${i}: ${'x'.repeat(150)}`).join('\n\n');
    const chunks = chunkText(doc, { maxChars: 200, overlapChars: 60 });
    expect(chunks.length).toBeGreaterThan(1);
    for (let i = 1; i < chunks.length; i++) {
      const prevTail = chunks[i - 1]!.text.slice(-60).trimStart();
      expect(chunks[i]!.text.startsWith(prevTail.slice(0, 20))).toBe(true);
    }
  });

  it('rejects invalid options', () => {
    expect(() => chunkText('hello world', { maxChars: 0 })).toThrow();
    expect(() => chunkText('hello world', { maxChars: 100, overlapChars: 100 })).toThrow();
    expect(() => chunkText('hello world', { maxChars: 100, overlapChars: -1 })).toThrow();
  });
});
