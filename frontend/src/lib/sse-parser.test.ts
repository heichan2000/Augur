import { describe, expect, it } from "vitest";

import type { SSEEvent } from "./sse";
import { SSEFrameParser, parseSSEStream } from "./sse-parser";

/** Encode strings into a byte stream, chunked exactly as given. */
function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>): Promise<SSEEvent[]> {
  const events: SSEEvent[] = [];
  for await (const event of parseSSEStream(stream)) events.push(event);
  return events;
}

describe("SSEFrameParser", () => {
  it("parses a single complete frame", () => {
    const parser = new SSEFrameParser();

    expect(parser.push('event: token\ndata: {"text":"Hello"}\n\n')).toEqual([
      { type: "token", data: { text: "Hello" } },
    ]);
  });

  it("parses several frames arriving in one chunk", () => {
    const parser = new SSEFrameParser();

    const events = parser.push(
      'event: token\ndata: {"text":"a"}\n\nevent: token\ndata: {"text":"b"}\n\nevent: done\ndata: {}\n\n',
    );

    expect(events).toEqual([
      { type: "token", data: { text: "a" } },
      { type: "token", data: { text: "b" } },
      { type: "done", data: {} },
    ]);
  });

  it("buffers a frame split across chunk boundaries", () => {
    const parser = new SSEFrameParser();

    expect(parser.push("event: tok")).toEqual([]);
    expect(parser.push('en\ndata: {"text":"Hel')).toEqual([]);
    expect(parser.push('lo"}')).toEqual([]);
    expect(parser.push("\n\n")).toEqual([{ type: "token", data: { text: "Hello" } }]);
  });

  it("preserves JSON-escaped newlines and quotes in token text", () => {
    const parser = new SSEFrameParser();

    const events = parser.push('event: token\ndata: {"text":"Hello\\nWorld \\"quoted\\""}\n\n');

    expect(events).toEqual([{ type: "token", data: { text: 'Hello\nWorld "quoted"' } }]);
  });

  it("parses a tool_use frame with its input object", () => {
    const parser = new SSEFrameParser();

    const events = parser.push(
      'event: tool_use\ndata: {"id":"call_abc","name":"search_docs","input":{"query":"hello"}}\n\n',
    );

    expect(events).toEqual([
      { type: "tool_use", data: { id: "call_abc", name: "search_docs", input: { query: "hello" } } },
    ]);
  });

  it("parses a typed error frame", () => {
    const parser = new SSEFrameParser();

    const events = parser.push('event: error\ndata: {"type":"rate_limit","message":"Too many"}\n\n');

    expect(events).toEqual([
      { type: "error", data: { type: "rate_limit", message: "Too many" } },
    ]);
  });

  it("ignores unknown event types so future phases can add events", () => {
    const parser = new SSEFrameParser();

    const events = parser.push(
      'event: citation\ndata: {"path":"docs/index.md"}\n\nevent: done\ndata: {}\n\n',
    );

    expect(events).toEqual([{ type: "done", data: {} }]);
  });

  it("keeps unknown fields on a known event rather than dropping the event", () => {
    const parser = new SSEFrameParser();

    const events = parser.push('event: token\ndata: {"text":"hi","index":4}\n\n');

    expect(events).toEqual([{ type: "token", data: { text: "hi", index: 4 } }]);
  });

  it("skips a frame whose data is not valid JSON instead of throwing", () => {
    const parser = new SSEFrameParser();

    const events = parser.push("event: token\ndata: not-json\n\nevent: done\ndata: {}\n\n");

    expect(events).toEqual([{ type: "done", data: {} }]);
  });

  it("skips a frame with no event line", () => {
    const parser = new SSEFrameParser();

    expect(parser.push('data: {"text":"orphan"}\n\n')).toEqual([]);
  });

  it("ignores SSE comment lines used as keep-alives", () => {
    const parser = new SSEFrameParser();

    const events = parser.push(': keep-alive\n\nevent: token\ndata: {"text":"hi"}\n\n');

    expect(events).toEqual([{ type: "token", data: { text: "hi" } }]);
  });

  it("accepts CRLF line endings", () => {
    const parser = new SSEFrameParser();

    const events = parser.push('event: token\r\ndata: {"text":"hi"}\r\n\r\n');

    expect(events).toEqual([{ type: "token", data: { text: "hi" } }]);
  });

  it("tolerates a missing space after the field colon", () => {
    const parser = new SSEFrameParser();

    expect(parser.push('event:token\ndata:{"text":"hi"}\n\n')).toEqual([
      { type: "token", data: { text: "hi" } },
    ]);
  });
});

describe("parseSSEStream", () => {
  it("yields events across chunk boundaries in order", async () => {
    const events = await collect(
      streamOf(
        'event: token\ndata: {"text":"Hel',
        'lo"}\n\nevent: token\ndata: {"text":" world"}\n\n',
        "event: done\ndata: {}\n\n",
      ),
    );

    expect(events).toEqual([
      { type: "token", data: { text: "Hello" } },
      { type: "token", data: { text: " world" } },
      { type: "done", data: {} },
    ]);
  });

  it("reassembles a multi-byte character split across chunks", async () => {
    // "é" is 0xC3 0xA9 — split the two bytes into separate chunks.
    const encoder = new TextEncoder();
    const full = encoder.encode('event: token\ndata: {"text":"é"}\n\n');
    const splitAt = full.indexOf(0xc3) + 1;

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(full.slice(0, splitAt));
        controller.enqueue(full.slice(splitAt));
        controller.close();
      },
    });

    expect(await collect(stream)).toEqual([{ type: "token", data: { text: "é" } }]);
  });

  it("ends without a done event when the stream is cut mid-frame", async () => {
    const events = await collect(
      streamOf('event: token\ndata: {"text":"partial"}\n\nevent: token\ndata: {"text":"cut'),
    );

    expect(events).toEqual([{ type: "token", data: { text: "partial" } }]);
  });
});
