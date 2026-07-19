/**
 * Incremental parser for the Augur `/chat` SSE wire format.
 *
 * See docs/sse-contract.md. The wire framing is:
 *
 *   event: <type>\n
 *   data: <compact-single-line-json>\n
 *   \n
 *
 * Two seams live here:
 *   - `SSEFrameParser` — pure, synchronous, chunk-in / events-out. Buffers a
 *     partial frame across pushes so network chunking is invisible to callers.
 *   - `parseSSEStream` — adapts a byte stream to an async iterable of events,
 *     decoding UTF-8 across chunk boundaries.
 *
 * Per the contract's additive-evolution rule, this parser must survive things
 * it does not understand: unknown event types are dropped, unknown fields on a
 * known event are passed through untouched, and a malformed frame is skipped
 * rather than killing the stream.
 */
import { SSE_EVENT_TYPES, type SSEEvent, type SSEEventType } from "./sse";

const KNOWN_EVENT_TYPES: ReadonlySet<string> = new Set(SSE_EVENT_TYPES);

/** Matches the blank line that terminates an SSE frame, in LF or CRLF form. */
const FRAME_BOUNDARY = /\r?\n\r?\n/g;

/** Strips the field name and the single optional space after the colon. */
function readField(line: string, name: string): string | null {
  if (!line.startsWith(`${name}:`)) return null;
  const value = line.slice(name.length + 1);
  return value.startsWith(" ") ? value.slice(1) : value;
}

/**
 * Parse one frame's text into an event, or null if it should be ignored:
 * a comment-only frame, a frame with no `event:` line, an event type this
 * version doesn't know, or a `data:` payload that isn't a JSON object.
 */
function parseFrame(frame: string): SSEEvent | null {
  let eventType: string | null = null;
  const dataLines: string[] = [];

  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith(":")) continue; // keep-alive comment

    const event = readField(line, "event");
    if (event !== null) {
      eventType = event;
      continue;
    }

    const data = readField(line, "data");
    if (data !== null) dataLines.push(data);
  }

  if (eventType === null || !KNOWN_EVENT_TYPES.has(eventType)) return null;
  if (dataLines.length === 0) return null;

  let payload: unknown;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return null;
  }

  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return null;

  // The event type is validated above; the payload is trusted to match the
  // contract for that type. Unknown fields ride along untouched by design.
  return { type: eventType as SSEEventType, data: payload } as SSEEvent;
}

export class SSEFrameParser {
  private buffer = "";

  /** Feed decoded text; returns every event completed by this chunk. */
  push(chunk: string): SSEEvent[] {
    this.buffer += chunk;

    const events: SSEEvent[] = [];
    let consumed = 0;
    let match: RegExpExecArray | null;

    FRAME_BOUNDARY.lastIndex = 0;
    while ((match = FRAME_BOUNDARY.exec(this.buffer)) !== null) {
      const event = parseFrame(this.buffer.slice(consumed, match.index));
      if (event !== null) events.push(event);
      consumed = match.index + match[0].length;
    }

    if (consumed > 0) this.buffer = this.buffer.slice(consumed);
    return events;
  }
}

/**
 * Decode a byte stream into SSE events. A stream that ends mid-frame simply
 * stops yielding — the incomplete frame is discarded, never half-emitted. The
 * caller distinguishes a clean end (a `done` event arrived) from a dropped
 * connection (it did not).
 */
export async function* parseSSEStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<SSEEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEFrameParser();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      yield* parser.push(decoder.decode(value, { stream: true }));
    }

    // Flush any bytes the decoder was holding for a split code point.
    const tail = decoder.decode();
    if (tail.length > 0) yield* parser.push(tail);
  } finally {
    reader.releaseLock();
  }
}
