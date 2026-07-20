/**
 * Transport for one chat turn: POST the message, stream back SSE events.
 *
 * Everything the caller needs arrives as an `SSEEvent`, including transport
 * failures — a refused connection or a non-2xx response is normalised into an
 * `error` event with one of the contract's codes. That keeps the UI's error
 * handling to a single path.
 */
import { parseSSEStream } from "./sse-parser";
import type { SSEEvent } from "./sse";

/** Same-origin proxy to the FastAPI backend; see src/app/api/chat/route.ts. */
const CHAT_ENDPOINT = "/api/chat";

function errorEvent(type: string, message: string): SSEEvent {
  return { type: "error", data: { type, message } };
}

/** A non-2xx proxy response carries a typed body; fall back if it doesn't. */
async function errorFromResponse(response: Response): Promise<SSEEvent> {
  try {
    const body: unknown = await response.json();
    if (
      typeof body === "object" &&
      body !== null &&
      "type" in body &&
      "message" in body &&
      typeof body.type === "string" &&
      typeof body.message === "string"
    ) {
      return errorEvent(body.type, body.message);
    }
  } catch {
    // Body was not JSON — fall through to the generic message.
  }

  return errorEvent("internal", `The request failed with status ${response.status}.`);
}

export type ChatTurnRequest = {
  sessionId: string;
  message: string;
  /** Abort to stop generation; the stream simply ends. */
  signal: AbortSignal;
};

export async function* streamChatTurn({
  sessionId,
  message,
  signal,
}: ChatTurnRequest): AsyncGenerator<SSEEvent> {
  let response: Response;

  try {
    response = await fetch(CHAT_ENDPOINT, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
      signal,
    });
  } catch {
    // An abort is the user pressing Stop, not a failure — end quietly and let
    // the caller record it as `stopped`.
    if (signal.aborted) return;
    yield errorEvent("internal", "Could not reach the server. Check your connection.");
    return;
  }

  if (!response.ok || response.body === null) {
    yield await errorFromResponse(response);
    return;
  }

  try {
    yield* parseSSEStream(response.body);
  } catch {
    // The connection dropped mid-stream. Yield nothing: the absence of a
    // `done` event is what tells the reducer the answer is incomplete.
    return;
  }
}
