/**
 * SSE event contract — TypeScript types mirroring the backend wire schema.
 *
 * Wire framing (UTF-8):
 *   event: <type>\n
 *   data: <compact-single-line-json>\n
 *   \n
 *
 * This file contains TYPE DECLARATIONS ONLY. No parser, fetch logic, or
 * React components belong here. See docs/sse-contract.md for the full spec.
 *
 * Additive evolution rule: future phases MAY add fields to any payload.
 * Consumers must ignore unknown fields and keep working.
 */

export type SSEEventType = "token" | "tool_use" | "error" | "done";

export type SSEEvent =
  | { type: "token"; data: { text: string } }
  | { type: "tool_use"; data: { id: string; name: string; input: Record<string, unknown> } }
  | { type: "error"; data: { type: string; message: string } }
  | { type: "done"; data: { stop_reason?: string | null } };

export const SSE_EVENT_TYPES = ["token", "tool_use", "error", "done"] as const;
