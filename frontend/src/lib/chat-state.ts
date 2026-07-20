/**
 * Conversation state for the chat view.
 *
 * The backend owns conversation history (see the Phase 1 spec: the frontend is
 * a thin client). This reducer owns only what the *current* screen needs —
 * which turns to draw and what each one is doing right now. It is pure and
 * id's are supplied by the caller, so every transition is testable without
 * timers, randomness, or a network.
 *
 * The status vocabulary distinguishes the four ways a turn can stop, because
 * the design treats them differently:
 *   complete    — `done` arrived; normal ending
 *   failed      — a typed `error` event arrived; show the error card
 *   interrupted — the stream ended with neither; the answer is partial
 *   stopped     — the user hit Stop; the answer is partial but intentionally so
 */
import type { SSEEvent } from "./sse";

export type ToolCallStatus = "running" | "done";

export type ToolCall = {
  id: string;
  name: string;
  input: Record<string, unknown>;
  status: ToolCallStatus;
};

/**
 * The four codes the backend emits today (docs/sse-contract.md). Kept open to
 * `string` so a code added in a later phase still renders as a real error
 * instead of being silently swallowed.
 */
export type KnownErrorCode = "rate_limit" | "provider_error" | "invalid_request" | "internal";
export type ErrorCode = KnownErrorCode | (string & {});

export type TurnError = { code: ErrorCode; message: string };

export type AssistantTurnStatus =
  | "awaiting"
  | "streaming"
  | "complete"
  | "failed"
  | "interrupted"
  | "stopped";

export type UserTurn = { kind: "user"; id: string; text: string };

export type AssistantTurn = {
  kind: "assistant";
  id: string;
  text: string;
  toolCalls: ToolCall[];
  status: AssistantTurnStatus;
  error: TurnError | null;
};

export type Turn = UserTurn | AssistantTurn;

export type ChatState = {
  turns: Turn[];
  /** `busy` while a turn is in flight — the composer is locked. */
  status: "idle" | "busy";
};

export type ChatAction =
  | { type: "send"; text: string; userTurnId: string; assistantTurnId: string }
  | { type: "sse"; event: SSEEvent }
  | { type: "stream_ended" }
  | { type: "stopped" }
  | { type: "retry"; assistantTurnId: string }
  | { type: "discard"; assistantTurnId: string };

export const initialChatState: ChatState = { turns: [], status: "idle" };

/** A turn is still open to stream events only in these two states. */
function isOpen(turn: AssistantTurn): boolean {
  return turn.status === "awaiting" || turn.status === "streaming";
}

export function lastAssistantTurn(state: ChatState): AssistantTurn | null {
  for (let i = state.turns.length - 1; i >= 0; i--) {
    const turn = state.turns[i];
    if (turn.kind === "assistant") return turn;
  }
  return null;
}

/**
 * The text of the user turn that prompted the given assistant turn — what a
 * retry re-sends, and what "Edit message" restores to the composer.
 */
export function promptFor(state: ChatState, assistantTurnId: string): string | null {
  const index = state.turns.findIndex(
    (turn) => turn.kind === "assistant" && turn.id === assistantTurnId,
  );
  if (index <= 0) return null;

  const previous = state.turns[index - 1];
  return previous.kind === "user" ? previous.text : null;
}

function settleToolCalls(toolCalls: ToolCall[]): ToolCall[] {
  if (!toolCalls.some((call) => call.status === "running")) return toolCalls;
  return toolCalls.map((call) =>
    call.status === "running" ? { ...call, status: "done" as const } : call,
  );
}

/**
 * Apply `update` to the open assistant turn. Events arriving after a turn has
 * closed (a late frame after `done`, or anything after the user hit Stop) are
 * ignored rather than reopening it.
 */
function updateOpenTurn(
  state: ChatState,
  update: (turn: AssistantTurn) => AssistantTurn,
): ChatState {
  const index = state.turns.findLastIndex((turn) => turn.kind === "assistant");
  if (index === -1) return state;

  const turn = state.turns[index] as AssistantTurn;
  if (!isOpen(turn)) return state;

  const turns = [...state.turns];
  turns[index] = update(turn);
  return { ...state, turns };
}

/**
 * End the open turn: give it a terminal status, settle any tool still marked
 * running, and free the composer. Every way a turn can stop routes through here.
 */
function closeTurn(
  state: ChatState,
  status: Extract<
    AssistantTurnStatus,
    "complete" | "failed" | "interrupted" | "stopped"
  >,
  error: TurnError | null = null,
): ChatState {
  const next = updateOpenTurn(state, (turn) => ({
    ...turn,
    status,
    toolCalls: settleToolCalls(turn.toolCalls),
    error,
  }));
  return { ...next, status: "idle" };
}

function applyEvent(state: ChatState, event: SSEEvent): ChatState {
  switch (event.type) {
    case "token":
      return updateOpenTurn(state, (turn) => ({
        ...turn,
        status: "streaming",
        text: turn.text + event.data.text,
        // Text means the model is past its tools for this round.
        toolCalls: settleToolCalls(turn.toolCalls),
      }));

    case "tool_use":
      return updateOpenTurn(state, (turn) => ({
        ...turn,
        // The backend emits no tool-result event, so a tool is treated as
        // finished once the next thing arrives — the following tool_use,
        // the first token, or `done`.
        toolCalls: [
          ...settleToolCalls(turn.toolCalls),
          {
            id: event.data.id,
            name: event.data.name,
            input: event.data.input,
            status: "running",
          },
        ],
      }));

    case "error":
      return closeTurn(state, "failed", {
        code: event.data.type,
        message: event.data.message,
      });

    case "done":
      return closeTurn(state, "complete");
  }
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "send":
      return {
        status: "busy",
        turns: [
          ...state.turns,
          { kind: "user", id: action.userTurnId, text: action.text },
          {
            kind: "assistant",
            id: action.assistantTurnId,
            text: "",
            toolCalls: [],
            status: "awaiting",
            error: null,
          },
        ],
      };

    case "sse":
      return applyEvent(state, action.event);

    case "stream_ended":
      // Only meaningful if the turn never closed itself — that is a dropped
      // connection, and the partial answer stays on screen marked incomplete.
      return closeTurn(state, "interrupted");

    case "stopped":
      return closeTurn(state, "stopped");

    case "retry": {
      const turns = state.turns.map((turn) =>
        turn.kind === "assistant" && turn.id === action.assistantTurnId
          ? { ...turn, text: "", toolCalls: [], status: "awaiting" as const, error: null }
          : turn,
      );
      return { turns, status: "busy" };
    }

    case "discard": {
      const index = state.turns.findIndex(
        (turn) => turn.kind === "assistant" && turn.id === action.assistantTurnId,
      );
      if (index === -1) return state;

      // Drop the assistant turn and the user turn that prompted it, so the
      // rejected message can go back to the composer for editing.
      const start = index > 0 && state.turns[index - 1].kind === "user" ? index - 1 : index;
      return {
        turns: [...state.turns.slice(0, start), ...state.turns.slice(index + 1)],
        status: "idle",
      };
    }
  }
}
