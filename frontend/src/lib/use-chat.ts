"use client";

import { useCallback, useEffect, useReducer, useRef, useSyncExternalStore } from "react";

import { createBrowserStore } from "./browser-store";
import { streamChatTurn } from "./chat-client";
import {
  chatReducer,
  initialChatState,
  isBusy,
  promptFor,
  type AssistantTurn,
  type ChatState,
} from "./chat-state";

const SESSION_STORAGE_KEY = "augur-session-id";

/**
 * The session id the backend keys conversation history by.
 *
 * sessionStorage *is* the store, so a reload continues the same conversation
 * while a new tab starts a fresh one. Reading it directly (rather than mirroring
 * it into React state) keeps the value stable across renders and SSR-safe: the
 * server snapshot is null, and the id appears once the client has mounted.
 */
/** Fallback when storage is unavailable (private mode); lives for this page. */
let ephemeralSessionId: string | null = null;

const sessionStore = createBrowserStore<string | null>(() => {
  try {
    return sessionStorage.getItem(SESSION_STORAGE_KEY) ?? ephemeralSessionId;
  } catch {
    return ephemeralSessionId;
  }
}, null);

function ensureSessionId(): void {
  if (sessionStore.read() !== null) return;

  const created = crypto.randomUUID();
  try {
    sessionStorage.setItem(SESSION_STORAGE_KEY, created);
  } catch {
    ephemeralSessionId = created;
  }
  sessionStore.notify();
}

export type UseChat = {
  state: ChatState;
  sessionId: string | null;
  send: (text: string) => void;
  stop: () => void;
  retry: (turn: AssistantTurn) => void;
  /** Removes a rejected turn and hands its text back for editing. */
  discard: (turn: AssistantTurn) => string | null;
};

/** The in-flight stream: its abort handle and the assistant turn it feeds. */
type ActiveStream = { controller: AbortController; turnId: string };

export function useChat(): UseChat {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const activeStreamRef = useRef<ActiveStream | null>(null);
  const sessionId = useSyncExternalStore(
    sessionStore.subscribe,
    sessionStore.read,
    sessionStore.readServer,
  );

  // sessionStorage is unavailable during SSR, so the id is minted on mount.
  useEffect(ensureSessionId, []);

  // Abandon any in-flight request if the view goes away.
  useEffect(() => () => activeStreamRef.current?.controller.abort(), []);

  // Derived once so send/retry depend on the boolean, not the whole state.
  const busy = isBusy(state);

  const consume = useCallback(
    async (session: string, message: string, assistantTurnId: string) => {
      const controller = new AbortController();
      const stream: ActiveStream = { controller, turnId: assistantTurnId };
      activeStreamRef.current = stream;

      try {
        for await (const event of streamChatTurn({
          sessionId: session,
          message,
          signal: controller.signal,
        })) {
          dispatch({ type: "sse", assistantTurnId, event });
        }
      } catch (error) {
        // `streamChatTurn` turns transport and parse failures into `error`
        // events, so a throw here is a defect it did not anticipate. The turn
        // still ends in the `finally` below — swallowing it would only hide a
        // bug, so it goes to the console on the way past.
        console.error("The chat stream threw unexpectedly.", error);
      } finally {
        // Completion, abort, and throw all land here, so no turn can be left
        // in `awaiting` with the composer locked. A turn that closed itself
        // (done / error / Stop) ignores this; one that did not is marked
        // interrupted, keeping its partial answer and offering Retry.
        dispatch({ type: "stream_ended", assistantTurnId });
        if (activeStreamRef.current === stream) activeStreamRef.current = null;
      }
    },
    [],
  );

  const send = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (trimmed === "" || sessionId === null || busy) return;

      const assistantTurnId = crypto.randomUUID();
      dispatch({
        type: "send",
        text: trimmed,
        userTurnId: crypto.randomUUID(),
        assistantTurnId,
      });
      void consume(sessionId, trimmed, assistantTurnId);
    },
    [busy, consume, sessionId],
  );

  const stop = useCallback(() => {
    const stream = activeStreamRef.current;
    if (stream === null) return;

    stream.controller.abort();
    dispatch({ type: "stopped", assistantTurnId: stream.turnId });
  }, []);

  const retry = useCallback(
    (turn: AssistantTurn) => {
      const message = promptFor(state, turn.id);
      if (message === null || sessionId === null || busy) return;

      dispatch({ type: "retry", assistantTurnId: turn.id });
      void consume(sessionId, message, turn.id);
    },
    [busy, consume, sessionId, state],
  );

  const discard = useCallback(
    (turn: AssistantTurn) => {
      // The Edit message button is disabled while busy; this guard covers any
      // other caller, so a turn can never vanish mid-stream.
      if (busy) return null;

      const message = promptFor(state, turn.id);
      dispatch({ type: "discard", assistantTurnId: turn.id });
      return message;
    },
    [busy, state],
  );

  return { state, sessionId, send, stop, retry, discard };
}
