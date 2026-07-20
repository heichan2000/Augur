"use client";

import { useCallback, useEffect, useReducer, useRef, useSyncExternalStore } from "react";

import { createBrowserStore } from "./browser-store";
import { streamChatTurn } from "./chat-client";
import {
  chatReducer,
  initialChatState,
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

export function useChat(): UseChat {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const abortRef = useRef<AbortController | null>(null);
  const sessionId = useSyncExternalStore(
    sessionStore.subscribe,
    sessionStore.read,
    sessionStore.readServer,
  );

  // sessionStorage is unavailable during SSR, so the id is minted on mount.
  useEffect(ensureSessionId, []);

  // Abandon any in-flight request if the view goes away.
  useEffect(() => () => abortRef.current?.abort(), []);

  const consume = useCallback(async (session: string, message: string) => {
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      for await (const event of streamChatTurn({
        sessionId: session,
        message,
        signal: controller.signal,
      })) {
        dispatch({ type: "sse", event });
      }
      // A turn that closed itself (done / error) ignores this; one that did
      // not is marked interrupted.
      dispatch({ type: "stream_ended" });
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, []);

  const send = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (trimmed === "" || sessionId === null || state.status === "busy") return;

      dispatch({
        type: "send",
        text: trimmed,
        userTurnId: crypto.randomUUID(),
        assistantTurnId: crypto.randomUUID(),
      });
      void consume(sessionId, trimmed);
    },
    [consume, sessionId, state.status],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    dispatch({ type: "stopped" });
  }, []);

  const retry = useCallback(
    (turn: AssistantTurn) => {
      const message = promptFor(state, turn.id);
      if (message === null || sessionId === null) return;

      dispatch({ type: "retry", assistantTurnId: turn.id });
      void consume(sessionId, message);
    },
    [consume, sessionId, state],
  );

  const discard = useCallback(
    (turn: AssistantTurn) => {
      const message = promptFor(state, turn.id);
      dispatch({ type: "discard", assistantTurnId: turn.id });
      return message;
    },
    [state],
  );

  return { state, sessionId, send, stop, retry, discard };
}
