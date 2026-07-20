"use client";

import { useEffect, useRef, useState } from "react";

import type { AssistantTurn } from "@/lib/chat-state";
import { useChat } from "@/lib/use-chat";

import { Composer, type ComposerHandle } from "./composer";
import { EmptyState } from "./empty-state";
import { ThemeToggle } from "./theme-toggle";
import { AssistantMessage, UserMessage } from "./turns";

/** Enough of the session id to identify a conversation without the noise. */
const sessionLabel = (id: string) => id.slice(0, 8);

export function Chat() {
  const { state, sessionId, send, stop, retry, discard } = useChat();
  const [draft, setDraft] = useState("");
  const composerRef = useRef<ComposerHandle>(null);
  const threadEndRef = useRef<HTMLDivElement>(null);
  const busy = state.status === "busy";

  // Follow the newest turn as it streams.
  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [state.turns]);

  function submit(text: string) {
    send(text);
    setDraft("");
  }

  function handleDiscard(turn: AssistantTurn) {
    const original = discard(turn);
    if (original !== null) composerRef.current?.restore(original);
  }

  const isEmpty = state.turns.length === 0;

  return (
    <div className="mx-auto flex min-h-dvh w-full max-w-[1120px] flex-col bg-surface sm:my-6 sm:min-h-[calc(100dvh-3rem)] sm:rounded-2xl sm:border sm:border-edge">
      <header className="flex h-[52px] shrink-0 items-center justify-between gap-4 border-b border-edge-subtle px-5">
        <div className="flex min-w-0 items-center gap-3">
          <span className="text-[15px] font-semibold tracking-[0.02em] text-text">Augur</span>
          <span aria-hidden className="h-4 w-px shrink-0 bg-edge" />
          <span className="truncate font-mono text-[11px] text-faint">
            developer-docs assistant
          </span>
        </div>

        <div className="flex shrink-0 items-center gap-3">
          {sessionId !== null && (
            <span className="hidden font-mono text-[11px] text-faint sm:inline">
              session {sessionLabel(sessionId)}
            </span>
          )}
          <ThemeToggle />
        </div>
      </header>

      {isEmpty ? (
        <EmptyState onPick={submit} />
      ) : (
        <div className="flex-1 overflow-y-auto px-6 py-8">
          <div className="mx-auto flex w-full max-w-[768px] flex-col gap-[26px]">
            {state.turns.map((turn) =>
              turn.kind === "user" ? (
                <UserMessage key={turn.id} turn={turn} />
              ) : (
                <AssistantMessage
                  key={turn.id}
                  turn={turn}
                  onRetry={() => retry(turn)}
                  onDiscard={() => handleDiscard(turn)}
                />
              ),
            )}
            <div ref={threadEndRef} />
          </div>
        </div>
      )}

      <div className="flex shrink-0 justify-center px-6 pb-5">
        <Composer
          value={draft}
          onChange={setDraft}
          onSubmit={() => submit(draft)}
          onStop={stop}
          busy={busy}
          handleRef={composerRef}
        />
      </div>
    </div>
  );
}
