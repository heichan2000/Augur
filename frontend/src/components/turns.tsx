"use client";

import type { AssistantTurn, UserTurn } from "@/lib/chat-state";

import { Markdown } from "./markdown";
import { ToolCallRow } from "./tool-call-row";
import { InterruptedNotice, StoppedNotice, TurnError } from "./turn-error";

export function UserMessage({ turn }: { turn: UserTurn }) {
  return (
    <div className="max-w-[600px] self-end rounded-xl border border-edge bg-raised px-[15px] py-2.5 text-[14.5px] leading-[1.55] whitespace-pre-wrap text-text">
      {turn.text}
    </div>
  );
}

/**
 * An assistant turn: tool activity, then the answer, then whatever ended it.
 *
 * Assistant turns are full-width prose rather than bubbles — a long technical
 * answer with code blocks should read as a document.
 */
export function AssistantMessage({
  turn,
  onRetry,
  onDiscard,
  retryDisabled = false,
}: {
  turn: AssistantTurn;
  onRetry: () => void;
  onDiscard: () => void;
  /** True while another turn is in flight — retrying now would race it. */
  retryDisabled?: boolean;
}) {
  const isStreaming = turn.status === "streaming";
  const isIncomplete = turn.status === "interrupted" || turn.status === "stopped";
  const hasText = turn.text.length > 0;

  return (
    <div className="flex min-w-0 flex-col gap-3.5">
      <h2 className="font-mono text-[10px] tracking-[0.12em] text-faint">AUGUR</h2>

      {turn.toolCalls.length > 0 && (
        <div className="flex flex-col">
          {turn.toolCalls.map((call) => (
            <ToolCallRow key={call.id} call={call} />
          ))}
        </div>
      )}

      {/* Before the first token: a status line, never a blank slot. */}
      {turn.status === "awaiting" && (
        <div className="flex items-center gap-2.5">
          <span aria-hidden className="augur-pulse size-2 rounded-full bg-accent" />
          <span className="text-[14px] text-secondary">Thinking…</span>
        </div>
      )}

      {hasText && (
        <div
          // Streaming text is announced politely so screen readers follow the
          // answer without being interrupted on every token.
          aria-live="polite"
          aria-busy={isStreaming}
          className={`flex min-w-0 flex-col gap-3.5 ${isIncomplete ? "opacity-65" : ""}`}
        >
          <Markdown>{turn.text}</Markdown>
          {isStreaming && <span aria-hidden className="augur-cursor -mt-3.5 self-start" />}
        </div>
      )}

      {turn.status === "failed" && turn.error !== null && (
        <TurnError
          error={turn.error}
          onRetry={onRetry}
          onDiscard={onDiscard}
          retryDisabled={retryDisabled}
        />
      )}

      {turn.status === "interrupted" && (
        <InterruptedNotice onRetry={onRetry} retryDisabled={retryDisabled} />
      )}

      {turn.status === "stopped" && <StoppedNotice />}
    </div>
  );
}
