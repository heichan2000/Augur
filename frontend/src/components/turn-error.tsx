"use client";

import type { TurnError as TurnErrorData } from "@/lib/chat-state";

import { DISABLED_BUTTON, SECONDARY_BUTTON } from "./button-styles";
import { PrimaryButton } from "./primary-button";

/**
 * How each error code is presented. The three tiers are deliberate:
 *
 *   transient — amber. Upstream and temporary; Retry re-sends unchanged.
 *   input     — neutral. The user must change something, so nothing is
 *               alarming and there is no Retry, only "Edit message".
 *   fault     — red, and the only red in the system. A real server failure.
 *
 * An unrecognised code (a future phase adding one) is treated as a fault so it
 * is surfaced loudly rather than swallowed.
 */
const PRESENTATION: Record<string, { tier: "transient" | "input" | "fault"; title: string }> = {
  rate_limit: { tier: "transient", title: "Provider rate limit reached" },
  provider_error: { tier: "transient", title: "The model provider failed" },
  invalid_request: { tier: "input", title: "That message couldn't be sent" },
  internal: { tier: "fault", title: "Something broke on our side" },
};

/**
 * The backend only writes a turn to conversation history once it completes
 * (`backend/app/chat.py` returns before persisting on every failure path). So a
 * turn that failed, was stopped, or was cut off is on screen but *not* in the
 * model's context — a follow-up question will not see it. Saying so beats
 * letting the transcript imply a shared history that doesn't exist.
 */
function UnsavedNotice() {
  return (
    <p className="font-mono text-[11px] leading-[1.5] text-faint">
      not saved to the conversation — the assistant won&apos;t recall it
    </p>
  );
}

const TIER_STYLES = {
  transient: {
    card: "border-accent-edge bg-accent-wash",
    chip: "bg-accent-chip text-accent",
  },
  input: {
    card: "border-edge bg-elevated",
    chip: "bg-edge-subtle text-secondary",
  },
  fault: {
    card: "border-danger-edge bg-danger-wash",
    chip: "bg-danger-chip text-danger",
  },
} as const;

export function TurnError({
  error,
  onRetry,
  onDiscard,
  busy = false,
}: {
  error: TurnErrorData;
  onRetry: () => void;
  onDiscard: () => void;
  /** True while another turn is in flight — one stream at a time, so both actions go inert. */
  busy?: boolean;
}) {
  const { tier, title } = PRESENTATION[error.code] ?? {
    tier: "fault" as const,
    title: "The request failed",
  };
  const styles = TIER_STYLES[tier];

  return (
    <div
      role="alert"
      className={`flex flex-col gap-2 rounded-[10px] border px-4 py-3.5 ${styles.card}`}
    >
      <div className="flex flex-wrap items-center gap-2.5">
        <span className={`rounded-[5px] px-[7px] py-0.5 font-mono text-[11px] ${styles.chip}`}>
          {error.code}
        </span>
        <span className="text-[14px] font-medium text-text">{title}</span>
      </div>

      <p className="text-[13.5px] leading-[1.6] text-secondary text-pretty">{error.message}</p>

      <UnsavedNotice />

      <div className="mt-1 flex items-center gap-3">
        {tier === "input" ? (
          <button
            type="button"
            onClick={onDiscard}
            disabled={busy}
            className={busy ? DISABLED_BUTTON : SECONDARY_BUTTON}
          >
            Edit message
          </button>
        ) : (
          <PrimaryButton onClick={onRetry} disabled={busy}>
            Retry
          </PrimaryButton>
        )}
      </div>
    </div>
  );
}

/**
 * The stream ended with neither `done` nor an `error` — the connection
 * dropped. The partial answer stays on screen, dimmed by the caller and
 * explicitly labelled incomplete, rather than silently freezing.
 */
export function InterruptedNotice({
  onRetry,
  busy = false,
}: {
  onRetry: () => void;
  /** True while another turn is in flight — one stream at a time. */
  busy?: boolean;
}) {
  return (
    <div
      role="alert"
      className="flex flex-wrap items-center gap-3 rounded-[10px] border border-edge bg-elevated px-4 py-3"
    >
      <span aria-hidden className="size-2 shrink-0 rounded-full bg-faint" />
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="text-[13.5px] font-medium text-text">Connection lost mid-answer</span>
        <span className="font-mono text-[11px] text-faint">
          the answer above is incomplete and may not have been saved
        </span>
      </div>
      <PrimaryButton onClick={onRetry} disabled={busy} className="ml-auto">
        Retry
      </PrimaryButton>
    </div>
  );
}

/** The user pressed Stop. Not a failure — no alert role, no colour. */
export function StoppedNotice() {
  return (
    <div className="flex items-center gap-2.5 font-mono text-[11px] text-faint">
      <span aria-hidden className="size-2 shrink-0 rounded-full bg-faint" />
      stopped · not saved to the conversation
    </div>
  );
}

/**
 * The two stop reasons that mean the response was cut off rather than
 * finished. Mirrors `TRUNCATION_STOP_REASONS` in `backend/app/provider.py`.
 *
 * The wire value set is open-ended (see the `done` event in
 * docs/sse-contract.md), so this is a positive match on the two values that
 * mean truncation, never a negative match on "not end_turn". A stop reason a
 * future model introduces renders as an ordinary complete turn until someone
 * decides otherwise.
 */
const TRUNCATION_STOP_REASONS = new Set(["max_tokens", "model_context_window_exceeded"]);

export function isTruncated(stopReason: string | null): boolean {
  return stopReason !== null && TRUNCATION_STOP_REASONS.has(stopReason);
}

/**
 * The model ran out of room mid-answer. Shaped like `StoppedNotice` — this is
 * not a failure and gets no colour, no alert role, and no buttons.
 *
 * The wording inverts the family's usual message because this is the one
 * notice whose turn *is* saved. Retry would re-send the original prompt and
 * leave the model seeing the same question twice; a canned "Continue" button
 * would put a user turn in the transcript that the user never typed. A user
 * who wants more types "continue", which works precisely because the
 * truncated turn is already in history.
 */
export function TruncatedNotice() {
  return (
    <div className="flex items-center gap-2.5 font-mono text-[11px] text-faint">
      <span aria-hidden className="size-2 shrink-0 rounded-full bg-faint" />
      response cut off at the length limit · saved to the conversation
    </div>
  );
}
