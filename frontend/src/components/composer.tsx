"use client";

import { useEffect, useImperativeHandle, useRef, type RefObject } from "react";

import { SECONDARY_BUTTON } from "./button-styles";
import { PrimaryButton } from "./primary-button";

export type ComposerHandle = {
  /** Puts text back in the field and focuses it (used by "Edit message"). */
  restore: (text: string) => void;
};

/**
 * The message input. Grows with its content up to a cap, then scrolls.
 *
 * Enter sends; Shift+Enter inserts a newline — technical questions are often
 * multi-line. While a turn streams, the field is locked and Send becomes Stop.
 */
export function Composer({
  value,
  onChange,
  onSubmit,
  onStop,
  busy,
  handleRef,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  busy: boolean;
  handleRef?: RefObject<ComposerHandle | null>;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useImperativeHandle(handleRef, () => ({
    restore(text: string) {
      onChange(text);
      textareaRef.current?.focus();
    },
  }));

  // Re-fit on every change so the box tracks wrapped lines, not just newlines.
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea === null) return;

    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 220)}px`;
  }, [value]);

  // Esc stops a streaming turn from anywhere in the composer.
  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Escape" && busy) {
      event.preventDefault();
      onStop();
      return;
    }

    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!busy) onSubmit();
    }
  }

  const canSend = value.trim().length > 0 && !busy;

  return (
    <div
      className={`flex w-full max-w-[768px] flex-col gap-2.5 rounded-xl border bg-input px-3.5 py-3 transition-shadow focus-within:border-accent focus-within:shadow-[0_0_0_3px_var(--focus-ring)] ${
        busy ? "border-edge-subtle opacity-60" : "border-edge"
      }`}
    >
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={busy}
        rows={1}
        aria-label="Ask about FastAPI"
        placeholder={busy ? "Answering…" : "Ask about FastAPI…"}
        className="w-full resize-none bg-transparent text-[14.5px] leading-[1.55] text-text placeholder:text-faint focus:outline-none disabled:cursor-not-allowed"
      />

      <div className="flex items-center justify-between gap-3">
        <span className="font-mono text-[11px] text-faint">
          {busy ? "Composer locked while answering · Esc to stop" : "Enter to send · Shift+Enter for newline"}
        </span>

        {busy ? (
          <button
            type="button"
            onClick={onStop}
            className={`flex items-center gap-2 ${SECONDARY_BUTTON}`}
          >
            <span aria-hidden className="size-2 rounded-[1.5px] bg-text" />
            Stop
          </button>
        ) : (
          <PrimaryButton onClick={onSubmit} disabled={!canSend}>
            Send
          </PrimaryButton>
        )}
      </div>
    </div>
  );
}
