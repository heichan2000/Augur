"use client";

/**
 * First-run screen. Four real, clickable questions rather than marketing copy.
 *
 * The last one exercises `get_current_time`, so the tool-calling path is
 * visible end-to-end in a demo without needing the docs corpus.
 */
const EXAMPLES = [
  "How do I add dependency injection to a route?",
  "When should an endpoint be `async def` instead of `def`?",
  "How do I handle file uploads with UploadFile?",
  "What time is it right now?",
];

export function EmptyState({ onPick }: { onPick: (question: string) => void }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-6 py-12">
      <span aria-hidden className="mb-7 size-[26px] rotate-45 border-[1.5px] border-accent" />

      <h1 className="text-center text-[24px] font-semibold tracking-[-0.01em] text-text">
        Ask about FastAPI
      </h1>

      <p className="mt-2.5 max-w-[460px] text-center text-[14.5px] leading-[1.6] text-secondary text-pretty">
        Streaming answers from Claude, routed through Augur&apos;s tool-calling layer.
        Retrieval over the FastAPI documentation — with citations — lands in Phase&nbsp;2.
      </p>

      <div className="mt-9 grid w-full max-w-[640px] grid-cols-1 gap-3 sm:grid-cols-2">
        {EXAMPLES.map((question) => (
          <button
            key={question}
            type="button"
            onClick={() => onPick(question)}
            className="rounded-[10px] border border-edge bg-elevated px-4 py-3.5 text-left text-[13.5px] leading-[1.5] text-prose transition-colors hover:border-edge-strong hover:bg-raised hover:text-text focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            {question}
          </button>
        ))}
      </div>

      <span className="mt-7 text-center font-mono text-[11px] tracking-[0.06em] text-faint">
        STREAMING · TOOL-CALLING SPINE · PHASE 1
      </span>
    </div>
  );
}
