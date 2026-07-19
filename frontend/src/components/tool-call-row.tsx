"use client";

import { useState } from "react";

import type { ToolCall } from "@/lib/chat-state";

/**
 * Friendlier phrasing for tools that are actually registered. Only
 * `get_current_time` exists in Phase 1; a tool added later renders fine
 * without an entry here, falling back to its raw name.
 */
const TOOL_LABELS: Record<string, { running: string; done: string }> = {
  get_current_time: { running: "Checking the current time…", done: "Checked the current time" },
};

function labelFor(call: ToolCall): string {
  const labels = TOOL_LABELS[call.name];
  if (labels === undefined) {
    return call.status === "running" ? `Running ${call.name}…` : `Ran ${call.name}`;
  }
  return call.status === "running" ? labels.running : labels.done;
}

const hasArguments = (call: ToolCall) => Object.keys(call.input).length > 0;

/**
 * One tool call in the thread: a quiet single line by default, expandable to
 * the raw arguments. It sits between message turns and must never compete
 * visually with the answer below it.
 *
 * Chained calls render as a sequence of these rows, so a multi-step turn reads
 * as a sequence rather than one badge.
 */
export function ToolCallRow({ call }: { call: ToolCall }) {
  const [expanded, setExpanded] = useState(false);
  const expandable = hasArguments(call);
  const label = labelFor(call);

  const summary = (
    <>
      {expandable ? (
        <span aria-hidden className="text-[11px] text-faint">
          {expanded ? "▾" : "▸"}
        </span>
      ) : (
        <span aria-hidden className="w-[7px]" />
      )}

      {call.status === "running" ? (
        <span
          aria-hidden
          className="augur-spin size-[11px] shrink-0 rounded-full border-[1.5px] border-transparent border-t-accent border-r-accent"
        />
      ) : (
        <span aria-hidden className="text-[12px] text-ok">
          ✓
        </span>
      )}

      <span className={`text-[13px] ${expanded ? "text-prose" : "text-secondary"}`}>{label}</span>

      <span className="rounded-[5px] border border-edge bg-raised px-1.5 py-px font-mono text-[11px] text-faint">
        {call.name}
      </span>

      {call.status === "running" && (
        <span className="augur-pulse ml-auto font-mono text-[11px] text-accent">running</span>
      )}
    </>
  );

  if (!expandable) {
    return <div className="flex items-center gap-2.5 py-[7px]">{summary}</div>;
  }

  return (
    <div
      className={
        expanded ? "my-1 flex flex-col rounded-[10px] border border-edge bg-elevated" : "contents"
      }
    >
      <button
        type="button"
        onClick={() => setExpanded((open) => !open)}
        aria-expanded={expanded}
        className={`flex w-full items-center gap-2.5 text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
          expanded ? "px-3.5 py-2.5" : "py-[7px]"
        }`}
      >
        {summary}
      </button>

      {expanded && (
        <div className="flex flex-col gap-2 px-3.5 pb-3">
          <span className="font-mono text-[10px] tracking-[0.1em] text-faint">ARGUMENTS</span>
          <pre className="m-0 overflow-x-auto rounded-lg border border-edge-subtle bg-code-bg px-3.5 py-2.5 font-mono text-[12px] leading-[1.7] text-code-fg">
            <code>{JSON.stringify(call.input, null, 2)}</code>
          </pre>
        </div>
      )}
    </div>
  );
}
