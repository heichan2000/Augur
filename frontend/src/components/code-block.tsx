"use client";

import { useEffect, useRef, useState } from "react";

/**
 * A fenced code block: language label, copy button, and a body that scrolls
 * horizontally inside itself. Long lines must never widen the page — that is
 * the whole point of `overflow-x-auto` on the <pre> and `min-w-0` on every
 * ancestor that could otherwise stretch.
 */
export function CodeBlock({
  language,
  children,
}: {
  language: string | null;
  children: React.ReactNode;
}) {
  const [copied, setCopied] = useState(false);
  const preRef = useRef<HTMLPreElement>(null);
  const resetRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (resetRef.current !== null) clearTimeout(resetRef.current);
  }, []);

  async function copy() {
    const text = preRef.current?.textContent ?? "";
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      if (resetRef.current !== null) clearTimeout(resetRef.current);
      resetRef.current = setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard blocked (insecure origin, denied permission). The text is
      // still selectable, so fail quietly rather than showing a false success.
    }
  }

  return (
    <div className="overflow-hidden rounded-[10px] border border-edge bg-code-bg">
      <div className="flex items-center justify-between border-b border-edge-subtle px-3.5 py-2">
        <span className="font-mono text-[11px] tracking-[0.06em] text-faint">
          {language ?? "code"}
        </span>
        <button
          type="button"
          onClick={copy}
          className="rounded px-1 py-0.5 font-mono text-[11px] text-faint transition-colors hover:text-text focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
        >
          {copied ? "✓ Copied" : "⧉ Copy"}
        </button>
      </div>
      <pre
        ref={preRef}
        className="m-0 overflow-x-auto px-[18px] py-4 font-mono text-[13px] leading-[1.75] text-code-fg"
      >
        {children}
      </pre>
    </div>
  );
}
