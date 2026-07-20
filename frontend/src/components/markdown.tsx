"use client";

import { Children, isValidElement, type ReactElement } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import { CodeBlock } from "./code-block";

/**
 * Renders an assistant answer.
 *
 * Answers over a documentation corpus are Markdown-heavy — prose, inline code,
 * fenced Python blocks, lists and tables. Every element is styled explicitly
 * here rather than left to browser defaults, and every element that can exceed
 * the measure (code, tables) scrolls inside its own container.
 */
export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
      components={{
        p: ({ children }) => (
          <p className="text-[15px] leading-[1.7] text-prose text-pretty">{children}</p>
        ),

        h1: ({ children }) => (
          <h2 className="text-[19px] font-semibold tracking-[-0.005em] text-text">{children}</h2>
        ),
        h2: ({ children }) => (
          <h3 className="text-[17px] font-semibold tracking-[-0.005em] text-text">{children}</h3>
        ),
        h3: ({ children }) => (
          <h4 className="text-[15px] font-semibold text-text">{children}</h4>
        ),
        h4: ({ children }) => (
          <h5 className="text-[14px] font-semibold text-text">{children}</h5>
        ),

        ul: ({ children }) => (
          <ul className="flex list-disc flex-col gap-1.5 pl-[22px] text-[15px] leading-[1.7] text-prose marker:text-faint">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="flex list-decimal flex-col gap-1.5 pl-[22px] text-[15px] leading-[1.7] text-prose marker:text-faint">
            {children}
          </ol>
        ),

        a: ({ href, children }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent-link underline-offset-2 hover:underline"
          >
            {children}
          </a>
        ),

        blockquote: ({ children }) => (
          <blockquote className="flex flex-col gap-2 rounded-[10px] border border-edge bg-elevated px-4 py-3 text-[14px] leading-[1.65] text-secondary">
            {children}
          </blockquote>
        ),

        hr: () => <hr className="border-edge-subtle" />,

        // Tables can be wider than the measure; scroll them, not the page.
        table: ({ children }) => (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[14px] text-prose">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border-b border-edge px-3 py-2 text-left font-mono text-[11px] font-medium tracking-[0.06em] text-faint uppercase">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border-b border-edge-subtle px-3 py-2.5 align-top">{children}</td>
        ),

        code: ({ className, children }) => {
          // rehype-highlight tags block code with `language-*`; anything else
          // is an inline span.
          if (typeof className === "string" && className.includes("language-")) {
            return <code className={className}>{children}</code>;
          }
          return (
            <code className="rounded-[5px] border border-edge bg-raised px-[5px] py-px font-mono text-[13px] text-text">
              {children}
            </code>
          );
        },

        pre: ({ children }) => {
          const child = Children.toArray(children)[0];
          const className =
            isValidElement(child) &&
            typeof (child as ReactElement<{ className?: string }>).props.className === "string"
              ? (child as ReactElement<{ className?: string }>).props.className!
              : "";
          const language = /language-([\w-]+)/.exec(className)?.[1] ?? null;

          return <CodeBlock language={language}>{children}</CodeBlock>;
        },
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
