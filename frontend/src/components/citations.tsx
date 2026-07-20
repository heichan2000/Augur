/**
 * Citation and abstention UI.
 *
 * NOT RENDERED IN PHASE 1. The `/chat` SSE contract currently carries no
 * citation event and there is no docs corpus to cite, so nothing in the app
 * mounts these yet — deliberately, rather than inventing sources the backend
 * never sent.
 *
 * They exist now because the Phase 1 principle is that Phase 2 must be an
 * extension, not a rewrite: when `search_docs` starts returning chunk metadata,
 * the answer body drops <Citations> in below the prose and <Abstention> in
 * place of it, with no change to the turn layout.
 */

export type Citation = {
  /** Index used by the inline [n] marker in the prose. */
  index: number;
  /** Corpus-relative source file, e.g. `docs/tutorial/dependencies/index.md`. */
  path: string;
  /** Heading trail within that file, e.g. `First Steps`. */
  heading: string;
  /** Link to the section in the live documentation. */
  href: string;
};

function CitationRow({ citation, showIndex }: { citation: Citation; showIndex: boolean }) {
  return (
    <a
      href={citation.href}
      target="_blank"
      rel="noopener noreferrer"
      className="-mx-2 flex flex-wrap items-baseline gap-2.5 rounded-[7px] px-2 py-[5px] no-underline transition-colors hover:bg-raised focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
    >
      {showIndex && (
        <span className="shrink-0 font-mono text-[10.5px] text-faint">[{citation.index}]</span>
      )}
      <span className="font-mono text-[12px] text-accent-link">{citation.path}</span>
      <span className="text-[12px] text-secondary">› {citation.heading}</span>
      <span aria-hidden className="ml-auto text-[11px] text-faint">
        ↗
      </span>
    </a>
  );
}

/** The Sources block beneath a grounded answer. */
export function Citations({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;

  return (
    <section className="mt-1 flex flex-col gap-2 border-t border-edge-subtle pt-3.5">
      <h4 className="font-mono text-[10px] tracking-[0.12em] text-faint">SOURCES</h4>
      <div className="flex flex-col gap-0.5">
        {citations.map((citation) => (
          <CitationRow key={citation.index} citation={citation} showIndex />
        ))}
      </div>
    </section>
  );
}

/**
 * The docs don't cover the question. Styled as a confident statement, not an
 * error — abstaining is the honest outcome, and it still shows its work.
 */
export function Abstention({
  message,
  nearest,
  searchSummary,
}: {
  message: string;
  nearest: Citation[];
  searchSummary: string | null;
}) {
  return (
    <div className="flex flex-col gap-2.5 rounded-xl border border-edge bg-elevated px-[22px] py-5">
      <div className="flex items-center gap-3">
        <span aria-hidden className="size-3 shrink-0 rotate-45 border-[1.5px] border-accent" />
        <h3 className="text-[16px] font-semibold text-text">The docs don&apos;t cover this</h3>
      </div>

      <p className="text-[14.5px] leading-[1.7] text-secondary text-pretty">{message}</p>

      {nearest.length > 0 && (
        <div className="mt-0.5 flex flex-col gap-0.5">
          {nearest.map((citation) => (
            <CitationRow key={citation.path + citation.heading} citation={citation} showIndex={false} />
          ))}
        </div>
      )}

      {searchSummary !== null && (
        <span className="mt-1 font-mono text-[11px] text-faint">{searchSummary}</span>
      )}
    </div>
  );
}
