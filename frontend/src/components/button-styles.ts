/**
 * Shared button class strings.
 *
 * These live in one place because the same three shapes recur across the
 * composer and every error card — a colour or radius change should not mean
 * hunting down five duplicated Tailwind chains.
 */

const FOCUS_RING =
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

/** Filled, high-contrast. The one obvious action: Send, Retry. */
export const PRIMARY_BUTTON =
  `shrink-0 rounded-lg bg-text px-3.5 py-1.5 text-[13px] font-medium text-surface transition-opacity hover:opacity-90 ${FOCUS_RING}`;

/** Outlined. Sits beside a primary action without competing: Stop, Edit message. */
export const SECONDARY_BUTTON =
  `shrink-0 rounded-lg border border-edge-strong px-3.5 py-1.5 text-[13px] font-medium text-text transition-colors hover:bg-raised ${FOCUS_RING}`;

/** Filled but inert — Send with nothing to send. */
export const DISABLED_BUTTON =
  `shrink-0 cursor-default rounded-lg bg-edge px-3.5 py-1.5 text-[13px] font-medium text-faint ${FOCUS_RING}`;
