"use client";

import type { ReactNode } from "react";

import { DISABLED_BUTTON, PRIMARY_BUTTON } from "./button-styles";

/**
 * The one obvious action, as a component: filled while actionable, inert while
 * not. Owns the disabled/primary switch once so Send and the Retry buttons
 * cannot drift apart.
 */
export function PrimaryButton({
  onClick,
  disabled = false,
  className = "",
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  /** Layout-only extras (e.g. `ml-auto`); the button's look is owned here. */
  className?: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`${className} ${disabled ? DISABLED_BUTTON : PRIMARY_BUTTON}`.trim()}
    >
      {children}
    </button>
  );
}
