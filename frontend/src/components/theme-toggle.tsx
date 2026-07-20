"use client";

import { useSyncExternalStore } from "react";

import { createBrowserStore } from "@/lib/browser-store";

type Theme = "light" | "dark";

const STORAGE_KEY = "augur-theme";

/**
 * The `data-theme` attribute on <html> *is* the store — the inline bootstrap in
 * the layout writes it before first paint, so reading it here cannot flash the
 * wrong theme on hydration.
 */
const themeStore = createBrowserStore<Theme | null>(() => {
  const pinned = document.documentElement.dataset.theme;
  return pinned === "light" || pinned === "dark" ? pinned : null;
}, null);

function pinTheme(next: Theme): void {
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // Storage unavailable — the choice still applies for this page.
  }
  themeStore.notify();
}

/**
 * Pins the colour scheme to an explicit choice. Until the user picks one,
 * neither option reads as active and the OS preference wins.
 */
export function ThemeToggle() {
  const theme = useSyncExternalStore(
    themeStore.subscribe,
    themeStore.read,
    themeStore.readServer,
  );

  return (
    <div className="flex items-center gap-px rounded-md border border-edge p-px">
      {(["light", "dark"] as const).map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => pinTheme(option)}
          aria-pressed={theme === option}
          aria-label={`Use ${option} theme`}
          className={`rounded-[5px] px-2 py-1 font-mono text-[10px] tracking-[0.08em] uppercase transition-colors focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent ${
            theme === option ? "bg-raised text-text" : "text-faint hover:text-secondary"
          }`}
        >
          {option}
        </button>
      ))}
    </div>
  );
}
