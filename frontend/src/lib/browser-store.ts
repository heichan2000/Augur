/**
 * A tiny `useSyncExternalStore` adapter for values that live in the browser
 * rather than in React — the theme (a `data-` attribute on <html>) and the
 * session id (a sessionStorage entry).
 *
 * Reading the browser directly, instead of mirroring into React state, keeps
 * those values stable across renders and avoids the setState-in-effect pattern.
 * `read` must return a value that is stable between changes, or React will
 * re-render forever; returning a string or null from storage satisfies that.
 */
export type BrowserStore<T> = {
  subscribe: (onChange: () => void) => () => void;
  read: () => T;
  /** Nothing browser-held exists during SSR, so this is always the empty value. */
  readServer: () => T;
  /** Call after writing to the browser, to re-render subscribers. */
  notify: () => void;
};

export function createBrowserStore<T>(read: () => T, serverValue: T): BrowserStore<T> {
  const listeners = new Set<() => void>();

  return {
    subscribe(onChange) {
      listeners.add(onChange);
      return () => {
        listeners.delete(onChange);
      };
    },
    read,
    readServer: () => serverValue,
    notify() {
      for (const listener of listeners) listener();
    },
  };
}
