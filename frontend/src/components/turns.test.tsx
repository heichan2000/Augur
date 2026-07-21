import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AssistantTurn, ToolCall } from "@/lib/chat-state";

import { AssistantMessage } from "./turns";

afterEach(cleanup);

function turnWith(overrides: Partial<AssistantTurn> = {}): AssistantTurn {
  return {
    kind: "assistant",
    id: "a1",
    text: "",
    toolCalls: [],
    status: "awaiting",
    error: null,
    stopReason: null,
    ...overrides,
  };
}

function renderTurn(turn: AssistantTurn, { busy = false } = {}) {
  const onRetry = vi.fn();
  const onDiscard = vi.fn();
  render(<AssistantMessage turn={turn} onRetry={onRetry} onDiscard={onDiscard} busy={busy} />);
  return { onRetry, onDiscard };
}

describe("awaiting the first token", () => {
  it("shows a status line rather than an empty slot", () => {
    renderTurn(turnWith({ status: "awaiting" }));

    expect(screen.getByText("Thinking…")).toBeInTheDocument();
  });
});

describe("streaming", () => {
  it("renders the partial text and marks the region busy", () => {
    renderTurn(turnWith({ status: "streaming", text: "FastAPI resolves" }));

    expect(screen.getByText(/FastAPI resolves/)).toBeInTheDocument();
    // aria-busy tells assistive tech the answer is still arriving.
    const live = document.querySelector("[aria-live='polite']");
    expect(live).toHaveAttribute("aria-busy", "true");
  });

  it("drops the busy flag once the turn completes", () => {
    renderTurn(turnWith({ status: "complete", text: "Done." }));

    expect(document.querySelector("[aria-live='polite']")).toHaveAttribute("aria-busy", "false");
  });
});

describe("markdown rendering", () => {
  it("renders a fenced code block with its language and a copy button", () => {
    renderTurn(
      turnWith({
        status: "complete",
        text: "Use it like this:\n\n```python\nfrom fastapi import Depends\n```",
      }),
    );

    expect(screen.getByText("python")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /copy/i })).toBeInTheDocument();
    // Highlighting splits the source across spans, so assert on the text content.
    expect(document.querySelector("pre")?.textContent).toContain("from fastapi import Depends");
  });

  it("renders GFM tables", () => {
    renderTurn(
      turnWith({
        status: "complete",
        text: "| You want | Use |\n| --- | --- |\n| Shared params | Depends(fn) |",
      }),
    );

    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "You want" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "Depends(fn)" })).toBeInTheDocument();
  });

  it("copies the code block body to the clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    renderTurn(turnWith({ status: "complete", text: "```python\nprint(1)\n```" }));
    await userEvent.click(screen.getByRole("button", { name: /copy/i }));

    expect(writeText).toHaveBeenCalledWith("print(1)\n");
  });
});

describe("tool calls", () => {
  const call = (overrides: Partial<ToolCall> = {}): ToolCall => ({
    id: "t1",
    name: "search_docs",
    input: { query: "dependency injection" },
    status: "done",
    ...overrides,
  });

  it("renders a chain of calls as a sequence of rows", () => {
    renderTurn(
      turnWith({
        status: "streaming",
        text: "Answer",
        toolCalls: [
          call({ id: "t1", name: "get_current_time", input: {} }),
          call({ id: "t2", name: "search_docs" }),
        ],
      }),
    );

    expect(screen.getByText("Checked the current time")).toBeInTheDocument();
    expect(screen.getByText("Ran search_docs")).toBeInTheDocument();
  });

  it("labels a running call and shows a running indicator", () => {
    renderTurn(
      turnWith({ toolCalls: [call({ name: "get_current_time", status: "running" })] }),
    );

    expect(screen.getByText("Checking the current time…")).toBeInTheDocument();
    expect(screen.getByText("running")).toBeInTheDocument();
  });

  it("hides raw arguments until the row is expanded", async () => {
    renderTurn(turnWith({ toolCalls: [call()] }));

    expect(screen.queryByText("ARGUMENTS")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { expanded: false }));

    expect(screen.getByText("ARGUMENTS")).toBeInTheDocument();
    expect(screen.getByText(/dependency injection/)).toBeInTheDocument();
  });

  it("is not expandable when the tool takes no arguments", () => {
    renderTurn(turnWith({ toolCalls: [call({ name: "get_current_time", input: {} })] }));

    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("falls back to the raw name for a tool it does not know", () => {
    renderTurn(turnWith({ toolCalls: [call({ name: "search_arxiv", input: {} })] }));

    expect(screen.getByText("Ran search_arxiv")).toBeInTheDocument();
  });
});

describe("typed errors", () => {
  it("offers Retry for a transient rate limit", async () => {
    const { onRetry } = renderTurn(
      turnWith({
        status: "failed",
        error: { code: "rate_limit", message: "Too many requests" },
      }),
    );

    const alert = screen.getByRole("alert");
    expect(within(alert).getByText("rate_limit")).toBeInTheDocument();
    expect(within(alert).getByText("Provider rate limit reached")).toBeInTheDocument();

    await userEvent.click(within(alert).getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it("offers Edit message instead of Retry when the input was rejected", async () => {
    const { onDiscard } = renderTurn(
      turnWith({
        status: "failed",
        error: { code: "invalid_request", message: "Message too long" },
      }),
    );

    const alert = screen.getByRole("alert");
    expect(within(alert).queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();

    await userEvent.click(within(alert).getByRole("button", { name: "Edit message" }));
    expect(onDiscard).toHaveBeenCalledOnce();
  });

  it("surfaces an unrecognised code rather than swallowing it", () => {
    renderTurn(
      turnWith({ status: "failed", error: { code: "quota_exceeded", message: "New code" } }),
    );

    expect(screen.getByText("quota_exceeded")).toBeInTheDocument();
    expect(screen.getByText("The request failed")).toBeInTheDocument();
  });

  it("keeps the partial answer visible alongside the error", () => {
    renderTurn(
      turnWith({
        status: "failed",
        text: "Half an answer",
        error: { code: "internal", message: "Boom" },
      }),
    );

    expect(screen.getByText(/Half an answer/)).toBeInTheDocument();
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });
});

describe("interrupted and stopped turns", () => {
  it("marks a dropped connection as incomplete and offers Retry", async () => {
    const { onRetry } = renderTurn(turnWith({ status: "interrupted", text: "Half an ans" }));

    expect(screen.getByText("Connection lost mid-answer")).toBeInTheDocument();
    expect(screen.getByText(/the answer above is incomplete/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it("disables Retry on a failed turn while another turn is in flight", async () => {
    const { onRetry } = renderTurn(
      turnWith({ status: "failed", error: { code: "rate_limit", message: "429" } }),
      { busy: true },
    );

    const retry = screen.getByRole("button", { name: "Retry" });
    expect(retry).toBeDisabled();

    await userEvent.click(retry);
    expect(onRetry).not.toHaveBeenCalled();
  });

  it("disables Edit message on a rejected turn while another turn is in flight", async () => {
    const { onDiscard } = renderTurn(
      turnWith({ status: "failed", error: { code: "invalid_request", message: "Too long" } }),
      { busy: true },
    );

    const edit = screen.getByRole("button", { name: "Edit message" });
    expect(edit).toBeDisabled();

    await userEvent.click(edit);
    expect(onDiscard).not.toHaveBeenCalled();
  });

  it("disables Retry on an interrupted turn while another turn is in flight", async () => {
    const { onRetry } = renderTurn(turnWith({ status: "interrupted", text: "Half" }), {
      busy: true,
    });

    expect(screen.getByRole("button", { name: "Retry" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).not.toHaveBeenCalled();
  });

  it("presents a stopped turn without an alert", () => {
    renderTurn(turnWith({ status: "stopped", text: "Enough" }));

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText(/stopped/)).toBeInTheDocument();
  });
});

describe("truncated turns", () => {
  it("says the answer was cut off at the length limit", () => {
    renderTurn(turnWith({ status: "complete", text: "The three main ", stopReason: "max_tokens" }));

    expect(screen.getByText(/cut off at the length limit/)).toBeInTheDocument();
  });

  it("shows the same notice when the context window was exceeded", () => {
    renderTurn(
      turnWith({
        status: "complete",
        text: "The three main ",
        stopReason: "model_context_window_exceeded",
      }),
    );

    expect(screen.getByText(/cut off at the length limit/)).toBeInTheDocument();
  });

  it("says the truncated answer was saved, and offers no way to act on it", () => {
    renderTurn(turnWith({ status: "complete", text: "The three main ", stopReason: "max_tokens" }));

    expect(screen.getByText(/saved to the conversation/)).toBeInTheDocument();
    expect(screen.queryByText(/not saved to the conversation/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("does not dim a truncated answer — the text is the real answer", () => {
    renderTurn(turnWith({ status: "complete", text: "The three main ", stopReason: "max_tokens" }));

    expect(screen.getByText(/The three main/).closest(".opacity-65")).toBeNull();
  });

  it("shows nothing for a turn that ended normally", () => {
    renderTurn(turnWith({ status: "complete", text: "Done.", stopReason: "end_turn" }));

    expect(screen.queryByText(/cut off at the length limit/)).not.toBeInTheDocument();
  });

  it("shows nothing when there is no stop reason", () => {
    renderTurn(turnWith({ status: "complete", text: "Done.", stopReason: null }));

    expect(screen.queryByText(/cut off at the length limit/)).not.toBeInTheDocument();
  });

  it("shows nothing for a stop reason it does not recognise", () => {
    renderTurn(turnWith({ status: "complete", text: "Done.", stopReason: "banana" }));

    expect(screen.queryByText(/cut off at the length limit/)).not.toBeInTheDocument();
  });

  it("shows nothing on a turn still streaming toward that stop reason", () => {
    renderTurn(turnWith({ status: "streaming", text: "The three ", stopReason: null }));

    expect(screen.queryByText(/cut off at the length limit/)).not.toBeInTheDocument();
  });
});
