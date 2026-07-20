import { describe, expect, it } from "vitest";

import {
  chatReducer,
  initialChatState,
  lastAssistantTurn,
  type ChatAction,
  type ChatState,
} from "./chat-state";

/** Apply a sequence of actions, starting from a fresh conversation. */
function run(...actions: ChatAction[]): ChatState {
  return actions.reduce(chatReducer, initialChatState);
}

const send: ChatAction = {
  type: "send",
  text: "How do I add dependency injection?",
  userTurnId: "u1",
  assistantTurnId: "a1",
};

function assistant(state: ChatState) {
  const turn = lastAssistantTurn(state);
  if (turn === null) throw new Error("expected an assistant turn");
  return turn;
}

describe("send", () => {
  it("appends the user turn and an assistant turn awaiting its first token", () => {
    const state = run(send);

    expect(state.turns).toEqual([
      { kind: "user", id: "u1", text: "How do I add dependency injection?" },
      {
        kind: "assistant",
        id: "a1",
        text: "",
        toolCalls: [],
        status: "awaiting",
        error: null,
      },
    ]);
    expect(state.status).toBe("busy");
  });

  it("keeps earlier turns when sending a follow-up", () => {
    const state = run(
      send,
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "Use Depends." } } },
      { type: "sse", assistantTurnId: "a1", event: { type: "done", data: {} } },
      { type: "send", text: "And sub-dependencies?", userTurnId: "u2", assistantTurnId: "a2" },
    );

    expect(state.turns).toHaveLength(4);
    expect(state.turns[1]).toMatchObject({ id: "a1", status: "complete", text: "Use Depends." });
    expect(state.turns[3]).toMatchObject({ id: "a2", status: "awaiting" });
  });
});

describe("token events", () => {
  it("moves the turn to streaming on the first token", () => {
    const state = run(send, { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "Fast" } } });

    expect(assistant(state)).toMatchObject({ status: "streaming", text: "Fast" });
  });

  it("accumulates token text in arrival order", () => {
    const state = run(
      send,
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "Fast" } } },
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "API " } } },
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "resolves it." } } },
    );

    expect(assistant(state).text).toBe("FastAPI resolves it.");
  });
});

describe("tool_use events", () => {
  const toolUse = (id: string, name: string): ChatAction => ({
    type: "sse",
    assistantTurnId: "a1",
    event: { type: "tool_use", data: { id, name, input: { query: "deps" } } },
  });

  it("records a tool call as running", () => {
    const state = run(send, toolUse("t1", "search_docs"));

    expect(assistant(state).toolCalls).toEqual([
      { id: "t1", name: "search_docs", input: { query: "deps" }, status: "running" },
    ]);
  });

  it("settles the previous tool call when the next one starts", () => {
    const state = run(send, toolUse("t1", "get_current_time"), toolUse("t2", "search_docs"));

    expect(assistant(state).toolCalls.map((call) => [call.id, call.status])).toEqual([
      ["t1", "done"],
      ["t2", "running"],
    ]);
  });

  it("settles a running tool call once assistant text starts arriving", () => {
    const state = run(send, toolUse("t1", "search_docs"), {
      type: "sse",
      assistantTurnId: "a1",
      event: { type: "token", data: { text: "Depends resolves" } },
    });

    expect(assistant(state).toolCalls[0].status).toBe("done");
    expect(assistant(state).status).toBe("streaming");
  });

  it("keeps the turn out of streaming while only tools have run", () => {
    const state = run(send, toolUse("t1", "search_docs"));

    expect(assistant(state).status).toBe("awaiting");
  });
});

describe("done", () => {
  it("completes the turn and frees the composer", () => {
    const state = run(
      send,
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "Done." } } },
      { type: "sse", assistantTurnId: "a1", event: { type: "done", data: {} } },
    );

    expect(assistant(state)).toMatchObject({ status: "complete", text: "Done." });
    expect(state.status).toBe("idle");
  });

  it("settles any still-running tool call", () => {
    const state = run(
      send,
      { type: "sse", assistantTurnId: "a1", event: { type: "tool_use", data: { id: "t1", name: "x", input: {} } } },
      { type: "sse", assistantTurnId: "a1", event: { type: "done", data: {} } },
    );

    expect(assistant(state).toolCalls[0].status).toBe("done");
  });
});

describe("typed errors", () => {
  it("fails the turn with the code and message from the stream", () => {
    const state = run(send, {
      type: "sse",
      assistantTurnId: "a1",
      event: { type: "error", data: { type: "rate_limit", message: "Too many requests" } },
    });

    expect(assistant(state)).toMatchObject({
      status: "failed",
      error: { code: "rate_limit", message: "Too many requests" },
    });
    expect(state.status).toBe("idle");
  });

  it("preserves text already streamed before the error", () => {
    const state = run(
      send,
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "Partial answer" } } },
      {
        type: "sse",
        assistantTurnId: "a1",
        event: { type: "error", data: { type: "internal", message: "Boom" } },
      },
    );

    expect(assistant(state)).toMatchObject({ status: "failed", text: "Partial answer" });
  });

  it("keeps an unrecognised error code rather than discarding the failure", () => {
    const state = run(send, {
      type: "sse",
      assistantTurnId: "a1",
      event: { type: "error", data: { type: "quota_exceeded", message: "New code" } },
    });

    expect(assistant(state).error).toEqual({ code: "quota_exceeded", message: "New code" });
  });
});

describe("stream ending without done", () => {
  it("marks a dropped connection as interrupted and keeps the partial text", () => {
    const state = run(
      send,
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "Half an ans" } } },
      { type: "stream_ended", assistantTurnId: "a1" },
    );

    expect(assistant(state)).toMatchObject({ status: "interrupted", text: "Half an ans" });
    expect(state.status).toBe("idle");
  });

  it("leaves a completed turn untouched", () => {
    const state = run(
      send,
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "All of it" } } },
      { type: "sse", assistantTurnId: "a1", event: { type: "done", data: {} } },
      { type: "stream_ended", assistantTurnId: "a1" },
    );

    expect(assistant(state).status).toBe("complete");
  });

  it("leaves a failed turn showing its typed error", () => {
    const state = run(
      send,
      {
        type: "sse",
        assistantTurnId: "a1",
        event: { type: "error", data: { type: "provider_error", message: "Upstream" } },
      },
      { type: "stream_ended", assistantTurnId: "a1" },
    );

    expect(assistant(state)).toMatchObject({
      status: "failed",
      error: { code: "provider_error", message: "Upstream" },
    });
  });
});

describe("stream isolation", () => {
  it("ignores an event naming a closed turn even while another turn is open", () => {
    const state = run(
      send,
      {
        type: "sse",
        assistantTurnId: "a1",
        event: { type: "error", data: { type: "internal", message: "Boom" } },
      },
      { type: "send", text: "Second question", userTurnId: "u2", assistantTurnId: "a2" },
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "late" } } },
    );

    expect(state.turns[1]).toMatchObject({ id: "a1", status: "failed", text: "" });
    expect(state.turns[3]).toMatchObject({ id: "a2", status: "awaiting" });
    expect(state.status).toBe("busy");
  });

  it("keeps the composer locked when a stale stream ending names an already-closed turn", () => {
    const state = run(
      send,
      { type: "stopped", assistantTurnId: "a1" },
      { type: "send", text: "Second question", userTurnId: "u2", assistantTurnId: "a2" },
      { type: "stream_ended", assistantTurnId: "a1" },
    );

    expect(state.turns[1]).toMatchObject({ id: "a1", status: "stopped" });
    expect(state.turns[3]).toMatchObject({ id: "a2", status: "awaiting" });
    expect(state.status).toBe("busy");
  });
});

describe("stop", () => {
  it("marks the turn stopped and keeps what had streamed", () => {
    const state = run(
      send,
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "Enough" } } },
      { type: "stopped", assistantTurnId: "a1" },
    );

    expect(assistant(state)).toMatchObject({ status: "stopped", text: "Enough" });
    expect(state.status).toBe("idle");
  });
});

describe("retry", () => {
  it("streams a retried turn's events into that turn even when it is not the last", () => {
    const state = run(
      send,
      {
        type: "sse",
        assistantTurnId: "a1",
        event: { type: "error", data: { type: "rate_limit", message: "429" } },
      },
      { type: "send", text: "Second question", userTurnId: "u2", assistantTurnId: "a2" },
      { type: "sse", assistantTurnId: "a2", event: { type: "token", data: { text: "Second answer." } } },
      { type: "sse", assistantTurnId: "a2", event: { type: "done", data: {} } },
      { type: "retry", assistantTurnId: "a1" },
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "Recovered." } } },
      { type: "sse", assistantTurnId: "a1", event: { type: "done", data: {} } },
    );

    expect(state.turns[1]).toMatchObject({ id: "a1", status: "complete", text: "Recovered." });
    expect(state.turns[3]).toMatchObject({ id: "a2", status: "complete", text: "Second answer." });
    expect(state.status).toBe("idle");
  });

  it("resets a failed turn to awaiting and clears the error", () => {
    const state = run(
      send,
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "half" } } },
      { type: "sse", assistantTurnId: "a1", event: { type: "error", data: { type: "rate_limit", message: "429" } } },
      { type: "retry", assistantTurnId: "a1" },
    );

    expect(assistant(state)).toMatchObject({
      status: "awaiting",
      text: "",
      toolCalls: [],
      error: null,
    });
    expect(state.status).toBe("busy");
  });
});

describe("discarding an invalid request", () => {
  it("removes the rejected user turn and its assistant turn", () => {
    const state = run(
      send,
      {
        type: "sse",
        assistantTurnId: "a1",
        event: { type: "error", data: { type: "invalid_request", message: "Too long" } },
      },
      { type: "discard", assistantTurnId: "a1" },
    );

    expect(state.turns).toEqual([]);
    expect(state.status).toBe("idle");
  });

  it("stays busy when discarding while another turn is streaming", () => {
    const state = run(
      send,
      {
        type: "sse",
        assistantTurnId: "a1",
        event: { type: "error", data: { type: "invalid_request", message: "Too long" } },
      },
      { type: "send", text: "Second question", userTurnId: "u2", assistantTurnId: "a2" },
      { type: "sse", assistantTurnId: "a2", event: { type: "token", data: { text: "Second" } } },
      { type: "discard", assistantTurnId: "a1" },
    );

    expect(state.turns).toEqual([
      { kind: "user", id: "u2", text: "Second question" },
      expect.objectContaining({ id: "a2", status: "streaming", text: "Second" }),
    ]);
    expect(state.status).toBe("busy");
  });
});
