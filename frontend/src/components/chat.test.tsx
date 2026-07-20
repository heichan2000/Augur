import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Chat } from "./chat";

/** Build an SSE response body from raw frame strings. */
function sseResponse(...frames: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const frame of frames) controller.enqueue(encoder.encode(frame));
      controller.close();
    },
  });

  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

const token = (text: string) => `event: token\ndata: ${JSON.stringify({ text })}\n\n`;
const done = "event: done\ndata: {}\n\n";

function mockFetchOnce(response: Response) {
  const fetchMock = vi.fn().mockResolvedValue(response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  sessionStorage.clear();
  // scrollIntoView is not implemented in the test DOM.
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("first run", () => {
  it("offers example questions before any turn exists", () => {
    render(<Chat />);

    expect(screen.getByRole("heading", { name: "Ask about FastAPI" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "How do I add dependency injection to a route?" }),
    ).toBeInTheDocument();
  });
});

describe("sending a message", () => {
  it("streams the reply into the thread and frees the composer", async () => {
    mockFetchOnce(sseResponse(token("Use "), token("Depends."), done));

    render(<Chat />);
    await userEvent.click(
      screen.getByRole("button", { name: "How do I add dependency injection to a route?" }),
    );

    await waitFor(() => expect(screen.getByText(/Use Depends\./)).toBeInTheDocument());

    // The question is echoed as a user turn.
    expect(
      screen.getByText("How do I add dependency injection to a route?"),
    ).toBeInTheDocument();
    // Send is back, so the turn has ended.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument(),
    );
  });

  it("posts the message with the stored session id", async () => {
    const fetchMock = mockFetchOnce(sseResponse(token("Hi"), done));

    render(<Chat />);
    await userEvent.type(screen.getByRole("textbox"), "What time is it?");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/chat");
    expect(JSON.parse(init.body as string)).toEqual({
      session_id: sessionStorage.getItem("augur-session-id"),
      message: "What time is it?",
    });
  });

  it("reuses one session id across turns so history accumulates server-side", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(sseResponse(token("ok"), done))),
    );

    render(<Chat />);
    const textbox = screen.getByRole("textbox");

    await userEvent.type(textbox, "first");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument(),
    );

    await userEvent.type(textbox, "second");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    const fetchMock = vi.mocked(fetch);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    const sessions = fetchMock.mock.calls.map(
      ([, init]) => JSON.parse((init as RequestInit).body as string).session_id,
    );
    expect(sessions[0]).toBe(sessions[1]);
  });
});

describe("surfacing tool activity", () => {
  it("shows the tool the assistant called", async () => {
    mockFetchOnce(
      sseResponse(
        `event: tool_use\ndata: {"id":"t1","name":"get_current_time","input":{}}\n\n`,
        token("It is 3pm."),
        done,
      ),
    );

    render(<Chat />);
    await userEvent.click(screen.getByRole("button", { name: "What time is it right now?" }));

    await waitFor(() =>
      expect(screen.getByText("Checked the current time")).toBeInTheDocument(),
    );
    expect(screen.getByText("get_current_time")).toBeInTheDocument();
  });
});

describe("failure handling", () => {
  it("shows a typed error instead of freezing", async () => {
    mockFetchOnce(
      sseResponse(
        `event: error\ndata: {"type":"rate_limit","message":"Too many requests"}\n\n`,
      ),
    );

    render(<Chat />);
    await userEvent.click(screen.getByRole("button", { name: "What time is it right now?" }));

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(screen.getByText("rate_limit")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("marks a stream that ends without done as incomplete", async () => {
    mockFetchOnce(sseResponse(token("Half an ans")));

    render(<Chat />);
    await userEvent.click(screen.getByRole("button", { name: "What time is it right now?" }));

    await waitFor(() =>
      expect(screen.getByText("Connection lost mid-answer")).toBeInTheDocument(),
    );
    expect(screen.getByText(/Half an ans/)).toBeInTheDocument();
  });

  it("reports an unreachable backend as an error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    render(<Chat />);
    await userEvent.click(screen.getByRole("button", { name: "What time is it right now?" }));

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(screen.getByText(/Could not reach the server/)).toBeInTheDocument();
  });

  it("restores a rejected message to the composer for editing", async () => {
    mockFetchOnce(
      sseResponse(
        `event: error\ndata: {"type":"invalid_request","message":"Message too long"}\n\n`,
      ),
    );

    render(<Chat />);
    await userEvent.type(screen.getByRole("textbox"), "a rejected question");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Edit message" })).toBeInTheDocument(),
    );
    await userEvent.click(screen.getByRole("button", { name: "Edit message" }));

    expect(screen.getByRole("textbox")).toHaveValue("a rejected question");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("re-sends the same question on Retry", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse(`event: error\ndata: {"type":"internal","message":"Boom"}\n\n`),
      )
      .mockResolvedValueOnce(sseResponse(token("Second time lucky."), done));
    vi.stubGlobal("fetch", fetchMock);

    render(<Chat />);
    await userEvent.click(screen.getByRole("button", { name: "What time is it right now?" }));

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));

    await waitFor(() =>
      expect(screen.getByText(/Second time lucky\./)).toBeInTheDocument(),
    );

    const messages = fetchMock.mock.calls.map(
      ([, init]) => JSON.parse((init as RequestInit).body as string).message,
    );
    expect(messages).toEqual(["What time is it right now?", "What time is it right now?"]);
  });

  it("streams a retried answer into the failed turn even after later turns completed", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse(`event: error\ndata: {"type":"internal","message":"Boom"}\n\n`),
      )
      .mockResolvedValueOnce(sseResponse(token("Second answer."), done))
      .mockResolvedValueOnce(sseResponse(token("Recovered answer."), done));
    vi.stubGlobal("fetch", fetchMock);

    render(<Chat />);

    // Turn 1 fails.
    await userEvent.click(screen.getByRole("button", { name: "What time is it right now?" }));
    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());

    // Turn 2 completes normally.
    await userEvent.type(screen.getByRole("textbox"), "second question");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(screen.getByText(/Second answer\./)).toBeInTheDocument());

    // Retrying turn 1 streams the new answer into turn 1's slot…
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() =>
      expect(screen.getByText(/Recovered answer\./)).toBeInTheDocument(),
    );
    const thread = document.body.textContent ?? "";
    expect(thread.indexOf("Recovered answer.")).toBeLessThan(thread.indexOf("Second answer."));

    // …and nothing is left thinking or locked.
    expect(screen.queryByText("Thinking…")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument(),
    );

    const messages = fetchMock.mock.calls.map(
      ([, init]) => JSON.parse((init as RequestInit).body as string).message,
    );
    expect(messages).toEqual([
      "What time is it right now?",
      "second question",
      "What time is it right now?",
    ]);
  });

  it("keeps Retry inert while another turn is streaming", async () => {
    // A stream that never ends keeps the second turn in flight.
    const hangingResponse = new Response(new ReadableStream<Uint8Array>({ start() {} }), {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse(`event: error\ndata: {"type":"internal","message":"Boom"}\n\n`),
      )
      .mockResolvedValueOnce(hangingResponse);
    vi.stubGlobal("fetch", fetchMock);

    render(<Chat />);

    // Turn 1 fails, offering Retry.
    await userEvent.click(screen.getByRole("button", { name: "What time is it right now?" }));
    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());

    // Turn 2 is mid-stream: the composer shows Stop.
    await userEvent.type(screen.getByRole("textbox"), "second question");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument());

    // Turn 1's Retry is disabled and cannot start a second stream.
    const retry = screen.getByRole("button", { name: "Retry" });
    expect(retry).toBeDisabled();
    await userEvent.click(retry);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("keeps Edit message inert while another turn is streaming", async () => {
    const hangingResponse = new Response(new ReadableStream<Uint8Array>({ start() {} }), {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse(`event: error\ndata: {"type":"invalid_request","message":"Too long"}\n\n`),
      )
      .mockResolvedValueOnce(hangingResponse);
    vi.stubGlobal("fetch", fetchMock);

    render(<Chat />);

    // Turn 1 is rejected, offering Edit message.
    await userEvent.type(screen.getByRole("textbox"), "a rejected question");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Edit message" })).toBeInTheDocument(),
    );

    // Turn 2 is mid-stream: the composer shows Stop.
    await userEvent.type(screen.getByRole("textbox"), "second question");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument());

    // Edit message is disabled: the rejected turn stays put, the composer
    // stays locked on Stop, and no second stream starts.
    const edit = screen.getByRole("button", { name: "Edit message" });
    expect(edit).toBeDisabled();
    await userEvent.click(edit);

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
    expect(screen.getByRole("textbox")).toHaveValue("");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("stopping a turn", () => {
  it("marks the streaming turn stopped and frees the composer", async () => {
    // Emit part of an answer, then hang — the turn stays streaming until Stop.
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(token("Partial ans")));
      },
    });
    mockFetchOnce(
      new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
    );

    render(<Chat />);
    await userEvent.click(screen.getByRole("button", { name: "What time is it right now?" }));
    await waitFor(() => expect(screen.getByText(/Partial ans/)).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "Stop" }));

    expect(screen.getByText(/stopped · not saved/)).toBeInTheDocument();
    expect(screen.getByText(/Partial ans/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send" })).toBeInTheDocument();
  });
});
