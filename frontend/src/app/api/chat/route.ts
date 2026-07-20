/**
 * Same-origin pass-through to the FastAPI `/chat` SSE endpoint.
 *
 * The browser talks only to this route, so the backend needs no CORS policy
 * and its URL stays server-side. Nothing here inspects or rewrites the stream —
 * the SSE body is piped straight through, keeping the client thin and the
 * backend the source of truth (Phase 1 spec, Implementation Decisions).
 */
import { NextResponse } from "next/server";

const API_URL = process.env.AUGUR_API_URL ?? "http://localhost:8000";

/** Never prerender or cache: this is a live stream. */
export const dynamic = "force-dynamic";

export async function POST(request: Request): Promise<Response> {
  let upstream: Response;

  try {
    upstream = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: await request.text(),
      signal: request.signal,
      // Stream the response rather than buffering it whole.
      cache: "no-store",
    });
  } catch {
    // The backend is unreachable (not running, wrong URL, DNS). Report it in
    // the same typed shape the stream itself uses, so the client has one
    // error path instead of two.
    return NextResponse.json(
      { type: "internal", message: "Could not reach the Augur backend." },
      { status: 502 },
    );
  }

  if (!upstream.ok || upstream.body === null) {
    return NextResponse.json(
      {
        type: upstream.status === 422 ? "invalid_request" : "internal",
        message: `The Augur backend returned ${upstream.status}.`,
      },
      { status: upstream.status },
    );
  }

  return new Response(upstream.body, {
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    },
  });
}
