import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { fetchAgentHistory, fetchRunSnapshot, getRunSnapshot } from "@/lib/api/run-adapter";

const server = setupServer(
  http.get("http://localhost/v1/runs/:runId/society", ({ request }) => {
    const url = new URL(request.url);
    return HttpResponse.json(getRunSnapshot(String(url.pathname.split("/")[3]), Number(url.searchParams.get("step") ?? 0)));
  }),
  http.get("http://localhost/v1/runs/:runId/agents/:agentId/history", ({ params }) => {
    return HttpResponse.json({
      run_id: params.runId,
      agent_id: params.agentId,
      trace_sha256: "msw",
      total_records: 1,
      records: [
        {
          run_id: params.runId,
          step: 0,
          timestamp: "2026-04-02T12:00:00Z",
          observed_at: "2026-04-02T11:45:00Z",
          agent_id: params.agentId,
          zone: "DK1",
          archetype: "p2h",
          market_price_eur_mwh: 80,
          forecast_interval_eur_mwh: [70, 90],
          decision: { action: "bid" },
          rationale: "api rationale",
          verifier: { accepted: true, reason_codes: [], stage_failed: null },
          realized_outcome: null,
          tool_calls: [{ name: "simulate_bid", arguments: { side: "up" }, ok: true, result: { accepted: true }, error: null }]
        }
      ]
    });
  })
);

beforeAll(() => server.listen());
afterEach(() => {
  server.resetHandlers();
  vi.unstubAllEnvs();
});
afterAll(() => server.close());

describe("mockable API boundary", () => {
  it("can be backed by MSW with the same snapshot shape", async () => {
    const response = await fetch("http://localhost/v1/runs/mock/society?step=3");
    const body = await response.json();

    expect(body.step).toBe(3);
    expect(body.nodes[0].persona.archetype).toBe("p2h");
  });

  it("falls back through the adapter snapshot helper", async () => {
    const body = await fetchRunSnapshot("mock", 3);

    expect(body.step).toBe(3);
    expect(body.nodes[0].persona.archetype).toBe("p2h");
  });

  it("fetches lazy agent history through the same API boundary", async () => {
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", "http://localhost");

    const body = await fetchAgentHistory("run-api", "agent-001");

    expect(body.agent_id).toBe("agent-001");
    expect(body.records[0].rationale).toBe("api rationale");
    expect(body.records[0].tool_calls[0].result).toEqual({ accepted: true });
  });
});
