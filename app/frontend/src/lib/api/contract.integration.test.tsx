/**
 * Integration test for the run-view <-> frontend seam.
 *
 * Unlike the component tests (which mock data) this consumes a REAL payload produced by
 * the run-view backend (apps/run-view/tests/test_frontend_contract.py, regenerate with
 * HEIMDALL_WRITE_CONTRACT_FIXTURES=1) and drives it through the actual adapter + the real
 * components, proving the backend output parses and renders end to end. This is the layer
 * that catches FE<->BE contract drift (edges, focal_baselines, priority grounding, ...).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import type { ReactNode } from "react";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import precomputedFixture from "@/test-fixtures/precomputed.contract.json";
import historyFixture from "@/test-fixtures/agent-history.contract.json";
import { fetchAgentHistory, getPrecomputedRun } from "@/lib/api/run-adapter";
import { ActivitySidebar } from "@/components/activity-sidebar";
import { MarketTimeline } from "@/components/market-timeline";
import { ResultsView } from "@/components/app-shell";
import { SocietyGraph } from "@/components/society-graph";
import { usePlaybackStore } from "@/stores/run-playback";
import { useSelectedEntityStore } from "@/stores/selection";
import type { AgentHistoryResponse, PrecomputedRun } from "@/types/heimdall";

// WebGL is unavailable in jsdom; assert the backend graph via the DOM fallback.
vi.mock("sigma", () => ({
  default: class {
    constructor() {
      throw new Error("no webgl");
    }
  }
}));

const API = "http://localhost";
const precomputed = precomputedFixture as unknown as PrecomputedRun;
const history = historyFixture as unknown as AgentHistoryResponse;

const server = setupServer(
  http.get(`${API}/v1/runs`, () =>
    HttpResponse.json({
      runs: [
        { run_id: precomputed.run_id, total_steps: precomputed.total_steps, trace_sha256: "x", status: "completed", trace_path: "x" }
      ]
    })
  ),
  http.get(`${API}/v1/runs/:runId/precomputed`, () => HttpResponse.json(precomputed)),
  http.get(`${API}/v1/runs/:runId/agents/:agentId/history`, () => HttpResponse.json(history))
);

beforeAll(() => server.listen({ onUnhandledRequest: "bypass" }));
afterEach(() => {
  server.resetHandlers();
  vi.unstubAllEnvs();
});
afterAll(() => server.close());

function withClient(ui: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("run-view -> frontend contract (real backend payload)", () => {
  it("the adapter parses the real backend precomputed payload", async () => {
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    const run = await getPrecomputedRun(precomputed.run_id);

    expect(run.run_id).toBe(precomputed.run_id);
    expect(run.snapshots.length).toBeGreaterThan(0);
    const kinds = new Set(run.snapshots[0].edges.map((edge) => edge.kind));
    expect(kinds).toContain("consensus");
    expect(kinds).toContain("broadcast");
    expect(run.focal_baselines?.length).toBeGreaterThan(0);
    expect(run.snapshots[0].market.priority_signal?.grounding).toBe("realized_outcome");
  });

  it("the adapter parses the real backend agent-history payload", async () => {
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    const result = await fetchAgentHistory(precomputed.run_id, "agent-000");
    expect(result.records.length).toBeGreaterThan(0);
    expect(result.records[0]).toHaveProperty("decision");
    expect(Array.isArray(result.records[0].tool_calls)).toBe(true);
  });

  it("renders the backend payload through the real data path", () => {
    const snapshot = precomputed.snapshots[0];
    usePlaybackStore.setState({ step: 0, totalSteps: precomputed.total_steps, speed: 1, isPlaying: false });
    useSelectedEntityStore.setState({ selected: { kind: "focal" } });

    // Timeline: value-tier legend driven by backend priority_signal
    const timeline = render(<MarketTimeline snapshot={snapshot} run={precomputed} />);
    expect(timeline.getByText("Top value")).toBeInTheDocument();
    timeline.unmount();

    // Results: data-driven baseline leaderboard from backend focal_baselines
    const results = render(<ResultsView run={precomputed} snapshot={snapshot} />);
    expect(results.getByText("Focal policy baseline leaderboard")).toBeInTheDocument();
    expect(results.getByText(/Profit-guard baseline/)).toBeInTheDocument();
    results.unmount();

    // Activity rail: derived from backend traces + edges
    const activity = render(<ActivitySidebar run={precomputed} snapshot={snapshot} />);
    expect(activity.getByTestId("activity-feed")).toBeInTheDocument();
    activity.unmount();

    // Graph: backend nodes + edges render via the DOM fallback (no WebGL)
    const graph = render(
      withClient(
        <div style={{ width: 900, height: 700 }}>
          <SocietyGraph snapshot={snapshot} />
        </div>
      )
    );
    expect(graph.getByText(/DOM graph fallback/i)).toBeInTheDocument();
    expect(graph.container.querySelectorAll("svg line").length).toBe(snapshot.edges.length);
    expect(graph.getAllByRole("button").length).toBeGreaterThanOrEqual(snapshot.nodes.length);
  });
});
