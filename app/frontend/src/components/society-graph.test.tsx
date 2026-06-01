import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SocietyGraph } from "@/components/society-graph";
import { getRunSnapshot } from "@/lib/api/run-adapter";
import { useSelectedEntityStore } from "@/stores/selection";
import type { RunSnapshot } from "@/types/heimdall";

vi.mock("sigma", () => ({
  default: class SigmaMock {
    on() {}
    refresh() {}
    kill() {}
    getNodeDisplayData() {
      return null;
    }
  }
}));

function renderGraph() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <div style={{ width: 900, height: 700 }}>
        <SocietyGraph snapshot={getRunSnapshot(undefined, 8)} />
      </div>
    </QueryClientProvider>
  );
}

describe("SocietyGraph selected-agent history", () => {
  it("renders a full history button on the selected agent card", () => {
    useSelectedEntityStore.setState({ selected: { kind: "focal" } });
    renderGraph();

    expect(screen.getByTestId("selected-agent-card")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Full history" })).toBeInTheDocument();
  });

  it("shows an interaction card when an edge is selected", () => {
    const edge = getRunSnapshot(undefined, 8).edges[0];
    useSelectedEntityStore.setState({ selected: { kind: "edge", id: edge.id } });
    renderGraph();

    const card = screen.getByTestId("selected-edge-card");
    expect(card).toHaveTextContent(/Society broadcast|Same-side consensus/);
    expect(card).toHaveTextContent("Strength");
    expect(screen.queryByTestId("selected-agent-card")).not.toBeInTheDocument();
  });

  it("shows an information-agent digest card and the info-consensus strip", () => {
    const base = getRunSnapshot(undefined, 8);
    const peer = base.nodes.find((node) => !node.is_focal)!;
    const infoTrace = {
      ...base.selected_trace,
      agent_id: "agent-risk-info",
      info_digest: {
        finding: "Elevated downside risk on the up side.",
        confidence: 0.8,
        importance: 0.72,
        risk_label: "high",
        uncertainty_label: "medium",
        opportunity_label: "weak",
        watch_reasons: ["activation_risk", "price_volatility"],
        direction_hint: "down",
        signals: [{ label: "Up spread", value: 12.4 }]
      }
    };
    const snapshot: RunSnapshot = {
      ...base,
      nodes: [
        ...base.nodes,
        {
          ...peer,
          id: "agent-risk-info",
          persona: { ...peer.persona, agent_id: "agent-risk-info", archetype: "risk-info", display_name: "Risk Monitor" }
        }
      ],
      agent_traces: { "agent-risk-info": infoTrace }
    };
    useSelectedEntityStore.setState({ selected: { kind: "agent", id: "agent-risk-info" } });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <div style={{ width: 900, height: 700 }}>
          <SocietyGraph snapshot={snapshot} />
        </div>
      </QueryClientProvider>
    );

    expect(screen.getByTestId("selected-info-agent-card")).toHaveTextContent("Elevated downside risk");
    expect(screen.getByText("72%")).toBeInTheDocument();
  });

  it("opens a filtered full-history drawer with rationale and tool-call details", async () => {
    useSelectedEntityStore.setState({ selected: { kind: "focal" } });
    renderGraph();

    fireEvent.click(screen.getByRole("button", { name: "Full history" }));

    const drawer = await screen.findByTestId("agent-history-drawer");
    expect(within(drawer).getByText("Full history")).toBeInTheDocument();
    expect(within(drawer).getByRole("button", { name: "Bids" })).toBeInTheDocument();
    expect(within(drawer).getByRole("button", { name: "Filled bids" })).toBeInTheDocument();
    await waitFor(() => expect(within(drawer).getAllByText(/Forecaster-agent updates/).length).toBeGreaterThan(0));
    expect(within(drawer).getAllByText(/"Forecaster-agent updates/).length).toBeGreaterThan(0);
    expect(within(drawer).getAllByText(/tool calls/i).length).toBeGreaterThan(0);

    fireEvent.click(within(drawer).getByRole("button", { name: "Filled bids" }));
    expect(within(drawer).getAllByText(/Filled/i).length).toBeGreaterThan(0);

    fireEvent.click(within(drawer).getByRole("button", { name: "Rejected" }));
    expect(within(drawer).getAllByText("rejected").length).toBeGreaterThan(0);
  });
});
