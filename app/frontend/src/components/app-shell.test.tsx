import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { ConfigView, HelpView, MarketTicker, ResultsView, RunsView } from "@/components/app-shell";
import { RunProgressRail, type DashboardView } from "@/components/run-progress-rail";
import { getPrecomputedRun, getRunSnapshot, listRuns } from "@/lib/api/run-adapter";

vi.mock("next/dynamic", () => ({
  default: () => () => <div data-testid="society-graph-mock" />
}));

function withClient(ui: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe("App shell navigation and ticker", () => {
  it("shows a help rail button and switches to the help view", () => {
    const onViewChange = vi.fn<(view: DashboardView) => void>();
    render(<RunProgressRail snapshot={getRunSnapshot(undefined, 0)} activeView="live" onViewChange={onViewChange} />);

    const helpButton = screen.getByRole("button", { name: /help view/i });
    fireEvent.click(helpButton);

    expect(onViewChange).toHaveBeenCalledWith("help");
  });

  it("shows a runs rail button and switches to the runs view", () => {
    const onViewChange = vi.fn<(view: DashboardView) => void>();
    render(<RunProgressRail snapshot={getRunSnapshot(undefined, 0)} activeView="live" onViewChange={onViewChange} />);

    fireEvent.click(screen.getByRole("button", { name: /runs view/i }));

    expect(onViewChange).toHaveBeenCalledWith("runs");
  });

  it("renders the help guide content", () => {
    render(<HelpView />);

    expect(screen.getByText("How to read Heimdall")).toBeInTheDocument();
    expect(screen.getByText(/verifier-guarded LLM society/i)).toBeInTheDocument();
    expect(screen.getByText("Demo walkthrough")).toBeInTheDocument();
  });

  it("renders stable market ticker values for high prices", () => {
    const snapshot = structuredClone(getRunSnapshot(undefined, 0));
    snapshot.market.mfrr_price_eur_per_mwh = 1000;
    snapshot.health.tick_pnl_eur = 300;
    render(<MarketTicker snapshot={snapshot} />);

    expect(screen.getByTestId("market-ticker")).toHaveAccessibleName("Market status ticker");
    expect(screen.getByText("1000.0 €/MWh")).toBeInTheDocument();
    expect(screen.getByText("€300")).toBeInTheDocument();
  });

  it("renders the ML forecaster leaderboard and selected-tick diagnostics", async () => {
    const run = await getPrecomputedRun();
    const snapshot = run.snapshots[4];

    render(<ResultsView run={run} snapshot={snapshot} />);

    expect(screen.getByText("ML forecaster leaderboard")).toBeInTheDocument();
    expect(screen.getByText("f8")).toBeInTheDocument();
    expect(screen.getByText("Selected-tick forecast diagnostics")).toBeInTheDocument();
    expect(screen.getByText("Focal policy baseline leaderboard")).toBeInTheDocument();
  });

  it("groups runs by setup and selects a run window", async () => {
    const runs = await listRuns();
    const onSelectRun = vi.fn<(runId: string) => void>();

    render(<RunsView runs={runs} selectedRunId="mixed20-apr02-96-real-controls" onSelectRun={onSelectRun} />);

    expect(screen.getAllByText("Mixed-20 full days").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Apr 02 / 96 ticks / proxy controls").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByText("mixed20-apr02-96-proxy-controls"));

    expect(onSelectRun).toHaveBeenCalledWith("mixed20-apr02-96-proxy-controls");
  });

  it("renders top setups and combines run filters", async () => {
    const runs = await listRuns();
    const onSelectRun = vi.fn<(runId: string) => void>();

    render(<RunsView runs={runs} selectedRunId="mixed20-apr02-96-real-controls" onSelectRun={onSelectRun} />);

    expect(screen.getByText("Top setups")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Full day" }));
    fireEvent.click(screen.getByRole("button", { name: "Proxy controls" }));

    expect(screen.getAllByText("Apr 02 / 96 ticks / proxy controls").length).toBeGreaterThan(0);
    expect(screen.queryByText("Apr 02 / 96 ticks / real controls")).not.toBeInTheDocument();
  });

  it("shows an explicit fallback catalog banner", async () => {
    const runs = await listRuns();

    render(
      <RunsView
        runs={runs}
        selectedRunId="mixed20-apr02-96-real-controls"
        usingFallbackCatalog
        onSelectRun={vi.fn()}
      />
    );

    expect(screen.getByText(/fallback catalog data/i)).toBeInTheDocument();
  });

  it("lets the simulation config update agent count, model, time range, and agent types", () => {
    const snapshot = getRunSnapshot(undefined, 0);
    render(withClient(<ConfigView snapshot={snapshot} />));

    fireEvent.change(screen.getByLabelText("Agent amount"), { target: { value: "84" } });
    fireEvent.click(screen.getByLabelText("Society model"));
    fireEvent.click(screen.getByRole("option", { name: "Qwen 72B" }));
    fireEvent.change(screen.getByLabelText("Start time"), { target: { value: "2025-10-01T19:15" } });
    fireEvent.change(screen.getByLabelText("Simulation intervals"), { target: { value: "12" } });
    fireEvent.click(screen.getByLabelText("Wind BRPs"));

    expect(screen.getByDisplayValue("84")).toBeInTheDocument();
    expect(screen.getByLabelText("Society model")).toHaveTextContent("Qwen 72B");
    expect(screen.getByDisplayValue("2025-10-01T19:15")).toBeInTheDocument();
    expect(screen.getByDisplayValue("12")).toBeInTheDocument();
    expect(screen.getByText("01/10/2025, 22:00")).toBeInTheDocument();
    expect(screen.getByText("84 agents / Qwen 72B / 12 intervals")).toBeInTheDocument();
    expect(screen.getByText("84 / 84 agents")).toBeInTheDocument();
  });
});
