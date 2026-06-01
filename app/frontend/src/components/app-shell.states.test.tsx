import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppShell, ConfigView } from "@/components/app-shell";
import * as adapter from "@/lib/api/run-adapter";

vi.mock("next/dynamic", () => ({ default: () => () => <div data-testid="society-graph-mock" /> }));

vi.mock("@/lib/api/run-adapter", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api/run-adapter")>();
  return {
    ...actual,
    saveSocietySpec: vi.fn(),
    getPrecomputedRun: vi.fn(actual.getPrecomputedRun),
    listRunCatalog: vi.fn(actual.listRunCatalog)
  };
});

function withClient(ui: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

afterEach(() => vi.clearAllMocks());

describe("ConfigView save flow", () => {
  it("shows a success banner when the spec is saved", async () => {
    vi.mocked(adapter.saveSocietySpec).mockResolvedValue({ status: "saved" });
    render(withClient(<ConfigView snapshot={adapter.getRunSnapshot(undefined, 0)} />));

    fireEvent.click(screen.getByRole("button", { name: /Save society spec/ }));

    expect(await screen.findByText(/Society spec saved/i)).toBeInTheDocument();
    expect(adapter.saveSocietySpec).toHaveBeenCalledWith(expect.objectContaining({ society_id: expect.any(String) }));
  });

  it("shows an unavailable banner when the store is not configured", async () => {
    vi.mocked(adapter.saveSocietySpec).mockResolvedValue({ status: "unavailable", detail: "No run-view API configured." });
    render(withClient(<ConfigView snapshot={adapter.getRunSnapshot(undefined, 0)} />));

    fireEvent.click(screen.getByRole("button", { name: /Save society spec/ }));
    expect(await screen.findByText(/No run-view API configured/i)).toBeInTheDocument();
  });

  it("shows an error banner when the save fails", async () => {
    vi.mocked(adapter.saveSocietySpec).mockResolvedValue({ status: "error", detail: "Save failed (500)." });
    render(withClient(<ConfigView snapshot={adapter.getRunSnapshot(undefined, 0)} />));

    fireEvent.click(screen.getByRole("button", { name: /Save society spec/ }));
    expect(await screen.findByText(/Save failed \(500\)/i)).toBeInTheDocument();
  });
});

describe("AppShell empty / error states", () => {
  it("renders an explicit empty state when the run has no snapshots", async () => {
    vi.mocked(adapter.getPrecomputedRun).mockResolvedValue({
      run_id: "empty-run",
      total_steps: 0,
      snapshots: [],
      market_series: []
    });
    vi.mocked(adapter.listRunCatalog).mockResolvedValue({ runs: [], usingFallbackCatalog: false });

    render(withClient(<AppShell />));

    expect(await screen.findByText(/Run has no snapshots/i)).toBeInTheDocument();
  });
});
