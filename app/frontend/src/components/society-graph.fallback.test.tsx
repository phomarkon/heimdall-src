import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SocietyGraph } from "@/components/society-graph";
import { getRunSnapshot } from "@/lib/api/run-adapter";
import { useSelectedEntityStore } from "@/stores/selection";

// Force the WebGL renderer to be unavailable so the DOM fallback graph renders.
vi.mock("sigma", () => ({
  default: class ThrowingSigma {
    constructor() {
      throw new Error("WebGL unavailable");
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

describe("SocietyGraph DOM fallback (no WebGL)", () => {
  it("renders the DOM fallback with one selectable button per agent", () => {
    useSelectedEntityStore.setState({ selected: { kind: "focal" } });
    renderGraph();

    expect(screen.getByText(/DOM graph fallback/i)).toBeInTheDocument();
    const snapshot = getRunSnapshot(undefined, 8);
    const focal = snapshot.nodes.find((node) => node.is_focal)!;
    expect(screen.getByRole("button", { name: focal.persona.display_name })).toBeInTheDocument();
  });

  it("selects a peer agent from the fallback graph", () => {
    useSelectedEntityStore.setState({ selected: { kind: "focal" } });
    renderGraph();

    const snapshot = getRunSnapshot(undefined, 8);
    const peer = snapshot.nodes.find((node) => !node.is_focal)!;
    fireEvent.click(screen.getByRole("button", { name: peer.persona.display_name }));
    expect(useSelectedEntityStore.getState().selected).toEqual({ kind: "agent", id: peer.id });
  });
});
