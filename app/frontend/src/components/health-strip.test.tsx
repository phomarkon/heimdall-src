import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { getRunSnapshot } from "@/lib/api/run-adapter";
import { HealthStrip } from "@/components/health-strip";

describe("HealthStrip", () => {
  it("renders coverage, accept rate, P&L, interactions and wall time", () => {
    const snapshot = getRunSnapshot(undefined, 8);
    render(<HealthStrip snapshot={snapshot} />);

    expect(screen.getByText("Coverage")).toBeInTheDocument();
    expect(screen.getByText("Accept rate")).toBeInTheDocument();
    expect(screen.getByText("P&L")).toBeInTheDocument();
    expect(screen.getByText("Interactions")).toBeInTheDocument();
    expect(screen.getByText("Wall time")).toBeInTheDocument();

    // coverage + accept rate both render as percentages
    expect(screen.getAllByText(/%$/).length).toBeGreaterThanOrEqual(2);
    // interactions count equals the snapshot edge count
    expect(screen.getByText(String(snapshot.edges.length))).toBeInTheDocument();
  });
});
