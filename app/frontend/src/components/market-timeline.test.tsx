import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { getRunSnapshot } from "@/lib/api/run-adapter";
import { usePlaybackStore } from "@/stores/run-playback";
import { MarketTimeline } from "@/components/market-timeline";

describe("MarketTimeline", () => {
  it("scrubs to the selected interval", () => {
    usePlaybackStore.setState({ step: 0, speed: 1, isPlaying: false });
    render(<MarketTimeline snapshot={getRunSnapshot(undefined, 0)} />);

    fireEvent.change(screen.getByTestId("timeline-slider"), { target: { value: "24" } });

    expect(usePlaybackStore.getState().step).toBe(24);
  });

  it("renders the value-tier legend and the current position counter", () => {
    usePlaybackStore.setState({ step: 12, speed: 1, isPlaying: false });
    render(<MarketTimeline snapshot={getRunSnapshot(undefined, 12)} />);

    for (const label of ["Quiet", "Attempted", "Value", "High value", "Top value"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(screen.getByText("13/96")).toBeInTheDocument();
  });

  it("colors timeline ticks by importance tier (a distribution, not one color)", () => {
    usePlaybackStore.setState({ step: 0, speed: 1, isPlaying: false });
    const { container } = render(<MarketTimeline snapshot={getRunSnapshot(undefined, 0)} />);

    const tickCells = container.querySelectorAll("[title*='priority'], [title*='Priority']");
    expect(tickCells.length).toBeGreaterThan(0);
    // mock priority signal spans more than a single tier across the day
    const titles = new Set(Array.from(tickCells).map((node) => node.getAttribute("title")?.split(" /")[0]));
    expect(titles.size).toBeGreaterThan(1);
  });

  it("changes playback speed", () => {
    usePlaybackStore.setState({ step: 0, speed: 1, isPlaying: false });
    render(<MarketTimeline snapshot={getRunSnapshot(undefined, 0)} />);
    fireEvent.click(screen.getByLabelText("Playback speed"));
    fireEvent.click(screen.getByRole("option", { name: "2x" }));
    expect(usePlaybackStore.getState().speed).toBe(2);
  });
});
