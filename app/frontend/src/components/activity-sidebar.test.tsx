import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { getPrecomputedRun } from "@/lib/api/run-adapter";
import { ActivitySidebar, buildActivityFeed } from "@/components/activity-sidebar";

describe("ActivitySidebar", () => {
  it("derives tool calls, verifier decisions, and interactions from the precomputed run", async () => {
    const run = await getPrecomputedRun();
    const feed = buildActivityFeed(run, 8);

    expect(feed.some((item) => item.kind === "tool" && item.title.includes("Focal P2H"))).toBe(true);
    expect(feed.some((item) => item.kind === "verifier" && item.title.includes("blocked"))).toBe(true);
    expect(feed.some((item) => item.kind === "interaction" && item.actor.includes("->"))).toBe(true);
  });

  it("renders a full-height scrolling progress feed", async () => {
    const run = await getPrecomputedRun();
    render(<ActivitySidebar run={run} snapshot={run.snapshots[8]} />);

    const feed = screen.getByTestId("activity-feed");
    expect(feed).toHaveAccessibleName("Recent agent progress");
    expect(within(feed).getAllByText(/Tool called|Forecaster-agent/).length).toBeGreaterThan(0);
    expect(within(feed).getByText("Verifier blocked focal bid")).toBeInTheDocument();
    expect(screen.queryByTestId("verifier-ledger")).not.toBeInTheDocument();
  });
});
