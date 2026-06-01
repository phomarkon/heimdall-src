import { describe, expect, it } from "vitest";
import { getAgentTrace, getRunSnapshot, totalMockSteps } from "@/lib/api/run-adapter";

describe("mock Heimdall run", () => {
  it("generates the planned 50-agent, 96-tick fixture", () => {
    const snapshot = getRunSnapshot(undefined, 0);

    expect(snapshot.nodes).toHaveLength(50);
    expect(snapshot.total_steps).toBe(96);
    expect(totalMockSteps).toBe(96);
    expect(snapshot.nodes.filter((node) => node.is_focal)).toHaveLength(1);
  });

  it("keeps timeline scrubbing bounded", () => {
    expect(getRunSnapshot(undefined, -10).step).toBe(0);
    expect(getRunSnapshot(undefined, 1_000).step).toBe(95);
  });

  it("produces focal verifier pass and fail examples", () => {
    const accepted = getAgentTrace(undefined, "agent-p2h-focal", 4);
    const rejected = getAgentTrace(undefined, "agent-p2h-focal", 8);

    expect(accepted.verifier_verdict.accepted).toBe(true);
    expect(rejected.verifier_verdict.accepted).toBe(false);
    expect(rejected.verifier_verdict.retry_suggestion).toContain("Reduce quantity");
  });

  it("changes society edges over time and exposes consensus/broadcast kinds", () => {
    const early = getRunSnapshot(undefined, 2);
    const later = getRunSnapshot(undefined, 40).edges.map((edge) => edge.id);

    expect(early.edges.length).toBeGreaterThan(0);
    expect(early.edges.map((edge) => edge.id)).not.toEqual(later);
    expect(new Set(early.edges.map((edge) => edge.kind))).toContain("broadcast");
  });
});
