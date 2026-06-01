import { beforeEach, describe, expect, it } from "vitest";
import { useSelectedEntityStore } from "@/stores/selection";

describe("selection store", () => {
  beforeEach(() => useSelectedEntityStore.setState({ selected: { kind: "focal" } }));

  it("defaults to the focal selection", () => {
    expect(useSelectedEntityStore.getState().selected).toEqual({ kind: "focal" });
  });

  it("updates selection to agent / edge / event", () => {
    const { setSelected } = useSelectedEntityStore.getState();
    setSelected({ kind: "agent", id: "agent-003" });
    expect(useSelectedEntityStore.getState().selected).toEqual({ kind: "agent", id: "agent-003" });
    setSelected({ kind: "edge", id: "consensus-0-up-a-b" });
    expect(useSelectedEntityStore.getState().selected).toEqual({ kind: "edge", id: "consensus-0-up-a-b" });
    setSelected({ kind: "event", id: "spike-1" });
    expect(useSelectedEntityStore.getState().selected).toEqual({ kind: "event", id: "spike-1" });
  });
});
