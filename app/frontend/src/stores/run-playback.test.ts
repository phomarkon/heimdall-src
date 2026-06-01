import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePlaybackStore, useRunPlayback } from "@/stores/run-playback";

describe("playback store", () => {
  beforeEach(() => {
    usePlaybackStore.setState({ step: 0, totalSteps: 96, speed: 1, isPlaying: false });
  });

  it("clamps setStep within [0, totalSteps-1] and rounds", () => {
    const { setStep } = usePlaybackStore.getState();
    act(() => setStep(-5));
    expect(usePlaybackStore.getState().step).toBe(0);
    act(() => setStep(1000));
    expect(usePlaybackStore.getState().step).toBe(95);
    act(() => setStep(12.6));
    expect(usePlaybackStore.getState().step).toBe(13);
  });

  it("stepBy clamps at both ends", () => {
    act(() => usePlaybackStore.getState().setStep(95));
    act(() => usePlaybackStore.getState().stepBy(1));
    expect(usePlaybackStore.getState().step).toBe(95);
    act(() => usePlaybackStore.getState().setStep(0));
    act(() => usePlaybackStore.getState().stepBy(-1));
    expect(usePlaybackStore.getState().step).toBe(0);
  });

  it("setTotalSteps re-clamps the current step and enforces a minimum of 1", () => {
    act(() => usePlaybackStore.getState().setStep(50));
    act(() => usePlaybackStore.getState().setTotalSteps(24));
    expect(usePlaybackStore.getState().totalSteps).toBe(24);
    expect(usePlaybackStore.getState().step).toBe(23);
    act(() => usePlaybackStore.getState().setTotalSteps(0));
    expect(usePlaybackStore.getState().totalSteps).toBe(1);
    expect(usePlaybackStore.getState().step).toBe(0);
  });

  it("togglePlaying and resetForRun behave", () => {
    act(() => usePlaybackStore.getState().togglePlaying());
    expect(usePlaybackStore.getState().isPlaying).toBe(true);
    act(() => usePlaybackStore.getState().setStep(40));
    act(() => usePlaybackStore.getState().resetForRun());
    expect(usePlaybackStore.getState()).toMatchObject({ step: 0, isPlaying: false });
  });
});

describe("useRunPlayback timer", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    usePlaybackStore.setState({ step: 0, totalSteps: 96, speed: 1, isPlaying: false });
  });
  afterEach(() => vi.useRealTimers());

  it("advances the step on an interval while playing and stops at the end", () => {
    renderHook(() => useRunPlayback());
    act(() => usePlaybackStore.getState().togglePlaying());
    act(() => vi.advanceTimersByTime(400));
    expect(usePlaybackStore.getState().step).toBeGreaterThan(0);

    act(() => {
      usePlaybackStore.setState({ step: 95, isPlaying: true });
      vi.advanceTimersByTime(400);
    });
    expect(usePlaybackStore.getState().isPlaying).toBe(false);
  });
});
