"use client";

import { useEffect } from "react";
import { create } from "zustand";
import { totalMockSteps } from "@/lib/api/run-adapter";

type PlaybackState = {
  step: number;
  totalSteps: number;
  speed: number;
  isPlaying: boolean;
  setStep: (step: number) => void;
  setTotalSteps: (totalSteps: number) => void;
  stepBy: (delta: number) => void;
  setSpeed: (speed: number) => void;
  togglePlaying: () => void;
  resetForRun: () => void;
};

export const usePlaybackStore = create<PlaybackState>((set) => ({
  step: 0,
  totalSteps: totalMockSteps,
  speed: 1,
  isPlaying: false,
  setStep: (step) =>
    set((state) => ({ step: Math.max(0, Math.min(state.totalSteps - 1, Math.round(step))) })),
  setTotalSteps: (totalSteps) =>
    set((state) => {
      const nextTotal = Math.max(1, Math.round(totalSteps));
      return { totalSteps: nextTotal, step: Math.max(0, Math.min(nextTotal - 1, state.step)) };
    }),
  stepBy: (delta) =>
    set((state) => ({
      step: Math.max(0, Math.min(state.totalSteps - 1, state.step + delta))
    })),
  setSpeed: (speed) => set({ speed }),
  togglePlaying: () => set((state) => ({ isPlaying: !state.isPlaying })),
  resetForRun: () => set({ step: 0, isPlaying: false })
}));

export function useRunPlayback() {
  const step = usePlaybackStore((state) => state.step);
  const totalSteps = usePlaybackStore((state) => state.totalSteps);
  const speed = usePlaybackStore((state) => state.speed);
  const isPlaying = usePlaybackStore((state) => state.isPlaying);
  const stepBy = usePlaybackStore((state) => state.stepBy);

  useEffect(() => {
    if (!isPlaying) {
      return;
    }

    const timer = window.setInterval(() => {
      stepBy(1);
    }, Math.max(90, 360 / speed));

    return () => window.clearInterval(timer);
  }, [isPlaying, speed, stepBy]);

  useEffect(() => {
    if (step >= totalSteps - 1 && isPlaying) {
      usePlaybackStore.setState({ isPlaying: false });
    }
  }, [isPlaying, step, totalSteps]);

  return usePlaybackStore();
}
