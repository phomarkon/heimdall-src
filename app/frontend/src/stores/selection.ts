"use client";

import { create } from "zustand";

type Selection =
  | { kind: "agent"; id: string }
  | { kind: "edge"; id: string }
  | { kind: "event"; id: string }
  | { kind: "focal" };

type SelectionState = {
  selected: Selection;
  setSelected: (selected: Selection) => void;
};

export const useSelectedEntityStore = create<SelectionState>((set) => ({
  selected: { kind: "focal" },
  setSelected: (selected) => set({ selected })
}));
