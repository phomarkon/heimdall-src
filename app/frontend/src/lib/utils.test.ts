import { describe, expect, it } from "vitest";
import { cn, formatDateTime, formatEur, formatMw, formatPrice, formatTime } from "@/lib/utils";

describe("formatting helpers", () => {
  it("merges class names and dedupes tailwind conflicts", () => {
    expect(cn("p-2", "p-4")).toBe("p-4");
    expect(cn("text-sm", false && "hidden", "font-bold")).toBe("text-sm font-bold");
  });

  it("formats MW with one decimal", () => {
    expect(formatMw(12.34)).toBe("12.3 MW");
    expect(formatMw(-5)).toBe("-5.0 MW");
  });

  it("formats EUR as whole-euro currency", () => {
    expect(formatEur(8505.89)).toBe("€8,506");
    expect(formatEur(-138)).toBe("-€138");
    expect(formatEur(0)).toBe("€0");
  });

  it("formats price per MWh", () => {
    expect(formatPrice(67.13)).toBe("67.1 €/MWh");
  });

  it("formats time and datetime in UTC from string or Date", () => {
    const iso = "2026-04-02T05:30:00Z";
    expect(formatTime(iso)).toBe("05:30");
    expect(formatTime(new Date(iso))).toBe("05:30");
    expect(formatDateTime(iso)).toContain("05:30");
    expect(formatDateTime(iso)).toMatch(/02 Apr/);
  });
});
