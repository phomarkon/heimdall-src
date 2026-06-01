"use client";

import { CheckCircle2, Gauge, ShieldCheck, Timer, Zap } from "lucide-react";
import type { RunSnapshot } from "@/types/heimdall";
import { formatEur } from "@/lib/utils";

export function HealthStrip({ snapshot }: { snapshot: RunSnapshot }) {
  const items = [
    { label: "Coverage", value: `${(snapshot.health.coverage * 100).toFixed(1)}%`, icon: CheckCircle2 },
    { label: "Accept rate", value: `${(snapshot.health.verifier_acceptance_rate * 100).toFixed(0)}%`, icon: ShieldCheck },
    { label: "P&L", value: formatEur(snapshot.health.cumulative_pnl_eur), icon: Gauge },
    { label: "Interactions", value: `${snapshot.edges.length}`, icon: Zap },
    { label: "Wall time", value: `${snapshot.health.wall_time_minutes.toFixed(1)}m`, icon: Timer }
  ];

  return (
    <div className="pointer-events-none absolute left-4 right-4 top-4 z-20 grid max-w-[780px] grid-cols-5 border border-[#1f2933] bg-white/90 backdrop-blur max-md:left-3 max-md:right-3">
      {items.map((item, index) => {
        const Icon = item.icon;
        return (
          <div key={item.label} className={`px-3 py-2 max-md:px-2 ${index > 0 ? "border-l border-[#1f2933]" : ""}`}>
            <div className="flex items-center gap-2 text-[11px] font-black uppercase tracking-[0.12em] text-slate-500 max-md:text-[9px]">
              <Icon className="h-3.5 w-3.5 text-teal-700" aria-hidden="true" />
              {item.label}
            </div>
            <div className="mt-1 truncate text-sm font-black text-slate-900 max-md:text-xs">{item.value}</div>
          </div>
        );
      })}
    </div>
  );
}
