"use client";

import { BarChart3, CircleHelp, Database, Play, Settings2 } from "lucide-react";
import type { RunSnapshot } from "@/types/heimdall";
import { cn, formatTime } from "@/lib/utils";

export type DashboardView = "live" | "runs" | "config" | "results" | "help";

const stages: Array<{ id: DashboardView; label: string; icon: typeof Play }> = [
  { id: "live", label: "Live run", icon: Play },
  { id: "runs", label: "Runs", icon: Database },
  { id: "config", label: "Config", icon: Settings2 },
  { id: "results", label: "Results", icon: BarChart3 },
  { id: "help", label: "Help", icon: CircleHelp }
];

export function RunProgressRail({
  snapshot,
  activeView,
  onViewChange
}: {
  snapshot: RunSnapshot;
  activeView: DashboardView;
  onViewChange: (view: DashboardView) => void;
}) {
  return (
    <nav aria-label="Run workflow" className="col-start-1 row-start-2 row-end-4 flex flex-col items-center gap-4 border-r-2 border-[#1f2933] bg-[#fbfaf7] py-4">
      <ol className="flex flex-1 flex-col items-center gap-3">
        {stages.map((stage) => {
          const Icon = stage.icon;
          const active = stage.id === activeView;
          return (
            <li key={stage.id}>
              <button
                className={cn(
                  "group flex h-11 w-11 items-center justify-center border border-[#1f2933] transition",
                  active && "bg-[#dff5ec] text-teal-700",
                  !active && "bg-white text-slate-500 hover:bg-[#eef3ff] hover:text-teal-700"
                )}
                aria-label={`${stage.label} view${active ? " active" : ""}`}
                title={stage.label}
                type="button"
                onClick={() => onViewChange(stage.id)}
              >
                <Icon className="h-4 w-4" aria-hidden="true" />
              </button>
            </li>
          );
        })}
      </ol>
      <div className="text-center text-[10px] uppercase tracking-[0.18em] text-slate-500">
        <div>{formatTime(snapshot.market.timestamp)}</div>
        <div>UTC</div>
      </div>
    </nav>
  );
}
