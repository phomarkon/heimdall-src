"use client";

import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { RetroSelect } from "@/components/app-shell";
import { getMarketSeries } from "@/lib/api/run-adapter";
import { cn, formatDateTime, formatPrice, formatTime } from "@/lib/utils";
import { usePlaybackStore } from "@/stores/run-playback";
import type { PrecomputedRun, RunSnapshot } from "@/types/heimdall";

export function MarketTimeline({ snapshot, run }: { snapshot: RunSnapshot; run?: PrecomputedRun }) {
  const { step, speed, isPlaying, setStep, stepBy, setSpeed, togglePlaying } = usePlaybackStore();
  const series = getMarketSeries(run);
  const start = series[0];
  const end = series[series.length - 1];
  const isCompactTimeline = series.length > 24 && series.length <= 48;
  const isDenseTimeline = series.length > 48;

  return (
    <div className="grid h-full grid-cols-[260px_minmax(0,1fr)_260px] items-center gap-5 px-4 py-3 max-lg:grid-cols-[220px_minmax(0,1fr)] max-md:grid-cols-1 max-md:gap-2 max-md:py-3">
      <div className="flex h-full flex-col justify-center gap-3">
        <div className="flex items-center gap-2">
          <IconButton label="Previous interval" onClick={() => stepBy(-1)}>
            <SkipBack className="h-5 w-5" aria-hidden="true" />
          </IconButton>
          <IconButton label={isPlaying ? "Pause replay" : "Play replay"} onClick={togglePlaying}>
            {isPlaying ? <Pause className="h-5 w-5" aria-hidden="true" /> : <Play className="h-5 w-5" aria-hidden="true" />}
          </IconButton>
          <IconButton label="Next interval" onClick={() => stepBy(1)}>
            <SkipForward className="h-5 w-5" aria-hidden="true" />
          </IconButton>
          <RetroSelect
            label="Playback speed"
            className="h-11 w-24 text-sm"
            options={[
              { value: 0.5, label: "0.5x" },
              { value: 1, label: "1x" },
              { value: 2, label: "2x" },
              { value: 4, label: "4x" }
            ]}
            value={speed}
            onChange={(value) => setSpeed(Number(value))}
            placement="top"
          />
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs font-black uppercase tracking-[0.06em] text-slate-600">
          <span className="flex items-center gap-2 whitespace-nowrap">
            <span className="h-4 w-4 shrink-0 border border-[#1f2933] bg-[#27b76f]" />
            Quiet
          </span>
          <span className="flex items-center gap-2 whitespace-nowrap">
            <span className="h-4 w-4 shrink-0 border border-[#1f2933] bg-[#fde68a]" />
            Attempted
          </span>
          <span className="flex items-center gap-2 whitespace-nowrap">
            <span className="h-4 w-4 shrink-0 border border-[#1f2933] bg-[#facc15]" />
            Value
          </span>
          <span className="flex items-center gap-2 whitespace-nowrap">
            <span className="h-4 w-4 shrink-0 border border-[#1f2933] bg-[#f97316]" />
            High value
          </span>
          <span className="flex items-center gap-2 whitespace-nowrap">
            <span className="h-4 w-4 shrink-0 border border-[#1f2933] bg-[#dc2626]" />
            Top value
          </span>
        </div>
      </div>

      <div className="min-w-0 self-start pt-1">
        <div className="mb-1 flex items-center justify-between gap-3">
          <div className="text-xs font-black uppercase tracking-[0.08em] text-slate-600">
            Tick timeline
          </div>
          <div className="text-xs font-black uppercase tracking-[0.08em] text-slate-600">{step + 1}/{snapshot.total_steps}</div>
        </div>
        <div className="relative -mt-2 h-16 overflow-visible">
          <div
            className={cn(
              "absolute inset-x-0 top-1/2 grid h-8 -translate-y-1/2 items-center overflow-visible",
              isDenseTimeline ? "gap-0" : isCompactTimeline ? "gap-[3px]" : "gap-[6px]"
            )}
            style={{ gridTemplateColumns: `repeat(${series.length}, minmax(0, 1fr))` }}
            aria-hidden="true"
          >
            {series.map((tick) => {
              const importance = tickImportance(tick);
              const active = tick.step === step;
              return (
                <div
                  key={tick.step}
                  title={`${importance.label} / score ${importance.score.toFixed(2)} / ${formatDateTime(tick.timestamp)}`}
                  className={cn(
                    "h-8 min-w-0 transition",
                    isCompactTimeline && "w-1/2 justify-self-center border border-[#1f2933]",
                    !isCompactTimeline && !isDenseTimeline && "border border-[#1f2933]",
                    importance.kind === "critical" && (active ? "bg-[#b91c1c]" : "bg-[#dc2626]"),
                    importance.kind === "high" && (active ? "bg-[#c2410c]" : "bg-[#f97316]"),
                    importance.kind === "medium" && (active ? "bg-[#d97706]" : "bg-[#facc15]"),
                    importance.kind === "watch" && (active ? "bg-[#eab308]" : "bg-[#fde68a]"),
                    importance.kind === "low" && (active ? "bg-[#15803d]" : "bg-[#27b76f]"),
                    active && "relative z-10 h-14 self-center ring-white",
                    active && isDenseTimeline && "ring-4",
                    active && !isDenseTimeline && "border border-[#1f2933] ring-8"
                  )}
                />
              );
            })}
          </div>
          <input
            aria-label="Simulation timeline"
            data-testid="timeline-slider"
            className="timeline-range absolute inset-0 z-20 h-16 w-full cursor-pointer"
            type="range"
            min={0}
            max={snapshot.total_steps - 1}
            step={1}
            value={step}
            onChange={(event) => setStep(Number(event.target.value))}
          />
        </div>
        <div className="mt-3 flex justify-between gap-4 text-base font-black uppercase tracking-[0.04em] text-slate-600">
          <span className="tabular-stable">{formatTime(start.timestamp)}</span>
          <span className="min-w-0 flex-1 text-center tabular-stable text-slate-950">{formatDateTime(snapshot.market.timestamp)}</span>
          <span className="tabular-stable">{formatTime(end.timestamp)}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 self-center text-sm max-lg:hidden">
        <div className="min-h-14 border border-[#1f2933] bg-white p-3">
          <div className="font-bold uppercase text-slate-500">mFRR</div>
          <div className="mt-1 font-black text-slate-900">{formatPrice(snapshot.market.mfrr_price_eur_per_mwh)}</div>
        </div>
        <div className="min-h-14 border border-[#1f2933] bg-white p-3">
          <div className="font-bold uppercase text-slate-500">Priority accuracy</div>
          <div className="mt-1 font-black text-slate-900">{formatPercent(run?.priority_accuracy?.score ?? 0)}</div>
          <div className="mt-0.5 text-[11px] font-bold uppercase tracking-[0.06em] text-slate-500">
            Capture {formatPercent(run?.priority_accuracy?.profit_capture_rate ?? 0)}
          </div>
        </div>
      </div>
    </div>
  );
}

function tickImportance(tick: ReturnType<typeof getMarketSeries>[number]) {
  if (tick.priority_signal) {
    return {
      kind: tick.priority_signal.tier,
      label: tick.priority_signal.label,
      score: tick.priority_signal.score
    } as const;
  }
  if (tick.events.some((event) => event.kind === "must_watch")) {
    return { kind: "watch", label: "Watch evidence", score: 0 } as const;
  }
  if (tick.events.some((event) => event.kind === "watch")) {
    return { kind: "watch", label: "Watch evidence", score: 0 } as const;
  }
  return { kind: "low", label: "Low priority", score: 0 } as const;
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function IconButton({
  label,
  onClick,
  children
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="flex h-11 w-11 items-center justify-center border border-[#1f2933] bg-white text-slate-700 transition hover:bg-[#dff5ec] hover:text-teal-700"
    >
      {children}
    </button>
  );
}
