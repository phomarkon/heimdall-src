"use client";

import {
  ArrowRightLeft,
  CircleDot,
  Clock3,
  Radio,
  ShieldCheck,
  ShieldX,
  Wrench
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { getAgentTrace } from "@/lib/api/run-adapter";
import { archetypeColor } from "@/lib/theme";
import { cn, formatDateTime, formatEur, formatMw, formatPrice, formatTime } from "@/lib/utils";
import { useSelectedEntityStore } from "@/stores/selection";
import type { AgentNode, PersonaArchetype, PrecomputedRun, RunSnapshot, SocietyEdge, ToolCall } from "@/types/heimdall";

type ActivityKind = "tool" | "interaction" | "verifier" | "market";

type ActivityItem = {
  id: string;
  kind: ActivityKind;
  timestamp: string;
  title: string;
  actor: string;
  detail: string;
  meta: string;
  tone: "teal" | "blue" | "amber" | "rose" | "slate";
  agentId?: string;
  edgeId?: string;
  eventId?: string;
  agentArchetype?: PersonaArchetype;
};

const iconByKind: Record<ActivityKind, LucideIcon> = {
  tool: Wrench,
  interaction: ArrowRightLeft,
  verifier: ShieldCheck,
  market: Radio
};

export function ActivitySidebar({ run, snapshot }: { run: PrecomputedRun; snapshot: RunSnapshot }) {
  const setSelected = useSelectedEntityStore((state) => state.setSelected);
  const feed = buildActivityFeed(run, snapshot.step, 52);
  const toolCalls = feed.filter((item) => item.kind === "tool").length;
  const verifierActions = feed.filter((item) => item.kind === "verifier").length;

  return (
    <section className="flex h-full min-h-0 flex-col">
      <div className="border-b-2 border-[#1f2933] bg-[#fbfaf7] px-5 py-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="retro-label text-slate-500">Modelchat</p>
            <h2 className="mt-1 text-base font-black uppercase text-slate-950">[Recent actions]</h2>
          </div>
          <span className="border-2 border-[#27b7a4] bg-[#dff5ec] px-2 py-1 text-xs font-bold text-teal-700">
            {formatTime(snapshot.market.timestamp)} UTC
          </span>
        </div>
        <div className="mt-2 grid grid-cols-3 gap-1.5 text-xs">
          <SummaryStat label="Actions" value={feed.length.toString()} />
          <SummaryStat label="Tools" value={toolCalls.toString()} />
          <SummaryStat label="Verify" value={verifierActions.toString()} />
        </div>
      </div>

      <div
        className="retro-noise min-h-0 flex-1 overflow-y-auto px-3 py-3"
        aria-label="Recent agent progress"
        data-testid="activity-feed"
      >
        <div className="space-y-2" role="list">
          {feed.map((item) => (
            <ActivityCard
              key={item.id}
              item={item}
              onSelect={() => {
                if (item.edgeId) {
                  setSelected({ kind: "edge", id: item.edgeId });
                } else if (item.agentId) {
                  setSelected({ kind: "agent", id: item.agentId });
                } else if (item.eventId) {
                  setSelected({ kind: "event", id: item.eventId });
                }
              }}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

export function buildActivityFeed(run: PrecomputedRun, currentStep: number, limit = 32): ActivityItem[] {
  const items: ActivityItem[] = [];
  const start = Math.max(0, currentStep - 10);
  const focalId =
    run.snapshots[currentStep]?.nodes.find((node) => node.is_focal)?.id ??
    run.snapshots[currentStep]?.selected_trace.agent_id ??
    "agent-000";

  for (let step = currentStep; step >= start && items.length < limit; step -= 1) {
    const snapshot = run.snapshots[step];
    if (!snapshot) {
      continue;
    }

    const focalTrace = snapshot.selected_trace;
    const traces = Object.values(snapshot.agent_traces ?? { [focalTrace.agent_id]: focalTrace });
    const visibleTraces = step === currentStep ? traces : [focalTrace];

    for (const trace of visibleTraces) {
      const calls = step === currentStep ? trace.tool_calls.slice(0, 3) : trace.tool_calls.slice(0, 1);
      for (const call of calls) {
        items.push(toolActivity(call, trace.persona.display_name, trace.agent_id, trace.timestamp, trace.persona.archetype));
      }
    }

    const verdict = focalTrace.verifier_verdict;
    items.push({
      id: `verdict-${step}`,
      kind: "verifier",
      timestamp: focalTrace.timestamp,
      title: verdict.accepted ? "Bid certified for mFRR submission" : "Verifier blocked focal bid",
      actor: "Two-stage verifier",
      detail: verdict.accepted
        ? `Physical checks pass; π_min ${formatEur(verdict.worst_case_profit_eur ?? 0)} clears τ ${formatEur(verdict.threshold_eur ?? 0)}.`
        : verdict.retry_suggestion ?? "Verifier returned a retry suggestion.",
      meta: `${focalTrace.proposed_action.direction} ${formatMw(focalTrace.proposed_action.quantity_mw)} @ ${formatPrice(focalTrace.proposed_action.price_eur_per_mwh)}`,
      tone: verdict.accepted ? "teal" : "rose",
      agentId: focalTrace.agent_id,
      agentArchetype: focalTrace.persona.archetype
    });

    items.push({
      id: `quote-${step}`,
      kind: "tool",
      timestamp: focalTrace.timestamp,
      title: focalTrace.persona.display_name,
      actor: "Tool called: propose_action",
      detail: `Proposes ${focalTrace.proposed_action.direction} ${formatMw(focalTrace.proposed_action.quantity_mw)} in ${focalTrace.proposed_action.market}; spread chosen from ACI interval and storage headroom.`,
      meta: `${formatPrice(focalTrace.proposed_action.price_eur_per_mwh)} / delivery ${formatTime(focalTrace.proposed_action.delivery_quarter)}`,
      tone: "teal",
      agentId: focalTrace.agent_id,
      agentArchetype: focalTrace.persona.archetype
    });

    if (!snapshot.agent_traces) {
      const peer = snapshot.nodes[((step * 7) % (snapshot.nodes.length - 1)) + 1];
      const peerTrace = getAgentTrace(run, peer.id, step);
      const peerCall = peerTrace.tool_calls[(step + 1) % Math.max(1, peerTrace.tool_calls.length)] ?? peerTrace.tool_calls[0];
      if (peerCall) {
        items.push(toolActivity(peerCall, peer.persona.display_name, peer.id, peerTrace.timestamp, peer.persona.archetype));
      }
    }

    for (const event of snapshot.market.events) {
      items.push({
        id: event.id,
        kind: "market",
        timestamp: snapshot.market.timestamp,
        title: event.kind === "gate_closure" ? "mFRR gate-closure monitor" : event.label,
        actor: "Market simulator",
        detail:
          event.kind === "price_spike"
            ? `mFRR moved to ${formatPrice(snapshot.market.mfrr_price_eur_per_mwh)} with ${formatMw(snapshot.market.imbalance_mw)} imbalance.`
            : `Interval state updated for DK1/DK2 at ${formatDateTime(snapshot.market.timestamp)}.`,
        meta: `${snapshot.market.gate_closure_minutes} min gate`,
        tone: event.kind === "rejected_bid" || event.kind === "price_spike" ? "amber" : "blue",
        eventId: event.id
      });
    }

    for (const edge of interactionCandidates(snapshot.edges, focalId)) {
      const source = findNode(snapshot.nodes, edge.source);
      const target = findNode(snapshot.nodes, edge.target);
      items.push({
        id: edge.id,
        kind: "interaction",
        timestamp: snapshot.market.timestamp,
        title: interactionTitle(edge),
        actor: `${source?.persona.display_name ?? edge.source} -> ${target?.persona.display_name ?? edge.target}`,
        detail: edge.detail,
        meta: `${edge.kind} / strength ${Math.round(edge.strength * 100)}%`,
        tone: edge.kind === "consensus" ? "teal" : "blue",
        edgeId: edge.id,
        agentArchetype: source?.persona.archetype
      });
    }
  }

  return items.slice(0, limit);
}

function toolActivity(call: ToolCall, actor: string, agentId: string, timestamp: string, agentArchetype?: PersonaArchetype): ActivityItem {
  return {
    id: `${agentId}-${call.id}`,
    kind: "tool",
    timestamp,
    title: actor,
    actor: call.label,
    detail: call.summary,
    meta: `${call.status} / ${call.latency_ms}ms`,
    tone: call.status === "error" ? "rose" : call.kind === "forecast" ? "blue" : "slate",
    agentId,
    agentArchetype
  };
}

function interactionCandidates(edges: SocietyEdge[], focalId: string) {
  const focal = edges.filter((edge) => edge.source === focalId || edge.target === focalId);
  const consensus = edges.filter((edge) => edge.kind === "consensus");
  return [...focal, ...consensus, ...edges].filter(uniqueById).slice(0, 2);
}

function uniqueById(edge: SocietyEdge, index: number, edges: SocietyEdge[]) {
  return edges.findIndex((candidate) => candidate.id === edge.id) === index;
}

function findNode(nodes: AgentNode[], id: string) {
  return nodes.find((node) => node.id === id);
}

function interactionTitle(edge: SocietyEdge) {
  if (edge.kind === "broadcast") {
    return "Society broadcast shared";
  }
  return edge.side ? `Same-side consensus (${edge.side})` : "Same-side consensus";
}

function ActivityCard({ item, onSelect }: { item: ActivityItem; onSelect: () => void }) {
  const Icon = item.kind === "verifier" && item.tone === "rose" ? ShieldX : iconByKind[item.kind];
  const agentColor = item.agentArchetype ? archetypeColor[item.agentArchetype] : null;

  return (
    <button
      type="button"
      role="listitem"
      onClick={onSelect}
      className={cn(
        "group grid w-full grid-cols-[30px_minmax(0,1fr)] gap-3 border-2 bg-white p-2 text-left transition focus:outline-none focus:ring-2 focus:ring-teal-500/35",
        item.tone === "teal" && "border-[#27b7a4] hover:bg-[#effbf7]",
        item.tone === "blue" && "border-[#4d8cff] hover:bg-[#f1f6ff]",
        item.tone === "amber" && "border-[#d39b14] hover:bg-[#fff8e8]",
        item.tone === "rose" && "border-[#ff6542] hover:bg-[#fff2ee]",
        item.tone === "slate" && "border-[#1f2933] hover:bg-[#f7f6f2]"
      )}
    >
      <span className="flex flex-col items-center gap-1">
        <span
          className={cn(
            "mt-0.5 flex h-7 w-7 items-center justify-center border",
            item.tone === "teal" && "border-[#27b7a4] bg-[#dff5ec] text-teal-700",
            item.tone === "blue" && "border-[#4d8cff] bg-[#eef3ff] text-blue-700",
            item.tone === "amber" && "border-[#d39b14] bg-[#fff0c7] text-amber-700",
            item.tone === "rose" && "border-[#ff6542] bg-[#ffe7df] text-rose-700",
            item.tone === "slate" && "border-[#1f2933] bg-[#f7f6f2] text-slate-700"
          )}
        >
          <Icon className="h-3.5 w-3.5" aria-hidden="true" />
        </span>
        {agentColor ? <span className="h-3 w-7 border border-[#1f2933]" style={{ background: agentColor }} /> : null}
      </span>
      <span className="min-w-0">
        <span className="flex items-center justify-between gap-2">
          <span className="truncate text-xs font-black uppercase text-slate-950">{item.title}</span>
          <span className="shrink-0 text-[10px] font-medium text-slate-500">{formatTime(item.timestamp)}</span>
        </span>
        <span className="mt-0.5 block truncate text-[11px] font-medium text-slate-600">{item.actor}</span>
        <span className="mt-1 block overflow-hidden text-[11px] leading-4 text-slate-600 [display:-webkit-box] [-webkit-box-orient:vertical] [-webkit-line-clamp:2]">
          {item.detail}
        </span>
        <span className="mt-1.5 flex items-center gap-1 text-[10px] text-slate-500">
          <Clock3 className="h-3 w-3" aria-hidden="true" />
          {item.meta}
        </span>
      </span>
    </button>
  );
}

function SummaryStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-[#1f2933] bg-white p-1.5">
      <div className="flex items-center gap-1 text-[10px] font-black uppercase tracking-[0.12em] text-slate-500">
        <CircleDot className="h-3 w-3 text-teal-700" aria-hidden="true" />
        {label}
      </div>
      <div className="mt-0.5 text-sm font-semibold text-slate-950">{value}</div>
    </div>
  );
}
