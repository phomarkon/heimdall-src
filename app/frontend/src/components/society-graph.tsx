"use client";

import Graph from "graphology";
import { History, Share2, Users } from "lucide-react";
import Sigma from "sigma";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import type { AgentHistoryRecord, AgentNode, RunSnapshot, SocietyEdge } from "@/types/heimdall";
import { fetchAgentHistory } from "@/lib/api/run-adapter";
import { archetypeColor, edgeColor, llmFamilyColor } from "@/lib/theme";
import { useSelectedEntityStore } from "@/stores/selection";
import { cn, formatEur, formatMw, formatPrice, formatTime } from "@/lib/utils";
import type { PersonaArchetype } from "@/types/heimdall";

const TOOLTIP_WIDTH = 288;
const TOOLTIP_HEIGHT = 190;
const TOOLTIP_GAP = 56;
const TOOLTIP_MARGIN = 16;
const PROFIT_HALO_COLOR = "#b7f7c8";
const actionAgentLegend: Array<{ id: PersonaArchetype; label: string }> = [
  { id: "wind", label: "Wind" },
  { id: "ev", label: "EV" },
  { id: "retailer", label: "Retailer" },
  { id: "p2h", label: "P2H" },
  { id: "generator", label: "Generator" },
  { id: "arbitrageur", label: "Arbitrage" }
];

const infoAgentLegend: Array<{ id: PersonaArchetype; label: string }> = [
  { id: "grid-info", label: "Grid" },
  { id: "outage-info", label: "Outage" },
  { id: "price-info", label: "Price" },
  { id: "sizing-info", label: "Sizing" },
  { id: "uncertainty-info", label: "Uncertainty" },
  { id: "decision-info", label: "Decision" },
  { id: "risk-info", label: "Risk" }
];

export function SocietyGraph({ snapshot }: { snapshot: RunSnapshot }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<Sigma | null>(null);
  const hoveredNodeRef = useRef<string | null>(null);
  const selectedAgentIdRef = useRef<string | null>(null);
  const setSelected = useSelectedEntityStore((state) => state.setSelected);
  const selected = useSelectedEntityStore((state) => state.selected);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [hoverPosition, setHoverPosition] = useState<{ left: number; top: number } | null>(null);
  const [sigmaUnavailable, setSigmaUnavailable] = useState(false);
  const [historyAgentId, setHistoryAgentId] = useState<string | null>(null);

  const focalAgentId = snapshot.nodes.find((node) => node.is_focal)?.id ?? snapshot.selected_trace.agent_id;
  const selectedAgentId = selected.kind === "agent" ? selected.id : selected.kind === "focal" ? focalAgentId : null;
  selectedAgentIdRef.current = selectedAgentId;

  const graph = useMemo(() => {
    const nextGraph = new Graph();
    const maxProfit = Math.max(1, ...snapshot.nodes.map((node) => Math.max(0, node.pnl_eur)));
    const maxTickProfit = Math.max(1, ...snapshot.nodes.map((node) => Math.max(0, node.tick_pnl_eur ?? 0)));
    snapshot.nodes.forEach((node) => {
      const llmColor = llmFamilyColor[node.persona.llm_family] ?? "#64748b";
      const digest = snapshot.agent_traces?.[node.id]?.info_digest;
      const infoImportance = digest?.importance ?? 0;
      const isInfoAgent = node.persona.archetype.endsWith("-info");
      const profitRatio = Math.max(0, node.pnl_eur) / maxProfit;
      const nodeSize = isInfoAgent ? 8 + infoImportance * 10 : Math.min(31, 7 + profitRatio * 24);
      const tickProfit = Math.max(0, node.tick_pnl_eur ?? 0);
      const profitHalo = tickProfit > 0 ? Math.min(18, 4 + (tickProfit / maxTickProfit) * 14) : 0;
      const infoHalo = isInfoAgent && infoImportance > 0 ? 4 + infoImportance * 8 : 0;
      if (profitHalo > 0 || infoHalo > 0) {
        nextGraph.addNode(`${node.id}::profit-halo`, {
          x: node.x,
          y: node.y,
          label: "",
          size: nodeSize + Math.max(profitHalo, infoHalo),
          color: isInfoAgent ? "#fff0c7" : PROFIT_HALO_COLOR,
          isProfitHalo: true,
          zIndex: 0
        });
      }
      nextGraph.addNode(node.id, {
        x: node.x,
        y: node.y,
        label: node.persona.display_name,
        size: nodeSize,
        color: node.is_focal ? "#17a99a" : archetypeColor[node.persona.archetype],
        llmColor,
        archetype: node.persona.archetype,
        isFocal: node.is_focal,
        openPosition: node.open_position_mw,
        pnl: node.pnl_eur,
        tickPnl: tickProfit,
        zIndex: 2
      });
    });

    snapshot.edges.forEach((edge) => {
      if (nextGraph.hasNode(edge.source) && nextGraph.hasNode(edge.target) && !nextGraph.hasEdge(edge.source, edge.target)) {
        nextGraph.addDirectedEdgeWithKey(edge.id, edge.source, edge.target, {
          label: edge.label,
          size: edge.kind === "consensus" ? 1.4 + edge.strength * 2.4 : 1,
          color: edgeColor[edge.kind],
          kind: edge.kind
        });
      }
    });

    return nextGraph;
  }, [snapshot]);

  useEffect(() => {
    if (!containerRef.current || sigmaUnavailable) {
      return;
    }

    let renderer: Sigma;
    try {
      renderer = new Sigma(graph, containerRef.current, {
        allowInvalidContainer: true,
        hideEdgesOnMove: false,
        renderLabels: false,
        renderEdgeLabels: false,
        labelColor: { color: "#172033" },
        labelSize: 11,
        nodeReducer: (node, data) => {
          if (data.isProfitHalo) {
            return { ...data, label: undefined, forceLabel: false, highlighted: false };
          }
          const currentSelected = selectedAgentIdRef.current;
          const currentHovered = hoveredNodeRef.current;
          const focused = node === currentSelected || node === currentHovered;
          const related =
            !currentSelected ||
            focused ||
            graph.hasEdge(node, currentSelected) ||
            graph.hasEdge(currentSelected, node);
          return {
            ...data,
            color: data.color,
            size: focused ? data.size * 1.28 : data.size,
            label: undefined,
            zIndex: focused ? 4 : data.isFocal ? 3 : 1,
            forceLabel: false,
            highlighted: focused,
            colorOpacity: related ? 1 : 0.34
          };
        },
        edgeReducer: (edge, data) => {
          const currentSelected = selectedAgentIdRef.current;
          const source = graph.source(edge);
          const target = graph.target(edge);
          const related = !currentSelected || source === currentSelected || target === currentSelected;
          return {
            ...data,
            color: related ? data.color : "#cbd6e2",
            size: related ? data.size : 0.55,
            hidden: false
          };
        }
      });
    } catch (error) {
      console.warn("Sigma renderer unavailable; falling back to DOM graph.", error);
      setSigmaUnavailable(true);
      return;
    }

    renderer.on("clickNode", ({ node }) => {
      if (node.endsWith("::profit-halo")) {
        node = node.replace("::profit-halo", "");
      }
      setSelected(node === focalAgentId ? { kind: "focal" } : { kind: "agent", id: node });
    });
    renderer.on("clickEdge", ({ edge }) => {
      setSelected({ kind: "edge", id: edge });
    });
    renderer.on("enterNode", ({ node }) => {
      if (node.endsWith("::profit-halo")) {
        node = node.replace("::profit-halo", "");
      }
      hoveredNodeRef.current = node;
      setHoveredNode(node);
      const display = renderer.getNodeDisplayData(node);
      const bounds = containerRef.current?.getBoundingClientRect();
      if (display && bounds) {
        const nodeRadius = Math.max(18, (display.size ?? 10) * 2.4);
        const verticalGap = nodeRadius + TOOLTIP_GAP;
        const horizontalGap = nodeRadius + TOOLTIP_GAP;
        const canShowAbove = display.y - verticalGap - TOOLTIP_HEIGHT >= TOOLTIP_MARGIN;
        const canShowRight = display.x + horizontalGap + TOOLTIP_WIDTH <= bounds.width - TOOLTIP_MARGIN;
        const canShowLeft = display.x - horizontalGap - TOOLTIP_WIDTH >= TOOLTIP_MARGIN;
        const rawLeft = canShowRight
          ? display.x + horizontalGap
          : canShowLeft
            ? display.x - horizontalGap - TOOLTIP_WIDTH
            : display.x - TOOLTIP_WIDTH / 2;
        const rawTop = canShowAbove
          ? display.y - verticalGap - TOOLTIP_HEIGHT
          : display.y + verticalGap;
        setHoverPosition({
          left: Math.max(TOOLTIP_MARGIN, Math.min(rawLeft, bounds.width - TOOLTIP_WIDTH - TOOLTIP_MARGIN)),
          top: Math.max(TOOLTIP_MARGIN, Math.min(rawTop, bounds.height - TOOLTIP_HEIGHT - TOOLTIP_MARGIN))
        });
      }
      renderer.refresh();
    });
    renderer.on("leaveNode", () => {
      hoveredNodeRef.current = null;
      setHoveredNode(null);
      setHoverPosition(null);
      renderer.refresh();
    });

    rendererRef.current = renderer;
    const firstPaintFrame = window.requestAnimationFrame(() => renderer.refresh());
    const firstPaintFallback = window.setTimeout(() => renderer.refresh(), 120);

    return () => {
      rendererRef.current = null;
      window.cancelAnimationFrame(firstPaintFrame);
      window.clearTimeout(firstPaintFallback);
      renderer.kill();
    };
  }, [focalAgentId, graph, setSelected, sigmaUnavailable]);

  useEffect(() => {
    rendererRef.current?.refresh();
  }, [selectedAgentId]);

  const hovered = hoveredNode ? snapshot.nodes.find((node) => node.id === hoveredNode) : null;
  const selectedAgent = selectedAgentId ? snapshot.nodes.find((node) => node.id === selectedAgentId) : null;
  const selectedEdge = selected.kind === "edge" ? snapshot.edges.find((edge) => edge.id === selected.id) : null;
  const historyAgent = historyAgentId ? snapshot.nodes.find((node) => node.id === historyAgentId) : null;
  return (
    <div className="relative h-full w-full">
      <div
        ref={containerRef}
        data-testid="society-graph"
        className="sigma-container h-full w-full"
        aria-label="Interactive society graph of BRP agents and bilateral negotiations"
        role="application"
        tabIndex={0}
      >
        {sigmaUnavailable ? (
          <FallbackSocietyGraph snapshot={snapshot} selectedAgentId={selectedAgentId} focalAgentId={focalAgentId} onSelect={setSelected} />
        ) : null}
      </div>

      {hovered && hoverPosition ? (
        <div
          className="pointer-events-none absolute z-30 w-72 border border-[#1f2933] bg-white/95 p-3 backdrop-blur max-md:w-[min(18rem,calc(100%-2.5rem))]"
          style={{ left: hoverPosition.left, top: hoverPosition.top }}
        >
          <div className="flex items-center justify-between gap-3">
            <p className="truncate text-sm font-black uppercase text-slate-900">{hovered.persona.display_name}</p>
            <span
              className={cn(
                "border px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em]",
                hovered.is_focal ? "border-[#27b7a4] bg-[#dff5ec] text-teal-700" : "border-[#1f2933] bg-[#f7f6f2] text-slate-600"
              )}
            >
              {hovered.persona.archetype}
            </span>
          </div>
          <p className="mt-2 text-xs leading-5 text-slate-600">{hovered.belief}</p>
          <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
            <span className="border border-[#1f2933] bg-[#f7f6f2] p-2 text-slate-500">
              Profit
              <strong className="mt-1 block text-slate-900">{formatEur(hovered.pnl_eur)}</strong>
            </span>
            <span className="border border-[#1f2933] bg-[#f7f6f2] p-2 text-slate-500">
              Position
              <strong className="mt-1 block text-slate-900">{formatMw(hovered.open_position_mw)}</strong>
            </span>
          </div>
        </div>
      ) : null}

      {selectedAgent ? (
        <SelectedAgentCard
          node={selectedAgent}
          snapshot={snapshot}
          onClose={() => setSelected({ kind: "focal" })}
          onOpenHistory={() => setHistoryAgentId(selectedAgent.id)}
        />
      ) : null}

      {selectedEdge ? (
        <SelectedEdgeCard edge={selectedEdge} snapshot={snapshot} onClose={() => setSelected({ kind: "focal" })} />
      ) : null}

      {historyAgent ? (
        <AgentHistoryDrawer
          runId={snapshot.run_id}
          agent={historyAgent}
          onClose={() => setHistoryAgentId(null)}
        />
      ) : null}

      <InfoConsensusStrip snapshot={snapshot} selectedAgentId={selectedAgentId} onSelect={(id) => setSelected({ kind: "agent", id })} />
      <AgentTypeLegend />
    </div>
  );
}

function InfoConsensusStrip({
  snapshot,
  selectedAgentId,
  onSelect
}: {
  snapshot: RunSnapshot;
  selectedAgentId: string | null;
  onSelect: (id: string) => void;
}) {
  const infoNodes = snapshot.nodes.filter((node) => node.persona.archetype.endsWith("-info"));
  if (!infoNodes.length) {
    return null;
  }
  return (
    <div className="absolute left-4 top-4 z-20 grid w-[min(560px,calc(100%-2rem))] grid-cols-7 border border-[#1f2933] bg-white/95 text-[10px] font-black uppercase tracking-[0.08em] text-slate-700 backdrop-blur max-xl:grid-cols-4">
      {infoNodes.map((node) => {
        const digest = snapshot.agent_traces?.[node.id]?.info_digest;
        const impact = Math.round((digest?.importance ?? 0) * 100);
        const tone = digest?.risk_label === "high" ? "bg-[#ffe7df]" : digest?.uncertainty_label === "high" ? "bg-[#fff8e8]" : "bg-[#f7f6f2]";
        const selected = node.id === selectedAgentId;
        return (
          <button
            key={node.id}
            type="button"
            onClick={() => onSelect(node.id)}
            className={cn(
              tone,
              "min-w-0 border-r border-[#1f2933] px-2 py-1.5 text-left transition last:border-r-0 hover:bg-[#fff0c7] focus:outline-none focus:ring-2 focus:ring-teal-500/35",
              selected && "bg-[#fff0c7] ring-2 ring-[#1f2933]"
            )}
          >
            <div className="truncate" style={{ color: archetypeColor[node.persona.archetype] }}>
              {node.persona.archetype.replace("-info", "")}
            </div>
            <div className="mt-0.5 text-slate-950">{impact}%</div>
          </button>
        );
      })}
    </div>
  );
}

function AgentTypeLegend() {
  return (
    <div className="pointer-events-none absolute bottom-4 left-1/2 z-20 grid w-[min(1040px,calc(100%-2rem))] -translate-x-1/2 gap-1.5 border border-[#1f2933] bg-white/95 px-4 py-2 text-xs font-black uppercase tracking-[0.08em] text-slate-700 backdrop-blur max-lg:left-4 max-lg:w-[calc(100%-2rem)] max-lg:translate-x-0">
      <LegendRow label="Action" items={actionAgentLegend} />
      <LegendRow label="Info" items={infoAgentLegend} />
      <div className="flex items-center justify-center gap-x-3 overflow-hidden whitespace-nowrap max-lg:justify-start">
        <span className="w-14 text-slate-500">Links</span>
        <span className="flex min-w-0 items-center gap-1.5">
          <span className="h-1 w-5 shrink-0" style={{ background: edgeColor.consensus }} />
          Consensus
        </span>
        <span className="flex min-w-0 items-center gap-1.5">
          <span className="h-1 w-5 shrink-0 border-b-2 border-dashed" style={{ borderColor: edgeColor.broadcast }} />
          Broadcast
        </span>
      </div>
    </div>
  );
}

function LegendRow({ label, items }: { label: string; items: Array<{ id: PersonaArchetype; label: string }> }) {
  return (
    <div className="flex items-center justify-center gap-x-3 overflow-hidden whitespace-nowrap max-lg:justify-start">
      <span className="w-14 text-slate-500">{label}</span>
      {items.map((item) => (
        <span key={item.id} className="flex min-w-0 items-center gap-1.5">
          <span className="h-4 w-4 shrink-0 border border-[#1f2933]" style={{ background: archetypeColor[item.id] }} />
          {item.label}
        </span>
      ))}
    </div>
  );
}

function SelectedAgentCard({
  node,
  snapshot,
  onClose,
  onOpenHistory
}: {
  node: AgentNode;
  snapshot: RunSnapshot;
  onClose: () => void;
  onOpenHistory: () => void;
}) {
  const trace = snapshot.agent_traces?.[node.id] ?? (snapshot.selected_trace.agent_id === node.id ? snapshot.selected_trace : null);
  if (node.persona.archetype.endsWith("-info")) {
    return <SelectedInfoAgentCard node={node} trace={trace} onClose={onClose} onOpenHistory={onOpenHistory} />;
  }
  const action = trace?.proposed_action;
  const verdict = trace?.verifier_verdict;
  const outcome = trace?.realized_outcome;
  const status = outcome && outcome.fill_mw > 0 ? "filled" : verdict?.accepted ? "accepted" : verdict ? "blocked" : "observing";

  return (
    <aside
      className="absolute right-4 top-4 z-20 w-80 border-2 border-[#1f2933] bg-white/95 p-3 shadow-sm backdrop-blur max-lg:w-72"
      aria-label="Selected agent details"
      data-testid="selected-agent-card"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="retro-label text-slate-500">Selected agent</p>
          <h3 className="mt-1 truncate text-sm font-black uppercase text-slate-950">{node.persona.display_name}</h3>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <button
            type="button"
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center border border-[#1f2933] bg-[#f7f6f2] text-xs font-black text-slate-700 hover:bg-[#dff5ec]"
            aria-label="Reset selected agent"
            title="Reset selected agent"
          >
            x
          </button>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5 text-[10px] font-black uppercase tracking-[0.08em]">
        <span className="border border-[#1f2933] bg-[#f7f6f2] px-2 py-1 text-slate-600">{node.persona.archetype}</span>
        <span className={cn(
          "border px-2 py-1",
          status === "filled" || status === "accepted"
            ? "border-[#27b7a4] bg-[#dff5ec] text-teal-700"
            : status === "blocked"
              ? "border-[#ff6542] bg-[#ffe7df] text-rose-700"
              : "border-[#d39b14] bg-[#fff0c7] text-amber-700"
        )}>
          {status}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <AgentMetric label="Profit" value={formatEur(node.pnl_eur)} tone={node.pnl_eur > 0 ? "positive" : "neutral"} />
        <AgentMetric label="Position" value={formatMw(node.open_position_mw)} />
        <AgentMetric label="Model tier" value={formatModelTier(node.persona.llm_family)} title={`Raw simulator llm_id: ${node.persona.llm_family}`} />
        <AgentMetric label="Forecaster" value={node.persona.forecaster} />
      </div>

      {action ? (
        <div className="mt-3 border border-[#1f2933] bg-[#fbfaf7] p-2 text-xs">
          <div className="font-black uppercase text-slate-500">Latest action</div>
          <div className="mt-1 font-semibold text-slate-900">
            {action.direction} {formatMw(action.quantity_mw)} @ {action.price_eur_per_mwh.toFixed(1)} EUR/MWh
          </div>
        </div>
      ) : null}

      {outcome ? (
        <div className="mt-2 border border-[#1f2933] bg-[#fbfaf7] p-2 text-xs">
          <div className="font-black uppercase text-slate-500">Realized</div>
          <div className="mt-1 font-semibold text-slate-900">
            {formatMw(outcome.fill_mw)} filled / {formatEur(outcome.pnl_eur)}
          </div>
        </div>
      ) : null}

      <p className="mt-3 line-clamp-3 text-xs leading-5 text-slate-600">{node.belief}</p>

      <button
        type="button"
        onClick={onOpenHistory}
        className="mt-3 flex h-9 w-full items-center justify-center gap-2 border border-[#1f2933] bg-[#dff5ec] text-[10px] font-black uppercase tracking-[0.1em] text-teal-700 hover:bg-[#fff0c7]"
        aria-label="Full history"
        title="Full history"
      >
        <History className="h-4 w-4" aria-hidden="true" />
        Full history
      </button>
    </aside>
  );
}

function SelectedEdgeCard({
  edge,
  snapshot,
  onClose
}: {
  edge: SocietyEdge;
  snapshot: RunSnapshot;
  onClose: () => void;
}) {
  const source = snapshot.nodes.find((node) => node.id === edge.source);
  const target = snapshot.nodes.find((node) => node.id === edge.target);
  const Icon = edge.kind === "broadcast" ? Share2 : Users;
  return (
    <aside
      className="absolute right-4 top-4 z-20 w-80 border-2 border-[#1f2933] bg-white/95 p-3 shadow-sm backdrop-blur max-lg:w-72"
      aria-label="Selected interaction details"
      data-testid="selected-edge-card"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="retro-label text-slate-500">Interaction</p>
          <h3 className="mt-1 flex items-center gap-2 truncate text-sm font-black uppercase text-slate-950">
            <Icon className="h-4 w-4 shrink-0" style={{ color: edgeColor[edge.kind] }} aria-hidden="true" />
            {edge.kind === "broadcast" ? "Society broadcast" : "Same-side consensus"}
          </h3>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="flex h-7 w-7 shrink-0 items-center justify-center border border-[#1f2933] bg-[#f7f6f2] text-xs font-black text-slate-700 hover:bg-[#dff5ec]"
          aria-label="Reset selection"
          title="Reset selection"
        >
          x
        </button>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <AgentMetric label="From" value={source?.persona.display_name ?? edge.source} />
        <AgentMetric label="To" value={target?.persona.display_name ?? edge.target} />
        {edge.side ? <AgentMetric label="Side" value={edge.side} /> : null}
        <AgentMetric label="Strength" value={`${Math.round(edge.strength * 100)}%`} />
      </div>

      <p className="mt-3 text-xs leading-5 text-slate-600">{edge.detail}</p>
    </aside>
  );
}

function SelectedInfoAgentCard({
  node,
  trace,
  onClose,
  onOpenHistory
}: {
  node: AgentNode;
  trace: RunSnapshot["selected_trace"] | null;
  onClose: () => void;
  onOpenHistory: () => void;
}) {
  const digest = trace?.info_digest;
  return (
    <aside
      className="absolute right-4 top-4 z-20 w-[360px] border-2 border-[#1f2933] bg-white/95 p-3 shadow-sm backdrop-blur max-lg:w-80"
      aria-label="Selected information agent details"
      data-testid="selected-info-agent-card"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="retro-label text-slate-500">Information agent</p>
          <h3 className="mt-1 truncate text-sm font-black uppercase text-slate-950">{node.persona.display_name}</h3>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <button
            type="button"
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center border border-[#1f2933] bg-[#f7f6f2] text-xs font-black text-slate-700 hover:bg-[#dff5ec]"
            aria-label="Reset selected agent"
            title="Reset selected agent"
          >
            x
          </button>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5 text-[10px] font-black uppercase tracking-[0.08em]">
        <span className="border border-[#1f2933] bg-[#f7f6f2] px-2 py-1 text-slate-600">{node.persona.archetype.replace("-info", "")}</span>
        <span className="border border-[#d39b14] bg-[#fff8e8] px-2 py-1 text-amber-700">{Math.round((digest?.importance ?? 0) * 100)} impact</span>
        {digest?.direction_hint ? <span className="border border-[#4d8cff] bg-[#eef3ff] px-2 py-1 text-blue-700">hint {digest.direction_hint}</span> : null}
      </div>

      <p className="mt-3 line-clamp-5 text-xs leading-5 text-slate-700">{digest?.finding ?? node.belief}</p>

      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <AgentMetric label="Risk" value={digest?.risk_label ?? "n/a"} />
        <AgentMetric label="Uncertainty" value={digest?.uncertainty_label ?? "n/a"} />
        <AgentMetric label="Confidence" value={`${Math.round((digest?.confidence ?? 0) * 100)}%`} />
      </div>

      {digest?.watch_reasons.length ? (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {digest.watch_reasons.slice(0, 5).map((reason) => (
            <span key={reason} className="border border-[#d39b14] bg-[#fff8e8] px-2 py-1 text-[10px] font-black uppercase tracking-[0.08em] text-amber-700">
              {reason.replaceAll("_", " ")}
            </span>
          ))}
        </div>
      ) : null}

      {digest?.signals.length ? (
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
          {digest.signals.map((signal) => (
            <AgentMetric key={signal.label} label={signal.label} value={String(signal.value)} />
          ))}
        </div>
      ) : null}

      <button
        type="button"
        onClick={onOpenHistory}
        className="mt-3 flex h-9 w-full items-center justify-center gap-2 border border-[#1f2933] bg-[#dff5ec] text-[10px] font-black uppercase tracking-[0.1em] text-teal-700 hover:bg-[#fff0c7]"
        aria-label="Full history"
        title="Full history"
      >
        <History className="h-4 w-4" aria-hidden="true" />
        Full history
      </button>
    </aside>
  );
}

type HistoryFilter = "all" | "bids" | "filled" | "accepted" | "rejected" | "watch" | "errors";

const historyFilters: Array<{ id: HistoryFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "bids", label: "Bids" },
  { id: "filled", label: "Filled bids" },
  { id: "accepted", label: "Accepted" },
  { id: "rejected", label: "Rejected" },
  { id: "watch", label: "Watch" },
  { id: "errors", label: "Tool errors" }
];

function AgentHistoryDrawer({ runId, agent, onClose }: { runId: string; agent: AgentNode; onClose: () => void }) {
  const [filter, setFilter] = useState<HistoryFilter>("all");
  const { data, isLoading } = useQuery({
    queryKey: ["agent-history", runId, agent.id],
    queryFn: () => fetchAgentHistory(runId, agent.id)
  });
  const records = (data?.records ?? []).filter((record) => historyRecordMatches(record, filter));

  return (
    <aside
      className="absolute bottom-4 right-4 top-4 z-40 flex w-[min(620px,calc(100%-2rem))] select-none flex-col border-2 border-[#1f2933] bg-white shadow-[6px_6px_0_#1f2933]"
      aria-label="Agent full history"
      data-testid="agent-history-drawer"
    >
      <div className="border-b-2 border-[#1f2933] bg-[#fbfaf7] p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="retro-label text-slate-500">Full history</p>
            <h3 className="mt-1 truncate text-base font-black uppercase text-slate-950">{agent.persona.display_name}</h3>
            <p className="mt-1 text-xs font-semibold text-slate-500">
              {data?.total_records ?? 0} trace rows / {data?.trace_sha256.slice(0, 10) ?? "loading"} source
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center border border-[#1f2933] bg-white text-xs font-black text-slate-700 hover:bg-[#dff5ec]"
            aria-label="Close full history"
            title="Close full history"
          >
            x
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {historyFilters.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setFilter(item.id)}
              className={cn(
                "border border-[#1f2933] px-2.5 py-1.5 text-[10px] font-black uppercase tracking-[0.08em]",
                filter === item.id ? "bg-[#1f2933] text-white" : "bg-white text-slate-700 hover:bg-[#fff0c7]"
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto bg-white p-4 [touch-action:pan-y]">
        {isLoading ? <p className="text-sm font-semibold text-slate-500">Loading agent history...</p> : null}
        {!isLoading && !records.length ? <p className="text-sm font-semibold text-slate-500">No trace rows match this filter.</p> : null}
        <div className="space-y-3">
          {records.map((record) => (
            <HistoryRecordCard key={`${record.step}-${record.timestamp}`} record={record} />
          ))}
        </div>
      </div>
    </aside>
  );
}

function HistoryRecordCard({ record }: { record: AgentHistoryRecord }) {
  const action = textValue(record.decision.action, "abstain");
  const side = textValue(record.decision.side, "n/a");
  const quantity = numberValue(record.decision.quantity_mwh);
  const limit = numberValue(record.decision.limit_price_eur_mwh);
  const interval = record.forecast_interval_eur_mwh;
  const accepted = record.verifier.accepted;
  const tone =
    accepted === true
      ? "border-[#27b7a4] bg-[#dff5ec] text-teal-700"
      : accepted === false
        ? "border-[#ff6542] bg-[#ffe7df] text-rose-700"
        : "border-[#d39b14] bg-[#fff8e8] text-amber-700";

  return (
    <article className="border border-[#1f2933] bg-[#fbfaf7] p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-black uppercase tracking-[0.12em] text-slate-500">
            Step {record.step} / {formatTime(record.timestamp)}
          </p>
          <h4 className="mt-1 text-sm font-black uppercase text-slate-950">
            {action === "bid" ? `${side} ${quantity === null ? "n/a" : formatMw(quantity)} @ ${limit === null ? "n/a" : formatPrice(limit)}` : action}
          </h4>
        </div>
        <span className={cn("shrink-0 border px-2 py-1 text-[10px] font-black uppercase tracking-[0.08em]", tone)}>
          {accepted === true ? "accepted" : accepted === false ? "rejected" : action}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-4 gap-2 text-xs max-lg:grid-cols-2">
        <AgentMetric label="Zone" value={record.zone ?? "n/a"} />
        <AgentMetric label="Market" value={record.market_price_eur_mwh === null ? "n/a" : formatPrice(record.market_price_eur_mwh)} />
        <AgentMetric label="Forecast low" value={interval?.[0] === null || !interval ? "n/a" : formatPrice(interval[0])} />
        <AgentMetric label="Forecast high" value={interval?.[1] === null || !interval ? "n/a" : formatPrice(interval[1])} />
      </div>

      {record.realized_outcome ? (
        <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
          <AgentMetric label="Filled" value={formatMw(record.realized_outcome.fill_mw)} />
          <AgentMetric label="Realized" value={formatPrice(record.realized_outcome.realized_price_eur_per_mwh)} />
          <AgentMetric label="P&L" value={formatEur(record.realized_outcome.pnl_eur)} tone={record.realized_outcome.pnl_eur > 0 ? "positive" : "neutral"} />
        </div>
      ) : null}

      {record.rationale ? (
        <p className="mt-3 text-xs font-bold leading-5 text-slate-800">
          &quot;{record.rationale}&quot;
        </p>
      ) : null}

      {record.verifier.reason_codes.length ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {record.verifier.reason_codes.map((reason) => (
            <span key={reason} className="border border-[#ff6542] bg-[#ffe7df] px-2 py-1 text-[10px] font-black uppercase tracking-[0.08em] text-rose-700">
              {reason.replaceAll("_", " ")}
            </span>
          ))}
        </div>
      ) : null}

      {record.tool_calls.length ? (
        <details className="mt-3 border border-[#1f2933] bg-white p-2">
          <summary className="cursor-pointer text-xs font-black uppercase tracking-[0.1em] text-slate-700">
            {record.tool_calls.length} tool calls
          </summary>
          <div className="mt-2 space-y-2">
            {record.tool_calls.map((call, index) => (
              <details key={`${call.name}-${index}`} className="border border-[#1f2933] bg-[#fbfaf7] p-2">
                <summary className="cursor-pointer text-xs font-semibold text-slate-900">
                  {call.name} / {call.ok === false ? "error" : call.ok === true ? "ok" : "recorded"}
                </summary>
                <pre className="mt-2 max-h-72 select-text overflow-auto whitespace-pre-wrap break-words bg-white p-2 text-[11px] leading-5 text-slate-700">
                  {JSON.stringify({ arguments: call.arguments, result: call.result, error: call.error }, null, 2)}
                </pre>
              </details>
            ))}
          </div>
        </details>
      ) : null}
    </article>
  );
}

function historyRecordMatches(record: AgentHistoryRecord, filter: HistoryFilter) {
  const action = textValue(record.decision.action, "abstain");
  if (filter === "all") {
    return true;
  }
  if (filter === "bids") {
    return action === "bid";
  }
  if (filter === "filled") {
    return action === "bid" && (record.realized_outcome?.fill_mw ?? 0) > 0;
  }
  if (filter === "accepted") {
    return record.verifier.accepted === true;
  }
  if (filter === "rejected") {
    return record.verifier.accepted === false;
  }
  if (filter === "watch") {
    return action === "watch" || action === "must_watch";
  }
  return record.tool_calls.some((call) => call.ok === false || call.error);
}

function textValue(value: unknown, fallback: string) {
  return typeof value === "string" && value ? value : fallback;
}

function numberValue(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function AgentMetric({
  label,
  value,
  title,
  tone = "neutral"
}: {
  label: string;
  value: string;
  title?: string;
  tone?: "positive" | "neutral";
}) {
  return (
    <div className={cn("min-w-0 border border-[#1f2933] p-2", tone === "positive" ? "bg-[#dff5ec]" : "bg-white")}>
      <div className="text-[10px] font-black uppercase tracking-[0.08em] text-slate-500">{label}</div>
      <div className="mt-1 truncate text-sm font-black text-slate-900" title={title ?? value}>{value}</div>
    </div>
  );
}

function formatModelTier(llmFamily: string) {
  if (/^L\d+$/i.test(llmFamily)) {
    return `Tier ${llmFamily.slice(1)}`;
  }
  return llmFamily;
}

function FallbackSocietyGraph({
  snapshot,
  selectedAgentId,
  focalAgentId,
  onSelect
}: {
  snapshot: RunSnapshot;
  selectedAgentId: string | null;
  focalAgentId: string;
  onSelect: ReturnType<typeof useSelectedEntityStore.getState>["setSelected"];
}) {
  const toPoint = (x: number, y: number) => ({
    left: `${50 + x * 42}%`,
    top: `${50 + y * 42}%`
  });

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#fbfaf7]">
      <svg className="absolute inset-0 h-full w-full" aria-hidden="true">
        {snapshot.edges.map((edge) => {
          const source = snapshot.nodes.find((node) => node.id === edge.source);
          const target = snapshot.nodes.find((node) => node.id === edge.target);
          if (!source || !target) {
            return null;
          }
          const related = !selectedAgentId || edge.source === selectedAgentId || edge.target === selectedAgentId;
          return (
            <line
              key={edge.id}
              x1={`${50 + source.x * 42}%`}
              y1={`${50 + source.y * 42}%`}
              x2={`${50 + target.x * 42}%`}
              y2={`${50 + target.y * 42}%`}
              stroke={related ? edgeColor[edge.kind] : "#cbd6e2"}
              strokeWidth={edge.kind === "consensus" ? 1.4 + edge.strength * 2 : 1}
              strokeDasharray={edge.kind === "broadcast" ? "4 3" : undefined}
              opacity={related ? 0.9 : 0.35}
            />
          );
        })}
      </svg>
      {snapshot.nodes.map((node) => {
        const focused = node.id === selectedAgentId;
        const maxProfit = Math.max(1, ...snapshot.nodes.map((item) => Math.max(0, item.pnl_eur)));
        const maxTickProfit = Math.max(1, ...snapshot.nodes.map((item) => Math.max(0, item.tick_pnl_eur ?? 0)));
        const size = Math.max(14, Math.min(46, 15 + (Math.max(0, node.pnl_eur) / maxProfit) * 32));
        const tickProfit = Math.max(0, node.tick_pnl_eur ?? 0);
        const profitHalo = tickProfit > 0 ? Math.min(20, 4 + (tickProfit / maxTickProfit) * 14) : 0;
        return (
          <button
            key={node.id}
            type="button"
            title={node.persona.display_name}
            aria-label={node.persona.display_name}
            onClick={() => onSelect(node.id === focalAgentId ? { kind: "focal" } : { kind: "agent", id: node.id })}
            className={cn(
              "absolute -translate-x-1/2 -translate-y-1/2 border-2 border-[#1f2933] transition hover:z-20 hover:scale-125 focus:z-20 focus:outline-none focus:ring-2 focus:ring-teal-500/40",
              node.is_focal && "ring-4 ring-[#27b7a4]/30",
              focused && "z-10 scale-125"
            )}
            style={{
              ...toPoint(node.x, node.y),
              width: size,
              height: size,
              background: node.is_focal ? "#17a99a" : archetypeColor[node.persona.archetype],
              boxShadow: profitHalo > 0 ? `0 0 0 ${profitHalo}px ${PROFIT_HALO_COLOR}` : undefined
            }}
          >
            <span className="sr-only">{node.persona.display_name}</span>
          </button>
        );
      })}
      <div className="absolute left-4 top-4 border border-[#1f2933] bg-white/95 px-3 py-2 text-xs text-slate-600">
        DOM graph fallback / WebGL unavailable
      </div>
    </div>
  );
}
