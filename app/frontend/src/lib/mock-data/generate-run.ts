import type {
  AgentNode,
  AgentTrace,
  FocalBaselineRow,
  MarketTick,
  Persona,
  PersonaArchetype,
  RiskAttitude,
  RunCatalogEntry,
  RunSnapshot,
  SocietyEdge,
  Sophistication
} from "@/types/heimdall";
import type { PrecomputedRun } from "@/types/heimdall";

const RUN_ID = "heimdall-dk1-dk2-mock-001";
const TOTAL_STEPS = 96;
const START = Date.UTC(2025, 9, 1, 0, 0, 0);

const archetypes: PersonaArchetype[] = ["wind", "ev", "retailer", "p2h", "generator", "arbitrageur"];
const risks: RiskAttitude[] = ["averse", "neutral", "seeking"];
const sophistication: Sophistication[] = ["low", "medium", "high"];
const llms = ["Qwen", "Qwen", "Gemma", "Mistral", "Llama", "DeepSeek"];
const forecasters = ["F0 seasonal AR", "F7 split-CP transformer", "F8 online ACI", "F9 TimesFM-2.0", "F11 PriceFM"];

function timestampFor(step: number) {
  return new Date(START + step * 15 * 60 * 1000).toISOString();
}

function seededWave(index: number, step: number, scale = 1) {
  return Math.sin(index * 1.71 + step * 0.19) * scale + Math.cos(index * 0.41 + step * 0.07) * scale * 0.45;
}

function makePersona(index: number): Persona {
  const isFocal = index === 0;
  const archetype = isFocal ? "p2h" : archetypes[index % archetypes.length];
  const display = isFocal
    ? "Focal P2H Market Maker"
    : `${archetype.toUpperCase()} BRP ${String(index).padStart(2, "0")}`;

  return {
    agent_id: isFocal ? "agent-p2h-focal" : `agent-${String(index).padStart(2, "0")}`,
    display_name: display,
    archetype,
    risk_attitude: risks[index % risks.length],
    sophistication: sophistication[(index + 1) % sophistication.length],
    info_latency_min: [0, 0, 360, 1440][index % 4],
    capacity_mw: isFocal ? 50 : 8 + ((index * 7) % 54),
    storage_mwh: archetype === "p2h" || archetype === "ev" ? 25 + ((index * 11) % 90) : null,
    llm_family: isFocal ? "Qwen" : llms[index % llms.length],
    forecaster: isFocal ? "F8 online ACI" : forecasters[index % forecasters.length]
  };
}

const personas = Array.from({ length: 50 }, (_, index) => makePersona(index));

function positionFor(index: number) {
  const cluster = archetypes.indexOf(personas[index].archetype);
  const clusterAngle = (cluster / archetypes.length) * Math.PI * 2;
  const within = (index % 9) / 9;
  const radius = index === 0 ? 0.08 : 0.32 + within * 0.45;
  return {
    x: Math.cos(clusterAngle) * radius + Math.cos(index * 2.31) * 0.08,
    y: Math.sin(clusterAngle) * radius + Math.sin(index * 1.93) * 0.08
  };
}

function marketTick(step: number): MarketTick {
  const morningRamp = Math.sin((step / TOTAL_STEPS) * Math.PI * 2 - 0.7) * 24;
  const stress = step > 54 && step < 68 ? (68 - Math.abs(61 - step)) * 5.2 : 0;
  const dk1 = 61 + morningRamp + stress + Math.sin(step * 0.57) * 7;
  const dk2 = 57 + morningRamp * 0.7 + stress * 0.6 + Math.cos(step * 0.44) * 6;
  const mfrr = dk1 + Math.sin(step * 0.33) * 18 + (step % 17 === 0 ? 42 : 0);
  const events: MarketTick["events"] = [];

  if (step % 12 === 0) {
    events.push({ id: `gate-${step}`, kind: "gate_closure", label: "mFRR gate closure" });
  }
  if (mfrr > 120) {
    events.push({ id: `spike-${step}`, kind: "price_spike", label: "mFRR price spike" });
  }
  if (step % 9 === 4) {
    events.push({ id: `accept-${step}`, kind: "accepted_bid", label: "Verifier accepted focal bid" });
  }
  if (step % 21 === 8) {
    events.push({ id: `reject-${step}`, kind: "rejected_bid", label: "Verifier rejected retry" });
  }

  return {
    step,
    timestamp: timestampFor(step),
    dk1_price_eur_per_mwh: dk1,
    dk2_price_eur_per_mwh: dk2,
    mfrr_price_eur_per_mwh: mfrr,
    imbalance_mw: Math.sin(step * 0.28) * 115 + Math.cos(step * 0.11) * 42,
    gate_closure_minutes: 45 - ((step * 15) % 45),
    priority_signal: mockPrioritySignal(step),
    events
  };
}

function mockPrioritySignal(step: number): MarketTick["priority_signal"] {
  const stressBand = step > 54 && step < 68 ? 2.8 : 0;
  const score = Math.max(0, 0.8 + stressBand + Math.sin(step * 0.31) * 1.1 + (step % 9 === 4 ? 1.3 : 0));
  const tier = score > 4.1 ? "critical" : score > 2.8 ? "high" : score > 1.5 ? "medium" : "low";
  return {
    score: Number(score.toFixed(3)),
    rank: null,
    percentile: tier === "critical" ? 0.95 : tier === "high" ? 0.84 : tier === "medium" ? 0.62 : 0.25,
    tier,
    label: `${tier[0].toUpperCase()}${tier.slice(1)} priority`,
    drivers: tier === "low" ? [] : ["activation context", "bid pressure"],
    risks: tier === "critical" ? ["price-not-crossed risk"] : [],
    components: {
      activation_context: tier === "low" ? 0.15 : 0.62,
      clearability: tier === "critical" ? 0.74 : 0.42,
      bid_pressure: tier === "low" ? 0.2 : 0.58
    }
  };
}

function societyEdges(step: number): SocietyEdge[] {
  const ns = nodes(step);
  const action = ns.filter((node) => !node.persona.archetype.endsWith("-info"));
  const focal = ns.find((node) => node.is_focal) ?? ns[0];
  const edges: SocietyEdge[] = [];

  const groups: Array<["up" | "down", AgentNode[]]> = [
    ["up", action.filter((node) => node.open_position_mw >= 0).slice(0, 6)],
    ["down", action.filter((node) => node.open_position_mw < 0).slice(0, 6)]
  ];
  for (const [side, group] of groups) {
    if (group.length < 2) {
      continue;
    }
    const anchor = group.find((node) => node.is_focal) ?? group[0];
    const strength = Math.min(1, group.length / 8);
    for (const node of group) {
      if (node.id === anchor.id) {
        continue;
      }
      edges.push({
        id: `consensus-${step}-${side}-${node.id}`,
        source: anchor.id,
        target: node.id,
        kind: "consensus",
        side,
        direction: side === "up" ? "sell" : "buy",
        market: "mFRR",
        strength: Number(strength.toFixed(3)),
        label: `Same-side ${side}`,
        detail: `${group.length} agents leaning ${side} this interval.`,
        started_step: step,
        expires_step: step
      });
    }
  }

  const consumers = action.filter((node) => !node.is_focal).slice(0, 5 + (step % 4));
  for (const node of consumers) {
    edges.push({
      id: `broadcast-${step}-${node.id}`,
      source: focal.id,
      target: node.id,
      kind: "broadcast",
      side: null,
      direction: null,
      market: "mFRR",
      strength: 0.5,
      label: "Society broadcast",
      detail: "Shared market digest broadcast to the society this interval.",
      started_step: step,
      expires_step: step
    });
  }

  return edges;
}

function nodes(step: number): AgentNode[] {
  return personas.map((persona, index) => {
    const point = positionFor(index);
    const position = seededWave(index, step, persona.capacity_mw / 2);
    return {
      id: persona.agent_id,
      persona,
      x: point.x,
      y: point.y,
      open_position_mw: Number(position.toFixed(2)),
      pnl_eur: Math.round(seededWave(index + 9, step, 600) + step * (index === 0 ? 34 : 5)),
      tick_pnl_eur: step % (index + 3) === 0 ? Math.round(Math.max(0, 40 + seededWave(index + 13, step, 120))) : 0,
      belief:
        persona.info_latency_min === 0
          ? "Real-time access to EDS imbalance and mFRR prices; can quote before delayed peers react."
          : "Delayed information access; leaning on recent imbalance history and bilateral offers.",
      is_focal: index === 0,
      verifier_acceptance_rate: index === 0 ? 0.82 + Math.sin(step / 11) * 0.05 : null
    };
  });
}

function clampStep(step = 0) {
  return Math.max(0, Math.min(TOTAL_STEPS - 1, Math.round(step)));
}

function buildRunSnapshot(runId = RUN_ID, step = 0): RunSnapshot {
  const safeStep = Math.max(0, Math.min(TOTAL_STEPS - 1, Math.round(step)));
  const snapshotNodes = nodes(safeStep);
  const trace = getAgentTrace(runId, "agent-p2h-focal", safeStep);
  const interval = trace.verifier_verdict.conformal_interval;
  const realized = trace.realized_outcome?.realized_price_eur_per_mwh ?? marketTick(safeStep).mfrr_price_eur_per_mwh;
  return {
    run_id: runId,
    step: safeStep,
    total_steps: TOTAL_STEPS,
    nodes: snapshotNodes,
    edges: societyEdges(safeStep),
    selected_trace: trace,
    market: marketTick(safeStep),
    forecast_diagnostics: {
      forecaster_id: "f8",
      interval_low_eur_mwh: interval.quantile_low,
      interval_high_eur_mwh: interval.quantile_high,
      interval_width_eur_mwh: Number((interval.quantile_high - interval.quantile_low).toFixed(2)),
      realized_price_eur_mwh: realized,
      covered: interval.quantile_low <= realized && realized <= interval.quantile_high,
      spot_mfrr_spread_eur_mwh: Number((marketTick(safeStep).mfrr_price_eur_per_mwh - marketTick(safeStep).dk1_price_eur_per_mwh).toFixed(2)),
      up_edge_eur_mwh: Number((interval.quantile_low - marketTick(safeStep).dk1_price_eur_per_mwh).toFixed(2)),
      down_edge_eur_mwh: Number((marketTick(safeStep).dk1_price_eur_per_mwh - interval.quantile_high).toFixed(2)),
      expected_spread_eur_mwh: Number((realized - marketTick(safeStep).dk1_price_eur_per_mwh).toFixed(2)),
      worst_case_profit_eur: trace.verifier_verdict.worst_case_profit_eur
    },
    health: {
      coverage: 0.901 + Math.sin(safeStep / 16) * 0.018,
      verifier_acceptance_rate: 0.78 + Math.cos(safeStep / 13) * 0.06,
      cumulative_pnl_eur: Math.round(snapshotNodes.reduce((sum, node) => sum + node.pnl_eur, 0)),
      gpu_utilization: 0.64 + Math.sin(safeStep / 9) * 0.12,
      wall_time_minutes: safeStep * 0.55
    }
  };
}

export function getAgentTrace(runId = RUN_ID, agentId = "agent-p2h-focal", step = 0): AgentTrace {
  const safeStep = clampStep(step);
  const persona = personas.find((entry) => entry.agent_id === agentId) ?? personas[0];
  const price = marketTick(safeStep).mfrr_price_eur_per_mwh;
  const rejected = persona.agent_id === "agent-p2h-focal" && safeStep % 21 === 8;
  const physicalRejected = rejected && safeStep % 42 === 8;
  const lower = price - 18 - Math.max(0, Math.sin(safeStep / 4) * 4);
  const upper = price + 22 + Math.max(0, Math.cos(safeStep / 5) * 6);
  const proposedQuantity = persona.agent_id === "agent-p2h-focal" ? 8 + (safeStep % 7) * 2 : Math.min(5, persona.capacity_mw * 0.1);
  const proposedDirection = safeStep % 2 === 0 ? "sell" : "buy";
  const proposedPrice = Number((price + seededWave(1, safeStep, 6)).toFixed(2));

  return {
    run_id: runId,
    step: safeStep,
    timestamp: timestampFor(safeStep),
    agent_id: persona.agent_id,
    persona,
    state: {
      soc_mwh: persona.storage_mwh ? Math.max(0, persona.storage_mwh / 2 + seededWave(2, safeStep, 12)) : null,
      exposure_mw: seededWave(4, safeStep, persona.capacity_mw / 2),
      cash_eur: 50_000 + safeStep * 340 + seededWave(8, safeStep, 2_000)
    },
    reasoning:
      persona.agent_id === "agent-p2h-focal"
        ? "Forecaster-agent updates the ACI interval, risk-agent checks CVaR and storage headroom, regulator-agent confirms the 45-minute mFRR gate, and quoter-agent proposes only if the verifier can certify worst-case profit."
        : "Persona balances residual position against expected imbalance cost, local forecast access, and recent bilateral hedge offers.",
    tool_calls: [
      {
        id: `forecast-${safeStep}`,
        kind: "forecast",
        label: persona.agent_id === "agent-p2h-focal" ? "Forecaster-agent: F8 online ACI" : `Forecast tool: ${persona.forecaster}`,
        status: "success",
        latency_ms: 180 + (safeStep % 9) * 24,
        summary: `Predicts next-quarter mFRR interval ${lower.toFixed(1)} to ${upper.toFixed(1)} €/MWh; target coverage α=0.10.`
      },
      {
        id: `simulate-${safeStep}`,
        kind: "simulate",
        label: persona.agent_id === "agent-p2h-focal" ? "Risk-agent: market impact replay" : "Market replay tool",
        status: "success",
        latency_ms: 94 + (safeStep % 5) * 13,
        summary:
          persona.agent_id === "agent-p2h-focal"
            ? `Stress replay estimates ${seededWave(5, safeStep, 2.4).toFixed(1)} MW price impact and keeps daily CVaR inside budget.`
            : "Stress replay checks residual exposure and expected imbalance cost."
      },
      {
        id: `regulation-${safeStep}`,
        kind: "regulation",
        label: persona.agent_id === "agent-p2h-focal" ? "Regulator-agent: gate check" : "Regulation bulletin scan",
        status: "success",
        latency_ms: 68 + (safeStep % 4) * 17,
        summary: `${marketTick(safeStep).gate_closure_minutes} min to mFRR gate closure; bid timing and 15-minute MTU are valid.`
      },
      {
        id: `verifier-${safeStep}`,
        kind: "verifier",
        label: "Verifier: physical + conformal",
        status: rejected ? "error" : "success",
        latency_ms: 42,
        summary: rejected
          ? "Rejected with structured retry: reduce MW or widen spread until π_min clears τ."
          : "Accepted: physical envelope passes and conformal worst-case profit clears τ."
      }
    ],
    proposed_action: {
      market: "mFRR",
      direction: proposedDirection,
      quantity_mw: proposedQuantity,
      price_eur_per_mwh: proposedPrice,
      delivery_quarter: timestampFor(safeStep + 3)
    },
    verifier_verdict: {
      accepted: !rejected,
      stage_failed: rejected ? (physicalRejected ? "physical" : "conformal") : null,
      physical_violation: physicalRejected
        ? { violated: "ramp_limit", current_delta_mw: 18, max_delta_mw: 12 }
        : null,
      worst_case_profit_eur: rejected ? -138 : 246 + safeStep * 3,
      threshold_eur: -100,
      retry_suggestion: rejected ? "Reduce quantity by at least 6 MW or widen bid spread." : null,
      conformal_interval: {
        horizon_minutes: 15,
        quantile_low: Number(lower.toFixed(2)),
        quantile_high: Number(upper.toFixed(2)),
        alpha: 0.1
      }
    },
    realized_outcome: safeStep > 2
      ? {
          fill_mw: 4 + (safeStep % 6),
          realized_price_eur_per_mwh: Number((price + seededWave(3, safeStep, 3)).toFixed(2)),
          pnl_eur: Math.round(120 + seededWave(3, safeStep, 90))
        }
      : null
  };
}

const marketSeriesCache = Array.from({ length: TOTAL_STEPS }, (_, step) => marketTick(step));
const snapshotCache = Array.from({ length: TOTAL_STEPS }, (_, step) => buildRunSnapshot(RUN_ID, step));
const precomputedRun: PrecomputedRun = {
  run_id: RUN_ID,
  total_steps: TOTAL_STEPS,
  snapshots: snapshotCache,
  market_series: marketSeriesCache,
  forecaster_leaderboard: [
    {
      model_id: "f3_ensemble",
      label: "5-seed PatchTST ensemble",
      seed_count: 5,
      q10_pinball: "113.2 ± 0.0",
      q50_pinball: "331.4 ± 0.0",
      q90_pinball: "333.2 ± 0.0",
      mean_pinball: "259.3 ± 0.0",
      raw_coverage: "64.5 ± 0.0",
      aci_coverage: "89.7 ± 0.0",
      status: "usable"
    },
    {
      model_id: "f7",
      label: "PatchTST split-CP",
      seed_count: 5,
      q10_pinball: "118.3 ± 4.4",
      q50_pinball: "336.2 ± 5.4",
      q90_pinball: "336.7 ± 5.2",
      mean_pinball: "263.8 ± 3.0",
      raw_coverage: "62.7 ± 3.2",
      aci_coverage: "89.7 ± 0.0",
      status: "usable"
    },
    {
      model_id: "f8",
      label: "Multivariate PatchTST ACI",
      seed_count: 5,
      q10_pinball: "123.2 ± 3.9",
      q50_pinball: "341.7 ± 19.6",
      q90_pinball: "329.3 ± 7.0",
      mean_pinball: "264.7 ± 8.6",
      raw_coverage: "57.5 ± 3.5",
      aci_coverage: "89.7 ± 0.1",
      status: "usable"
    },
    {
      model_id: "f9",
      label: "TimesFM-2.0 zero-shot",
      seed_count: null,
      q10_pinball: null,
      q50_pinball: null,
      q90_pinball: null,
      mean_pinball: null,
      raw_coverage: null,
      aci_coverage: null,
      status: "registered / dependency missing"
    }
  ],
  forecaster_summary: {
    active_forecaster_id: "f8",
    run_ids_by_forecaster: {
      f8: [RUN_ID],
      f7: ["bfa-core12-broadcast-apr02-0530-48-f7"],
      f3_ensemble: ["bfa-core12-broadcast-apr02-0530-48-f3_ensemble"]
    },
    coverage: snapshotCache[snapshotCache.length - 1].health.coverage,
    accepted_bid_rate: snapshotCache[snapshotCache.length - 1].health.verifier_acceptance_rate,
    cumulative_pnl_eur: snapshotCache[snapshotCache.length - 1].health.cumulative_pnl_eur,
    selected_tick_count: 18
  },
  priority_accuracy: {
    score: 0.71,
    precision: 0.68,
    recall: 0.74,
    profit_capture_rate: 0.64,
    selected_tick_count: 18,
    positive_tick_count: 22
  },
  focal_baselines: [
    {
      run_id: "mi00-baseline-profitguard-24-q32",
      label: "Profit-guard baseline 24-q32",
      kind: "baseline",
      profit_eur: 0,
      realized_profit_eur: 0,
      cvar_95_eur: 0,
      fill_rate: 0,
      bid_count: 24,
      regret_eur: 6175.18,
      n_runs: 1,
      status: "evaluated",
      source: "evaluations/mi00-baseline-profitguard-24-q32/run_summary.json"
    },
    {
      run_id: "verifierless-baseline-20260519:guarded",
      label: "Verifier ablation — guarded",
      kind: "ablation",
      profit_eur: null,
      realized_profit_eur: 18369.5,
      cvar_95_eur: 0,
      fill_rate: null,
      bid_count: null,
      regret_eur: null,
      n_runs: 12,
      status: "ablation mean",
      source: "evaluations/verifierless-baseline-20260519/paired_summary.json"
    }
  ]
};

const runCatalog: RunCatalogEntry[] = [
  {
    run_id: RUN_ID,
    total_steps: TOTAL_STEPS,
    trace_sha256: "mock",
    status: "completed",
    trace_path: "mock",
    setup_id: "mock",
    setup_label: "Mock demo",
    window_label: "Synthetic 96 ticks",
    start_timestamp: timestampFor(0),
    has_evaluation: true,
    pnl_eur: precomputedRun.forecaster_summary?.cumulative_pnl_eur ?? 0,
    bid_action_count: 42,
    cleared_mwh: 128,
    forecaster_id: "f8",
    control_mode: "mock"
  },
  {
    run_id: "mixed20-apr02-96-real-controls",
    total_steps: 96,
    trace_sha256: "mock-real-controls",
    status: "completed",
    trace_path: "mock",
    setup_id: "mixed20-full-days",
    setup_label: "Mixed-20 full days",
    window_label: "Apr 02 / 96 ticks / real controls",
    start_timestamp: "2026-04-02T05:30:00Z",
    has_evaluation: true,
    pnl_eur: 8505.89,
    bid_action_count: 180,
    cleared_mwh: 65.75,
    forecaster_id: "f8",
    control_mode: "real controls"
  },
  {
    run_id: "mixed20-apr02-96-proxy-controls",
    total_steps: 96,
    trace_sha256: "mock-proxy-controls",
    status: "completed",
    trace_path: "mock",
    setup_id: "mixed20-full-days",
    setup_label: "Mixed-20 full days",
    window_label: "Apr 02 / 96 ticks / proxy controls",
    start_timestamp: "2026-04-02T05:30:00Z",
    has_evaluation: true,
    pnl_eur: 25163.53,
    bid_action_count: 523,
    cleared_mwh: 220,
    forecaster_id: "f8",
    control_mode: "proxy controls"
  }
];

export function getPrecomputedRun() {
  return precomputedRun;
}

export function getRunCatalog() {
  return runCatalog;
}

export function getMarketSeries() {
  return marketSeriesCache;
}

export function getRunSnapshot(runId = RUN_ID, step = 0): RunSnapshot {
  const snapshot = snapshotCache[clampStep(step)];
  return runId === RUN_ID ? snapshot : { ...snapshot, run_id: runId };
}

export const mockRunId = RUN_ID;
export const totalMockSteps = TOTAL_STEPS;
