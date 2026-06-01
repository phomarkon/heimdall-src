export type PersonaArchetype =
  | "wind"
  | "ev"
  | "retailer"
  | "p2h"
  | "generator"
  | "arbitrageur"
  | "grid-info"
  | "outage-info"
  | "price-info"
  | "sizing-info"
  | "uncertainty-info"
  | "decision-info"
  | "risk-info";

export type AgentTemplateCategory = "action" | "information";

export interface AgentTemplate {
  template_id: string;
  label: string;
  category: AgentTemplateCategory;
  archetype: string;
  role?: string | null;
  persona?: string | null;
  risk_attitude?: RiskAttitude | null;
  forecaster_id?: string | null;
  asset?: {
    capacity_mw?: number | null;
    storage_mwh?: number | null;
  } | null;
  is_builtin?: boolean;
}

export type RiskAttitude = "averse" | "neutral" | "seeking";
export type Sophistication = "low" | "medium" | "high";
export type Market = "DA" | "ID" | "mFRR";
export type BidDirection = "buy" | "sell";
export type VerifierStage = "physical" | "conformal" | null;
export type SocietyEdgeKind = "consensus" | "broadcast";
export type ToolKind = "forecast" | "simulate" | "news" | "regulation" | "verifier";

export interface Persona {
  agent_id: string;
  display_name: string;
  archetype: PersonaArchetype;
  risk_attitude: RiskAttitude;
  sophistication: Sophistication;
  info_latency_min: number;
  capacity_mw: number;
  storage_mwh: number | null;
  llm_family: string;
  forecaster: string;
}

export interface AgentNode {
  id: string;
  persona: Persona;
  x: number;
  y: number;
  open_position_mw: number;
  pnl_eur: number;
  tick_pnl_eur?: number;
  belief: string;
  is_focal: boolean;
  verifier_acceptance_rate: number | null;
}

export interface SocietyEdge {
  id: string;
  source: string;
  target: string;
  kind: SocietyEdgeKind;
  side: "up" | "down" | null;
  direction: BidDirection | null;
  market: Market;
  strength: number;
  label: string;
  detail: string;
  started_step: number;
  expires_step: number;
}

export interface ToolCall {
  id: string;
  kind: ToolKind;
  label: string;
  status: "queued" | "running" | "success" | "error";
  latency_ms: number;
  summary: string;
  provenance?: "runner_seeded" | "llm_requested" | "forced_final" | "runner_diagnostic" | "retry" | "unknown";
}

export interface RawToolCall {
  name: string;
  arguments: Record<string, unknown>;
  ok: boolean | null;
  result: unknown;
  error: unknown;
  provenance?: "runner_seeded" | "llm_requested" | "forced_final" | "runner_diagnostic" | "retry" | "unknown";
}

export interface BidAction {
  market: Market;
  direction: BidDirection;
  quantity_mw: number;
  price_eur_per_mwh: number;
  delivery_quarter: string;
}

export interface VerifierVerdict {
  accepted: boolean;
  stage_failed: VerifierStage;
  physical_violation: Record<string, number | string> | null;
  worst_case_profit_eur: number | null;
  threshold_eur: number | null;
  retry_suggestion: string | null;
  conformal_interval: {
    horizon_minutes: number;
    quantile_low: number;
    quantile_high: number;
    alpha: number;
  };
}

export interface AgentTrace {
  run_id: string;
  step: number;
  timestamp: string;
  agent_id: string;
  persona: Persona;
  state: {
    soc_mwh: number | null;
    exposure_mw: number;
    cash_eur: number;
  };
  reasoning: string;
  tool_calls: ToolCall[];
  proposed_action: BidAction;
  verifier_verdict: VerifierVerdict;
  realized_outcome: {
    fill_mw: number;
    realized_price_eur_per_mwh: number;
    pnl_eur: number;
  } | null;
  info_digest?: InfoDigest | null;
}

export interface AgentHistoryRecord {
  run_id: string;
  step: number;
  timestamp: string;
  observed_at: string | null;
  agent_id: string;
  zone: string | null;
  archetype: PersonaArchetype;
  market_price_eur_mwh: number | null;
  forecast_interval_eur_mwh: [number | null, number | null] | null;
  decision: Record<string, unknown>;
  rationale: string;
  verifier: {
    accepted: boolean | null;
    reason_codes: string[];
    stage_failed: VerifierStage;
  };
  realized_outcome: AgentTrace["realized_outcome"];
  tool_calls: RawToolCall[];
}

export interface AgentHistoryResponse {
  run_id: string;
  agent_id: string;
  trace_sha256: string;
  total_records: number;
  records: AgentHistoryRecord[];
}

export interface InfoDigest {
  finding: string;
  confidence: number;
  importance: number;
  risk_label?: string | null;
  uncertainty_label?: string | null;
  opportunity_label?: string | null;
  watch_reasons: string[];
  direction_hint?: string | null;
  signals: Array<{ label: string; value: number | string }>;
}

export interface MarketTick {
  step: number;
  timestamp: string;
  dk1_price_eur_per_mwh: number;
  dk2_price_eur_per_mwh: number;
  mfrr_price_eur_per_mwh: number;
  imbalance_mw: number;
  gate_closure_minutes: number;
  priority_signal?: PrioritySignal;
  events: Array<{
    id: string;
    kind: "accepted_bid" | "rejected_bid" | "price_spike" | "gate_closure" | "watch" | "must_watch";
    label: string;
  }>;
}

export interface PrioritySignal {
  score: number;
  rank: number | null;
  percentile: number;
  tier: "low" | "watch" | "medium" | "high" | "critical";
  label: string;
  drivers: string[];
  risks: string[];
  grounding?: "realized_outcome" | "forward_estimate";
  components?: Record<string, number | boolean>;
}

export interface RunSnapshot {
  run_id: string;
  step: number;
  total_steps: number;
  nodes: AgentNode[];
  edges: SocietyEdge[];
  selected_trace: AgentTrace;
  agent_traces?: Record<string, AgentTrace>;
  market: MarketTick;
  forecast_diagnostics?: ForecastDiagnostics;
  health: {
    coverage: number;
    verifier_acceptance_rate: number;
    cumulative_pnl_eur: number;
    gpu_utilization: number;
    wall_time_minutes: number;
    tick_pnl_eur?: number;
    cleared_mwh?: number;
    filled_count?: number;
    bid_count?: number;
    status_counts?: Record<string, number>;
  };
}

export interface ForecastDiagnostics {
  forecaster_id: string;
  interval_low_eur_mwh: number | null;
  interval_high_eur_mwh: number | null;
  interval_width_eur_mwh: number | null;
  realized_price_eur_mwh: number | null;
  covered: boolean | null;
  spot_mfrr_spread_eur_mwh: number | null;
  up_edge_eur_mwh: number | null;
  down_edge_eur_mwh: number | null;
  expected_spread_eur_mwh: number | null;
  worst_case_profit_eur: number | null;
}

export interface ForecasterLeaderboardRow {
  model_id: string;
  label: string;
  seed_count: number | null;
  q10_pinball: string | null;
  q50_pinball: string | null;
  q90_pinball: string | null;
  mean_pinball: string | null;
  raw_coverage: string | null;
  aci_coverage: string | null;
  status: string;
}

export interface FocalBaselineRow {
  run_id: string;
  label: string;
  kind: "baseline" | "ablation";
  profit_eur: number | null;
  realized_profit_eur: number | null;
  cvar_95_eur: number | null;
  fill_rate: number | null;
  bid_count: number | null;
  regret_eur: number | null;
  n_runs: number;
  status: string;
  source: string;
}

export interface ForecasterSummary {
  active_forecaster_id: string;
  run_ids_by_forecaster: Record<string, string[]>;
  coverage: number;
  accepted_bid_rate: number;
  cumulative_pnl_eur: number;
  selected_tick_count: number;
}

export interface PrecomputedRun {
  run_id: string;
  total_steps: number;
  snapshots: RunSnapshot[];
  market_series: MarketTick[];
  priority_accuracy?: {
    score: number;
    precision: number;
    recall: number;
    profit_capture_rate: number;
    selected_tick_count: number;
    positive_tick_count: number;
  };
  forecaster_leaderboard?: ForecasterLeaderboardRow[];
  forecaster_summary?: ForecasterSummary;
  focal_baselines?: FocalBaselineRow[];
}

export interface RunCatalogEntry {
  run_id: string;
  total_steps: number;
  trace_sha256: string;
  status: string;
  trace_path: string;
  setup_id?: string;
  setup_label?: string;
  window_label?: string;
  start_timestamp?: string | null;
  has_evaluation?: boolean;
  pnl_eur?: number | null;
  bid_action_count?: number | null;
  cleared_mwh?: number | null;
  forecaster_id?: string | null;
  control_mode?: string | null;
}
