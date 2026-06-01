import {
  getAgentTrace as getMockAgentTrace,
  getMarketSeries as getMockMarketSeries,
  getPrecomputedRun as getMockPrecomputedRun,
  getRunCatalog as getMockRunCatalog,
  getRunSnapshot as getMockRunSnapshot,
  mockRunId,
  totalMockSteps
} from "@/lib/mock-data/generate-run";
import type { AgentHistoryResponse, AgentTemplate, AgentTrace, MarketTick, PrecomputedRun, RunCatalogEntry, RunSnapshot } from "@/types/heimdall";

const configuredRunId = process.env.NEXT_PUBLIC_HEIMDALL_RUN_ID;

export { mockRunId, totalMockSteps };

export function getRunId() {
  return configuredRunId ?? mockRunId;
}

export type RunCatalogResult = {
  runs: RunCatalogEntry[];
  usingFallbackCatalog: boolean;
};

export async function listRunCatalog(): Promise<RunCatalogResult> {
  const apiBase = getApiBase();
  if (!apiBase) {
    return { runs: getMockRunCatalog(), usingFallbackCatalog: true };
  }
  try {
    const body = await fetchJson<{ runs: RunCatalogEntry[] }>(`${apiBase}/v1/runs`);
    return { runs: body.runs, usingFallbackCatalog: false };
  } catch {
    return { runs: getMockRunCatalog(), usingFallbackCatalog: true };
  }
}

export async function listRuns(): Promise<RunCatalogEntry[]> {
  return (await listRunCatalog()).runs;
}

export async function getPrecomputedRun(runId = getRunId()): Promise<PrecomputedRun> {
  const apiBase = getApiBase();
  if (!apiBase) {
    return getMockPrecomputedRun();
  }
  try {
    return await fetchJson<PrecomputedRun>(`${apiBase}/v1/runs/${encodeURIComponent(runId)}/precomputed`);
  } catch {
    return getMockPrecomputedRun();
  }
}

export async function fetchRunSnapshot(runId = getRunId(), step = 0): Promise<RunSnapshot> {
  const apiBase = getApiBase();
  if (!apiBase) {
    return getMockRunSnapshot(runId, step);
  }
  try {
    return await fetchJson<RunSnapshot>(`${apiBase}/v1/runs/${encodeURIComponent(runId)}/society?step=${encodeURIComponent(step)}`);
  } catch {
    return getMockRunSnapshot(runId, step);
  }
}

export type SaveSocietySpecResult = { status: "saved" | "unavailable" | "error"; detail?: string };

export async function saveSocietySpec(spec: Record<string, unknown>): Promise<SaveSocietySpecResult> {
  const apiBase = getApiBase();
  if (!apiBase) {
    return { status: "unavailable", detail: "No run-view API configured; the spec is kept in this session only." };
  }
  try {
    const response = await fetch(`${apiBase}/v1/society-specs`, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(spec)
    });
    if (response.status === 503) {
      return { status: "unavailable", detail: "Run-view database is not configured, so the spec cannot be persisted." };
    }
    if (!response.ok) {
      return { status: "error", detail: `Save failed (${response.status}).` };
    }
    return { status: "saved" };
  } catch {
    return { status: "error", detail: "Network error while saving the society spec." };
  }
}

// Mirrors apps/run-view/src/heimdall_run_view/database.py:BUILTIN_AGENT_TEMPLATES so the
// template library still renders the built-ins when the API (or its database) is offline.
const BUILTIN_AGENT_TEMPLATES: AgentTemplate[] = [
  { template_id: "p2h", label: "P2H operator", category: "action", archetype: "p2h", is_builtin: true },
  { template_id: "ev", label: "EV aggregator", category: "action", archetype: "ev", is_builtin: true },
  { template_id: "wind", label: "Wind BRP", category: "action", archetype: "wind", is_builtin: true },
  { template_id: "generator", label: "Generator", category: "action", archetype: "generator", is_builtin: true },
  { template_id: "retailer", label: "Retailer", category: "action", archetype: "retailer", is_builtin: true },
  { template_id: "renewables", label: "Renewables BRP", category: "action", archetype: "renewables", is_builtin: true },
  { template_id: "risk-info", label: "Trading risk monitor", category: "information", archetype: "risk-info", is_builtin: true }
];

export type AgentTemplatesResult = {
  templates: AgentTemplate[];
  databaseAvailable: boolean;
  usingFallback: boolean;
};

export async function listAgentTemplates(): Promise<AgentTemplatesResult> {
  const apiBase = getApiBase();
  if (!apiBase) {
    return { templates: BUILTIN_AGENT_TEMPLATES, databaseAvailable: false, usingFallback: true };
  }
  try {
    const body = await fetchJson<{ templates: AgentTemplate[]; database?: string }>(`${apiBase}/v1/agent-templates`);
    return {
      templates: body.templates,
      databaseAvailable: body.database === "available",
      usingFallback: false
    };
  } catch {
    return { templates: BUILTIN_AGENT_TEMPLATES, databaseAvailable: false, usingFallback: true };
  }
}

export type SaveAgentTemplateResult = {
  status: "saved" | "unavailable" | "error";
  detail?: string;
  template?: AgentTemplate;
};

export async function saveAgentTemplate(template: AgentTemplate): Promise<SaveAgentTemplateResult> {
  const apiBase = getApiBase();
  if (!apiBase) {
    return { status: "unavailable", detail: "No run-view API configured; the template is kept in this session only." };
  }
  try {
    const response = await fetch(`${apiBase}/v1/agent-templates`, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(template)
    });
    if (response.status === 503) {
      return { status: "unavailable", detail: "Run-view database is not configured, so the template cannot be persisted." };
    }
    if (!response.ok) {
      return { status: "error", detail: `Save failed (${response.status}).` };
    }
    const body = (await response.json()) as { template: AgentTemplate };
    return { status: "saved", template: body.template };
  } catch {
    return { status: "error", detail: "Network error while saving the agent template." };
  }
}

export type DeleteAgentTemplateResult = {
  status: "deleted" | "unavailable" | "error";
  detail?: string;
};

export async function deleteAgentTemplate(templateId: string): Promise<DeleteAgentTemplateResult> {
  const apiBase = getApiBase();
  if (!apiBase) {
    return { status: "unavailable", detail: "No run-view API configured; nothing to delete." };
  }
  try {
    const response = await fetch(`${apiBase}/v1/agent-templates/${encodeURIComponent(templateId)}`, {
      method: "DELETE",
      headers: { accept: "application/json" }
    });
    if (response.status === 503) {
      return { status: "unavailable", detail: "Run-view database is not configured, so the template cannot be deleted." };
    }
    if (!response.ok) {
      return { status: "error", detail: `Delete failed (${response.status}).` };
    }
    return { status: "deleted" };
  } catch {
    return { status: "error", detail: "Network error while deleting the agent template." };
  }
}

export async function fetchAgentHistory(runId = getRunId(), agentId = "agent-p2h-focal"): Promise<AgentHistoryResponse> {
  const apiBase = getApiBase();
  if (!apiBase) {
    return mockAgentHistory(runId, agentId);
  }
  try {
    return await fetchJson<AgentHistoryResponse>(
      `${apiBase}/v1/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(agentId)}/history`
    );
  } catch {
    return mockAgentHistory(runId, agentId);
  }
}

export function getRunSnapshot(runId = mockRunId, step = 0): RunSnapshot {
  return getMockRunSnapshot(runId, step);
}

export function getMarketSeries(run?: PrecomputedRun): MarketTick[] {
  return run?.market_series ?? getMockMarketSeries();
}

export function getAgentTrace(run: PrecomputedRun | undefined, agentId = "agent-p2h-focal", step = 0): AgentTrace {
  const snapshot = run?.snapshots[Math.max(0, Math.min((run?.total_steps ?? 1) - 1, Math.round(step)))];
  if (snapshot?.selected_trace.agent_id === agentId) {
    return snapshot.selected_trace;
  }
  const node = snapshot?.nodes.find((item) => item.id === agentId);
  const trace = snapshot?.agent_traces?.[agentId];
  if (trace) {
    return trace;
  }
  if (snapshot && node) {
    return traceFromNode(snapshot, node.id);
  }
  return getMockAgentTrace(run?.run_id ?? mockRunId, agentId, step);
}

function mockAgentHistory(runId: string, agentId: string): AgentHistoryResponse {
  const run = getMockPrecomputedRun();
  const records = run.snapshots.map((snapshot) => {
    const trace = getAgentTrace(run, agentId, snapshot.step);
    return {
      run_id: runId,
      step: snapshot.step,
      timestamp: trace.timestamp,
      observed_at: trace.timestamp,
      agent_id: trace.agent_id,
      zone: "DK1",
      archetype: trace.persona.archetype,
      market_price_eur_mwh: snapshot.market.mfrr_price_eur_per_mwh,
      forecast_interval_eur_mwh: [
        trace.verifier_verdict.conformal_interval.quantile_low,
        trace.verifier_verdict.conformal_interval.quantile_high
      ] as [number, number],
      decision: {
        action: trace.verifier_verdict.accepted ? "bid" : "watch",
        side: trace.proposed_action.direction === "sell" ? "up" : "down",
        quantity_mwh: trace.proposed_action.quantity_mw,
        limit_price_eur_mwh: trace.proposed_action.price_eur_per_mwh,
        rationale: trace.reasoning
      },
      rationale: trace.reasoning,
      verifier: {
        accepted: trace.verifier_verdict.accepted,
        reason_codes: trace.verifier_verdict.retry_suggestion ? [trace.verifier_verdict.retry_suggestion] : [],
        stage_failed: trace.verifier_verdict.stage_failed
      },
      realized_outcome: trace.realized_outcome,
      tool_calls: trace.tool_calls.map((call) => ({
        name: call.label,
        arguments: {},
        ok: call.status !== "error",
        result: { summary: call.summary },
        error: call.status === "error" ? call.summary : null,
        provenance: call.provenance ?? "unknown"
      }))
    };
  });
  return {
    run_id: runId,
    agent_id: agentId,
    trace_sha256: "mock",
    total_records: records.length,
    records
  };
}

async function fetchJson<T>(url: string, attempts = 3): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url, { headers: { accept: "application/json" } });
      if (!response.ok) {
        throw new Error(`Heimdall API request failed: ${response.status}`);
      }
      return (await response.json()) as T;
    } catch (error) {
      lastError = error;
      // Brief backoff so a transient backend blip (e.g. a restart) doesn't strand
      // the dashboard on the mock fallback. Only the final failure falls through.
      if (attempt < attempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, 400 * (attempt + 1)));
      }
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Heimdall API request failed");
}

function getApiBase() {
  const configuredApiBase = process.env.NEXT_PUBLIC_HEIMDALL_API_URL?.replace(/\/$/, "");
  if (!configuredApiBase) {
    return "";
  }
  if (typeof window === "undefined") {
    return configuredApiBase;
  }
  try {
    const url = new URL(configuredApiBase);
    const appHost = window.location.hostname;
    const apiIsLoopback = ["127.0.0.1", "localhost", "0.0.0.0"].includes(url.hostname);
    const appIsLoopback = ["127.0.0.1", "localhost", "0.0.0.0"].includes(appHost);
    if (apiIsLoopback && !appIsLoopback) {
      url.hostname = appHost;
      return url.toString().replace(/\/$/, "");
    }
  } catch {
    return configuredApiBase;
  }
  return configuredApiBase;
}

function traceFromNode(snapshot: RunSnapshot, agentId: string): AgentTrace {
  const node = snapshot.nodes.find((item) => item.id === agentId) ?? snapshot.nodes[0];
  const price = snapshot.market.mfrr_price_eur_per_mwh;
  return {
    run_id: snapshot.run_id,
    step: snapshot.step,
    timestamp: snapshot.market.timestamp,
    agent_id: node.id,
    persona: node.persona,
    state: {
      soc_mwh: node.persona.storage_mwh === null ? null : node.persona.storage_mwh / 2,
      exposure_mw: node.open_position_mw,
      cash_eur: 50_000 + node.pnl_eur
    },
    reasoning: node.belief,
    tool_calls: [
      {
        id: `${node.id}-trace-unavailable-${snapshot.step}`,
        kind: "simulate",
        label: "Replay trace unavailable",
        status: "success",
        latency_ms: 0,
        summary: "This replay has no per-agent decision trace for the selected peer.",
        provenance: "unknown"
      }
    ],
    proposed_action: {
      market: "mFRR",
      direction: node.open_position_mw >= 0 ? "sell" : "buy",
      quantity_mw: Math.abs(node.open_position_mw),
      price_eur_per_mwh: price,
      delivery_quarter: snapshot.market.timestamp
    },
    verifier_verdict: {
      accepted: true,
      stage_failed: null,
      physical_violation: null,
      worst_case_profit_eur: null,
      threshold_eur: null,
      retry_suggestion: null,
      conformal_interval: {
        horizon_minutes: 15,
        quantile_low: price,
        quantile_high: price,
        alpha: 0.1
      }
    },
    realized_outcome: null
  };
}
