"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  Bot,
  BrainCircuit,
  ChevronDown,
  CheckCircle2,
  Clock,
  CircleHelp,
  Database,
  Eye,
  FastForward,
  ListChecks,
  Minus,
  Plus,
  RadioTower,
  Save,
  Search,
  Settings2,
  SlidersHorizontal,
  Trophy,
  X
} from "lucide-react";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { deleteAgentTemplate, getPrecomputedRun, getRunId, listAgentTemplates, listRunCatalog, saveAgentTemplate, saveSocietySpec } from "@/lib/api/run-adapter";
import { cn, formatDateTime, formatEur, formatMw, formatPrice } from "@/lib/utils";
import { usePlaybackStore, useRunPlayback } from "@/stores/run-playback";
import { ActivitySidebar } from "@/components/activity-sidebar";
import { HealthStrip } from "@/components/health-strip";
import { MarketTimeline } from "@/components/market-timeline";
import { type DashboardView, RunProgressRail } from "@/components/run-progress-rail";
import type { AgentTemplate, AgentTemplateCategory, PersonaArchetype, PrecomputedRun, RiskAttitude, RunCatalogEntry, RunSnapshot } from "@/types/heimdall";

const SocietyGraph = dynamic(() => import("@/components/society-graph").then((module) => module.SocietyGraph), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-[#fbfaf7]" aria-label="Loading society graph" />
});

export function AppShell() {
  const playback = useRunPlayback();
  const setTotalSteps = usePlaybackStore((state) => state.setTotalSteps);
  const resetForRun = usePlaybackStore((state) => state.resetForRun);
  const [activeView, setActiveView] = useState<DashboardView>("live");
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState(getRunId());
  const { data: catalogResult, isLoading: catalogLoading } = useQuery({
    queryKey: ["run-catalog"],
    queryFn: listRunCatalog
  });
  const catalog = catalogResult?.runs ?? [];
  const { data: run, isLoading: runLoading } = useQuery({
    queryKey: ["precomputed-run", selectedRunId],
    queryFn: () => getPrecomputedRun(selectedRunId)
  });
  useEffect(() => {
    if (run) {
      setTotalSteps(run.total_steps);
    }
  }, [run, setTotalSteps]);
  const snapshot = run?.snapshots[playback.step];

  if (!snapshot) {
    const runEmpty = !runLoading && !!run && run.snapshots.length === 0;
    const runMissing = !runLoading && !run;
    const heading = runLoading
      ? "Loading run data"
      : runEmpty
        ? "Run has no snapshots"
        : runMissing
          ? "Run unavailable"
          : "Preparing dashboard";
    const body = runEmpty
      ? `Run ${selectedRunId} returned zero intervals. Pick another run from the catalog.`
      : runMissing
        ? `Could not load ${selectedRunId} from the run-view API, and no demo fallback was available.`
        : `Fetching ${selectedRunId}. If the run-view API is unavailable, the dashboard falls back to local demo data.`;
    return (
      <main className="retro-noise flex h-screen w-screen items-center justify-center bg-[#fbfaf7] text-[#101217]">
        <section className="border-2 border-[#1f2933] bg-white p-6 text-center shadow-[6px_6px_0_#1f2933]">
          <div className="mx-auto mb-4 flex h-11 w-11 items-center justify-center border border-[#1f2933] bg-[#dff5ec]">
            {runEmpty || runMissing ? (
              <AlertTriangle className="h-5 w-5 text-[#b45309]" aria-hidden="true" />
            ) : (
              <RadioTower className="h-5 w-5 text-[#0f766e]" aria-hidden="true" />
            )}
          </div>
          <p className="retro-label text-slate-500">Heimdall</p>
          <h1 className="mt-2 text-xl font-black uppercase text-slate-950">{heading}</h1>
          <p className="mt-2 max-w-sm text-sm font-semibold text-slate-600">{body}</p>
        </section>
      </main>
    );
  }

  return (
    <main className="retro-noise grid h-screen w-screen grid-cols-[72px_minmax(0,1fr)_420px] grid-rows-[72px_minmax(0,1fr)_148px] overflow-hidden bg-[#fbfaf7] text-[#101217] max-xl:grid-cols-[64px_minmax(0,1fr)_360px] max-md:grid-cols-[56px_minmax(0,1fr)_300px] max-md:grid-rows-[72px_minmax(0,1fr)_156px] max-sm:grid-cols-[56px_minmax(0,1fr)]">
      <header className="col-span-3 flex items-center justify-between gap-4 border-b-2 border-[#1f2933] bg-[#fbfaf7] px-4 max-sm:col-span-2">
        <div className="flex min-w-0 shrink items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center border border-[#1f2933] bg-[#dff5ec]">
            <RadioTower className="h-5 w-5 text-[#0f766e]" aria-hidden="true" />
          </div>
          <div>
            <p className="text-xl font-black uppercase leading-none tracking-[-0.03em]">Heimdall</p>
            <p className="truncate text-xs font-semibold tracking-[0.04em] text-slate-600">DK1/DK2 verifier-guarded market simulation</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <MarketTicker snapshot={snapshot} />
          <button
            type="button"
            aria-label="Toggle activity rail"
            aria-expanded={mobileSidebarOpen}
            className="hidden h-10 w-10 items-center justify-center border border-[#1f2933] bg-white text-slate-700 transition hover:bg-[#dff5ec] max-sm:flex"
            onClick={() => setMobileSidebarOpen((open) => !open)}
          >
            <ListChecks className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>
      </header>

      <RunProgressRail snapshot={snapshot} activeView={activeView} onViewChange={setActiveView} />

      <section className="relative col-start-2 row-start-2 min-h-0 overflow-hidden border-r-2 border-[#1f2933] bg-[#fbfaf7] max-sm:border-r-0">
        {activeView === "live" ? (
          <>
            <SocietyGraph snapshot={snapshot} />
            <HealthStrip snapshot={snapshot} />
          </>
        ) : activeView === "config" ? (
          <ConfigView snapshot={snapshot} />
        ) : activeView === "runs" ? (
          <RunsView
            runs={catalog}
            selectedRunId={selectedRunId}
            catalogLoading={catalogLoading}
            usingFallbackCatalog={catalogResult?.usingFallbackCatalog ?? false}
            onSelectRun={(runId) => {
              setSelectedRunId(runId);
              resetForRun();
              setActiveView("live");
            }}
          />
        ) : activeView === "results" ? (
          <ResultsView run={run} snapshot={snapshot} />
        ) : (
          <HelpView />
        )}
      </section>

      {mobileSidebarOpen ? (
        <button
          type="button"
          aria-label="Close activity rail"
          className="fixed inset-0 z-40 hidden bg-black/30 max-sm:block"
          onClick={() => setMobileSidebarOpen(false)}
        />
      ) : null}

      <aside
        className={cn(
          "col-start-3 row-start-2 row-end-4 min-h-0 overflow-hidden border-l-2 border-[#1f2933] bg-[#fbfaf7]",
          mobileSidebarOpen
            ? "max-sm:fixed max-sm:inset-y-0 max-sm:right-0 max-sm:z-50 max-sm:w-[88%] max-sm:max-w-sm max-sm:shadow-[-6px_0_0_rgba(31,41,51,0.12)]"
            : "max-sm:hidden"
        )}
      >
        {mobileSidebarOpen ? (
          <button
            type="button"
            aria-label="Close activity rail"
            className="absolute right-2 top-2 z-10 flex h-8 w-8 items-center justify-center border border-[#1f2933] bg-white text-slate-700 max-sm:flex sm:hidden"
            onClick={() => setMobileSidebarOpen(false)}
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        ) : null}
        <ActivitySidebar run={run} snapshot={snapshot} />
      </aside>

      <footer className="col-span-2 col-start-1 row-start-3 border-t-2 border-[#1f2933] bg-[#fbfaf7] max-sm:col-span-2">
        <MarketTimeline snapshot={snapshot} run={run} />
      </footer>
    </main>
  );
}

const agentTypeOptions: Array<{ id: PersonaArchetype; label: string; description: string }> = [
  { id: "wind", label: "Wind BRPs", description: "Forecast-driven renewable imbalance and mFRR exposure." },
  { id: "ev", label: "EV aggregators", description: "Flexible load with charge windows and fleet constraints." },
  { id: "retailer", label: "Retailers", description: "Demand-serving BRPs with price and customer-risk pressure." },
  { id: "p2h", label: "P2H operators", description: "Power-to-heat assets with storage and thermal feasibility." },
  { id: "generator", label: "Generators", description: "Dispatchable supply with ramp and opportunity-cost logic." },
  { id: "arbitrageur", label: "Arbitrageurs", description: "Fast bilateral traders hunting spreads and hedges." },
  { id: "grid-info", label: "Grid info", description: "Grid constraint analysts providing network context." },
  { id: "outage-info", label: "Outage info", description: "Outage impact specialists scoring availability context." },
  { id: "price-info", label: "Price info", description: "Limit-price specialists evaluating clearing prices." },
  { id: "sizing-info", label: "Sizing info", description: "Candidate sizing specialists shaping bid quantities." },
  { id: "uncertainty-info", label: "Uncertainty info", description: "Auditors tracking forecast and side ambiguity." },
  { id: "decision-info", label: "Decision info", description: "Decision auditors reviewing action consistency." },
  { id: "risk-info", label: "Risk info", description: "Trading risk monitors checking downside and side risk." }
];

const modelOptions = [
  { value: "qwen-32b", label: "Qwen 32B" },
  { value: "qwen-72b", label: "Qwen 72B" }
];

const customCategoryOptions = [
  { value: "action", label: "Action agent" },
  { value: "information", label: "Information agent" }
];

const customArchetypeOptions = [
  ...agentTypeOptions.map((option) => ({ value: option.id, label: option.label })),
  { value: "custom", label: "Custom (free-form)" }
];

const riskAttitudeOptions = [
  { value: "averse", label: "Risk averse" },
  { value: "neutral", label: "Risk neutral" },
  { value: "seeking", label: "Risk seeking" }
];

// F-codes from the chapter-03 forecaster zoo. Empty value = inherit the base archetype's default.
const forecasterOptions = [
  { value: "", label: "Inherit from archetype" },
  { value: "F1", label: "F1 — quantile LightGBM" },
  { value: "F7", label: "F7 — Patch-TST + split-CP" },
  { value: "F8", label: "F8 — rich Patch-TST (focal default)" },
  { value: "F9", label: "F9 — TimesFM-2.0" },
  { value: "F10", label: "F10 — Chronos-Bolt" },
  { value: "F11", label: "F11 — PriceFM surrogate" }
];

type CustomAgentDraft = {
  label: string;
  archetype: string;
  category: AgentTemplateCategory;
  role: string;
  persona: string;
  riskAttitude: RiskAttitude;
  forecasterId: string;
  capacityMw: string;
  storageMwh: string;
};

const emptyCustomAgentDraft: CustomAgentDraft = {
  label: "",
  archetype: "p2h",
  category: "action",
  role: "",
  persona: "",
  riskAttitude: "neutral",
  forecasterId: "",
  capacityMw: "",
  storageMwh: ""
};

function slugify(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function draftToTemplate(draft: CustomAgentDraft, templateId: string): AgentTemplate {
  const capacity = draft.capacityMw.trim();
  const storage = draft.storageMwh.trim();
  const asset =
    capacity === "" && storage === ""
      ? null
      : {
          capacity_mw: capacity === "" ? null : Number(capacity),
          storage_mwh: storage === "" ? null : Number(storage)
        };
  return {
    template_id: templateId,
    label: draft.label.trim(),
    category: draft.category,
    archetype: draft.archetype,
    role: draft.role.trim() || null,
    persona: draft.persona.trim() || null,
    risk_attitude: draft.riskAttitude,
    forecaster_id: draft.forecasterId || null,
    asset
  };
}

const runFilters: Array<{ value: keyof RunFilters; label: string }> = [
  { value: "fullDay", label: "Full day" },
  { value: "screen", label: "Screen" },
  { value: "real", label: "Real controls" },
  { value: "proxy", label: "Proxy controls" }
];

type RunFilters = {
  fullDay: boolean;
  screen: boolean;
  real: boolean;
  proxy: boolean;
};

const emptyRunFilters: RunFilters = {
  fullDay: false,
  screen: false,
  real: false,
  proxy: false
};

export function ConfigView({ snapshot }: { snapshot: RunSnapshot }) {
  const initialCounts = Object.fromEntries(
    agentTypeOptions.map((option) => [
      option.id,
      snapshot.nodes.filter((node) => node.persona.archetype === option.id).length
    ])
  ) as Record<PersonaArchetype, number>;
  const [agentCount, setAgentCount] = useState(snapshot.nodes.length);
  const [model, setModel] = useState("qwen-32b");
  const [startTime, setStartTime] = useState(toDateTimeLocalValue(snapshot.market.timestamp));
  const [tickCount, setTickCount] = useState(snapshot.total_steps);
  const [enabledTypes, setEnabledTypes] = useState<Record<PersonaArchetype, boolean>>(
    Object.fromEntries(agentTypeOptions.map((option) => [option.id, initialCounts[option.id] > 0])) as Record<PersonaArchetype, boolean>
  );

  const selectedTypes = agentTypeOptions.filter((option) => enabledTypes[option.id]);
  const composition = buildComposition(agentCount, selectedTypes.map((option) => option.id), initialCounts);
  const selectedModelLabel = modelOptions.find((option) => option.value === model)?.label ?? "Qwen 32B";
  const endTime = calculateEndTime(startTime, tickCount);

  const queryClient = useQueryClient();
  const { data: templatesResult } = useQuery({ queryKey: ["agent-templates"], queryFn: listAgentTemplates });
  const databaseAvailable = templatesResult?.databaseAvailable ?? false;
  const customTemplates = (templatesResult?.templates ?? []).filter((template) => !template.is_builtin);
  const [customCounts, setCustomCounts] = useState<Record<string, number>>({});
  const [draft, setDraft] = useState<CustomAgentDraft>(emptyCustomAgentDraft);
  const draftTemplateId = slugify(draft.label);

  const createTemplate = useMutation({
    mutationFn: () => saveAgentTemplate(draftToTemplate(draft, draftTemplateId)),
    onSuccess: (result) => {
      if (result.status === "saved") {
        queryClient.invalidateQueries({ queryKey: ["agent-templates"] });
        setCustomCounts((current) => ({ ...current, [draftTemplateId]: Math.max(1, current[draftTemplateId] ?? 0) }));
        setDraft(emptyCustomAgentDraft);
      }
    }
  });
  const createResult = createTemplate.data;
  const deleteTemplate = useMutation({
    mutationFn: (templateId: string) => deleteAgentTemplate(templateId),
    onSuccess: (result, templateId) => {
      if (result.status === "deleted") {
        queryClient.invalidateQueries({ queryKey: ["agent-templates"] });
        setCustomCounts((current) => {
          const next = { ...current };
          delete next[templateId];
          return next;
        });
      }
    }
  });
  const deleteResult = deleteTemplate.data;
  const customAgents = customTemplates
    .map((template) => ({ template_id: template.template_id, count: customCounts[template.template_id] ?? 0 }))
    .filter((entry) => entry.count > 0);
  const customAgentTotal = customAgents.reduce((sum, entry) => sum + entry.count, 0);

  const saveMutation = useMutation({
    mutationFn: () =>
      saveSocietySpec({
        society_id: `draft-${model}-${agentCount}a-${tickCount}t`,
        label: `${agentCount} agents / ${selectedModelLabel} / ${tickCount} intervals`,
        agent_count: agentCount,
        model,
        start_time: startTime,
        tick_count: tickCount,
        agent_types: selectedTypes.map((option) => option.id),
        composition,
        custom_agents: customAgents
      })
  });
  const saveResult = saveMutation.data;

  return (
    <div className="retro-noise h-full overflow-y-auto bg-[#fbfaf7] px-8 py-6">
      <div className="mb-6 flex items-center justify-between gap-4">
        <div>
          <p className="retro-label text-slate-500">Run setup</p>
          <h1 className="mt-1 text-2xl font-black uppercase text-slate-950">[Configuration]</h1>
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => saveMutation.mutate()}
            disabled={saveMutation.isPending || selectedTypes.length === 0}
            className="flex h-10 items-center gap-2 border border-[#1f2933] bg-[#dff5ec] px-3 text-xs font-black uppercase tracking-[0.08em] text-teal-800 transition hover:bg-[#fff0c7] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Save className="h-4 w-4" aria-hidden="true" />
            {saveMutation.isPending ? "Saving..." : "Save society spec"}
          </button>
          <div className="flex h-10 w-10 items-center justify-center border border-[#1f2933] bg-[#dff5ec] text-teal-700">
            <Settings2 className="h-5 w-5" aria-hidden="true" />
          </div>
        </div>
      </div>

      {saveResult ? (
        <div
          role="status"
          className={cn(
            "mb-4 border p-3 text-sm font-semibold",
            saveResult.status === "saved"
              ? "border-[#27b7a4] bg-[#dff5ec] text-teal-800"
              : saveResult.status === "unavailable"
                ? "border-[#d39b14] bg-[#fff0c7] text-slate-800"
                : "border-[#ff6542] bg-[#ffe7df] text-rose-700"
          )}
        >
          {saveResult.status === "saved"
            ? "Society spec saved to the run-view store."
            : saveResult.detail ?? "Could not save the society spec."}
        </div>
      ) : null}

      <section className="retro-panel mt-5 p-5">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-950">
          <SlidersHorizontal className="h-4 w-4 text-teal-700" aria-hidden="true" />
          Simulation setup
        </h2>
        <div className="mt-4 grid grid-cols-4 gap-4 text-sm max-2xl:grid-cols-2 max-lg:grid-cols-1">
          <label className="block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
            <span className="retro-label block text-slate-500">Agent amount</span>
            <input
              aria-label="Agent amount"
              className="tabular-stable mt-2 h-10 w-full border border-[#1f2933] bg-white px-3 text-base font-black text-slate-950"
              min={1}
              max={250}
              type="number"
              value={agentCount}
              onChange={(event) => setAgentCount(clampNumber(Number(event.target.value), 1, 250))}
            />
          </label>
          <div className="block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
            <span className="retro-label block text-slate-500">Society model</span>
            <RetroSelect
              label="Society model"
              options={modelOptions}
              value={model}
              onChange={setModel}
            />
          </div>
          <label className="block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
            <span className="retro-label block text-slate-500">Start time</span>
            <input
              aria-label="Start time"
              className="tabular-stable mt-2 h-10 w-full border border-[#1f2933] bg-white px-3 text-sm font-black text-slate-950"
              type="datetime-local"
              value={startTime}
              onChange={(event) => setStartTime(event.target.value)}
            />
          </label>
          <div className="grid grid-cols-[92px_minmax(0,1fr)] gap-3 border border-[#cfd5dd] bg-[#f7f6f2] p-3">
            <div>
              <span className="retro-label block text-slate-500">Run length</span>
              <label>
                <span className="sr-only">Simulation intervals</span>
                <input
                  aria-label="Simulation intervals"
                  className="tabular-stable mt-2 h-10 w-full border border-[#1f2933] bg-white px-3 text-base font-black text-slate-950"
                  min={1}
                  max={672}
                  type="number"
                  value={tickCount}
                  onChange={(event) => setTickCount(clampNumber(Number(event.target.value), 1, 672))}
                />
              </label>
            </div>
            <div className="min-w-0">
              <span className="retro-label block text-slate-500">End time</span>
              <div className="tabular-stable mt-2 flex h-10 min-w-0 items-center border border-[#cfd5dd] bg-white px-3 font-black text-slate-950">
                <span className="truncate">{endTime}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-3 gap-3 text-sm max-xl:grid-cols-2 max-lg:grid-cols-1">
          <ConfigRow label="Scenario" value="DK1/DK2 post-break mFRR" />
          <ConfigRow label="Verifier" value="physical + conformal profit" />
          <ConfigRow label="Draft run" value={`${agentCount} agents / ${selectedModelLabel} / ${tickCount} intervals`} />
        </div>
      </section>

      <section className="retro-panel mt-5 p-5">
        <h2 className="text-sm font-semibold text-slate-950">Agent types</h2>
        <div className="mt-4 grid grid-cols-2 gap-3 text-sm max-xl:grid-cols-1">
          {agentTypeOptions.map((option) => (
            <label
              key={option.id}
              className="grid cursor-pointer grid-cols-[20px_1fr] gap-3 border border-[#cfd5dd] bg-white p-3 transition hover:bg-[#f7f6f2]"
            >
              <input
                aria-label={option.label}
                checked={enabledTypes[option.id]}
                className="mt-1 h-4 w-4 accent-[#0f8f7e]"
                type="checkbox"
                onChange={() =>
                  setEnabledTypes((current) => ({
                    ...current,
                    [option.id]: !current[option.id]
                  }))
                }
              />
              <span>
                <span className="block font-black text-slate-950">{option.label}</span>
                <span className="mt-1 block text-xs leading-5 text-slate-600">{option.description}</span>
              </span>
            </label>
          ))}
        </div>
      </section>

      <section className="retro-panel mt-5 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-950">
            <Bot className="h-4 w-4 text-teal-700" aria-hidden="true" />
            Custom agents
          </h2>
          <span
            className={cn(
              "flex items-center gap-2 border px-2 py-1 text-xs font-black uppercase tracking-[0.08em]",
              databaseAvailable ? "border-[#27b7a4] bg-[#dff5ec] text-teal-800" : "border-[#d39b14] bg-[#fff0c7] text-amber-800"
            )}
          >
            <span className={cn("h-2 w-2 rounded-full", databaseAvailable ? "bg-teal-500" : "bg-amber-500")} aria-hidden="true" />
            {databaseAvailable ? "Database connected" : "Database offline"}
          </span>
        </div>
        <p className="mt-2 text-xs leading-5 text-slate-600">
          Design a bespoke persona on top of the built-in archetypes. Saved agents persist in the run-view database and can be added to the society below.
        </p>

        {!databaseAvailable ? (
          <div className="mt-3 border border-[#d39b14] bg-[#fff0c7] p-3 text-sm font-semibold text-slate-800">
            The run-view database is offline, so custom agents can be designed but not saved. Start Postgres and the run-view API to persist them.
          </div>
        ) : null}

        {createResult && createResult.status !== "saved" ? (
          <div role="alert" className="mt-3 border border-[#ff6542] bg-[#ffe7df] p-3 text-sm font-semibold text-rose-700">
            {createResult.detail ?? "Could not save the agent template."}
          </div>
        ) : null}

        {deleteResult && deleteResult.status !== "deleted" ? (
          <div role="alert" className="mt-3 border border-[#ff6542] bg-[#ffe7df] p-3 text-sm font-semibold text-rose-700">
            {deleteResult.detail ?? "Could not delete the agent template."}
          </div>
        ) : null}

        <div className="mt-4 grid grid-cols-2 gap-4 text-sm max-lg:grid-cols-1">
          <label className="block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
            <span className="retro-label block text-slate-500">Agent name</span>
            <input
              aria-label="Custom agent name"
              className="mt-2 h-10 w-full border border-[#1f2933] bg-white px-3 text-sm font-black text-slate-950"
              placeholder="e.g. Aggressive P2H"
              value={draft.label}
              onChange={(event) => setDraft((current) => ({ ...current, label: event.target.value }))}
            />
            <span className="mt-1 block text-xs text-slate-400">id: {draftTemplateId || "—"}</span>
          </label>
          <div className="grid grid-cols-2 gap-3 max-sm:grid-cols-1">
            <div className="block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
              <span className="retro-label block text-slate-500">Category</span>
              <RetroSelect
                label="Custom agent category"
                options={customCategoryOptions}
                value={draft.category}
                onChange={(value) => setDraft((current) => ({ ...current, category: value as AgentTemplateCategory }))}
              />
            </div>
            <div className="block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
              <span className="retro-label block text-slate-500">Base archetype</span>
              <RetroSelect
                label="Custom agent archetype"
                options={customArchetypeOptions}
                value={draft.archetype}
                onChange={(value) => setDraft((current) => ({ ...current, archetype: value }))}
              />
            </div>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-3 text-sm max-sm:grid-cols-1">
          <div className="block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
            <span className="retro-label block text-slate-500">Risk attitude</span>
            <RetroSelect
              label="Custom agent risk attitude"
              options={riskAttitudeOptions}
              value={draft.riskAttitude}
              onChange={(value) => setDraft((current) => ({ ...current, riskAttitude: value as RiskAttitude }))}
            />
          </div>
          <div className="block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
            <span className="retro-label block text-slate-500">Forecaster</span>
            <RetroSelect
              label="Custom agent forecaster"
              options={forecasterOptions}
              value={draft.forecasterId}
              onChange={(value) => setDraft((current) => ({ ...current, forecasterId: value }))}
            />
          </div>
        </div>

        <div className="mt-3 grid grid-cols-[minmax(0,1fr)_140px_140px] gap-3 text-sm max-sm:grid-cols-1">
          <label className="block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
            <span className="retro-label block text-slate-500">Role (optional)</span>
            <input
              aria-label="Custom agent role"
              className="mt-2 h-10 w-full border border-[#1f2933] bg-white px-3 text-sm font-black text-slate-950"
              placeholder="e.g. action_agent"
              value={draft.role}
              onChange={(event) => setDraft((current) => ({ ...current, role: event.target.value }))}
            />
          </label>
          <label className="block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
            <span className="retro-label block text-slate-500">Capacity MW</span>
            <input
              aria-label="Custom agent capacity in MW"
              className="tabular-stable mt-2 h-10 w-full border border-[#1f2933] bg-white px-3 text-sm font-black text-slate-950"
              type="number"
              min={0}
              placeholder="—"
              value={draft.capacityMw}
              onChange={(event) => setDraft((current) => ({ ...current, capacityMw: event.target.value }))}
            />
          </label>
          <label className="block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
            <span className="retro-label block text-slate-500">Storage MWh</span>
            <input
              aria-label="Custom agent storage in MWh"
              className="tabular-stable mt-2 h-10 w-full border border-[#1f2933] bg-white px-3 text-sm font-black text-slate-950"
              type="number"
              min={0}
              placeholder="—"
              value={draft.storageMwh}
              onChange={(event) => setDraft((current) => ({ ...current, storageMwh: event.target.value }))}
            />
          </label>
        </div>

        <label className="mt-3 block border border-[#cfd5dd] bg-[#f7f6f2] p-3">
          <span className="retro-label block text-slate-500">Persona / system prompt</span>
          <textarea
            aria-label="Custom agent persona"
            className="mt-2 w-full resize-y border border-[#1f2933] bg-white px-3 py-2 text-sm font-medium leading-6 text-slate-900"
            rows={3}
            placeholder="Describe how this agent reasons, its risk appetite, and what it optimises for."
            value={draft.persona}
            onChange={(event) => setDraft((current) => ({ ...current, persona: event.target.value }))}
          />
        </label>

        <div className="mt-3 flex justify-end">
          <button
            type="button"
            onClick={() => createTemplate.mutate()}
            disabled={!databaseAvailable || !draftTemplateId || createTemplate.isPending}
            className="flex h-10 items-center gap-2 border border-[#1f2933] bg-[#dff5ec] px-3 text-xs font-black uppercase tracking-[0.08em] text-teal-800 transition hover:bg-[#fff0c7] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Plus className="h-4 w-4" aria-hidden="true" />
            {createTemplate.isPending ? "Saving..." : "Save custom agent"}
          </button>
        </div>

        <div className="mt-5">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-xs font-black uppercase tracking-[0.08em] text-slate-600">Saved custom agents</h3>
            <span className="tabular-stable text-xs text-slate-500">{customAgentTotal} in society</span>
          </div>
          {customTemplates.length > 0 ? (
            <div className="mt-3 grid grid-cols-2 gap-3 max-xl:grid-cols-1">
              {customTemplates.map((template) => {
                const count = customCounts[template.template_id] ?? 0;
                return (
                  <div key={template.template_id} className="flex items-center justify-between gap-3 border border-[#1f2933] bg-white p-3">
                    <div className="min-w-0">
                      <span className="block truncate font-black text-slate-950">{template.label}</span>
                      <span className="block truncate text-xs uppercase tracking-[0.06em] text-slate-500">
                        {template.archetype} / {template.category}
                        {template.risk_attitude ? ` / ${template.risk_attitude}` : ""}
                        {template.forecaster_id ? ` / ${template.forecaster_id}` : ""}
                      </span>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      <CountStepperButton label={`Remove one ${template.label}`} onClick={() => adjustCustomCount(setCustomCounts, template.template_id, -1)} disabled={count <= 0}>
                        <Minus className="h-4 w-4" aria-hidden="true" />
                      </CountStepperButton>
                      <span className="tabular-stable w-8 text-center text-base font-black text-slate-950">{count}</span>
                      <CountStepperButton label={`Add one ${template.label}`} onClick={() => adjustCustomCount(setCustomCounts, template.template_id, 1)}>
                        <Plus className="h-4 w-4" aria-hidden="true" />
                      </CountStepperButton>
                      <span className="mx-1 h-6 w-px bg-[#cfd5dd]" aria-hidden="true" />
                      <button
                        type="button"
                        aria-label={`Delete ${template.label}`}
                        title={databaseAvailable ? `Delete ${template.label}` : "Database offline — cannot delete"}
                        onClick={() => deleteTemplate.mutate(template.template_id)}
                        disabled={!databaseAvailable || deleteTemplate.isPending}
                        className="flex h-8 w-8 items-center justify-center border border-[#1f2933] bg-white text-slate-700 transition hover:bg-[#ffe7df] hover:text-rose-700 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        <X className="h-4 w-4" aria-hidden="true" />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="mt-3 border border-[#cfd5dd] bg-[#f7f6f2] p-3 text-sm text-slate-600">
              No custom agents yet. Define one above to add bespoke personas to the society.
            </p>
          )}
        </div>
      </section>

      <section className="retro-panel mt-5 p-5">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-slate-950">Society composition</h2>
          <span className="tabular-stable text-xs text-slate-500">{composition.reduce((sum, item) => sum + item.count, 0)} / {agentCount} agents</span>
        </div>
        {composition.length > 0 ? (
          <div className="mt-4 grid grid-cols-3 gap-3 text-sm max-xl:grid-cols-2">
            {composition.map((item) => (
              <ConfigRow key={item.id} label={item.label} value={`${item.count} agents`} />
            ))}
          </div>
        ) : (
          <p className="mt-4 border border-[#d39b14] bg-[#fff0c7] p-3 text-sm text-slate-800">
            Select at least one agent type to build a society.
          </p>
        )}
      </section>
    </div>
  );
}

export function RunsView({
  runs,
  selectedRunId,
  catalogLoading = false,
  usingFallbackCatalog = false,
  onSelectRun
}: {
  runs: RunCatalogEntry[];
  selectedRunId: string;
  catalogLoading?: boolean;
  usingFallbackCatalog?: boolean;
  onSelectRun: (runId: string) => void;
}) {
  const [selectedSetupId, setSelectedSetupId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [filters, setFilters] = useState<RunFilters>(emptyRunFilters);
  const filteredRuns = runs.filter((run) => runMatchesFilter(run, filters, search));
  const setups = groupRunsBySetup(filteredRuns);
  const topSetups = groupRunsBySetup(runs).filter((setup) => setup.bestPnl !== null).sort((a, b) => (b.bestPnl ?? -Infinity) - (a.bestPnl ?? -Infinity)).slice(0, 5);
  const activeSetupId = selectedSetupId && setups.some((setup) => setup.id === selectedSetupId)
    ? selectedSetupId
    : setups[0]?.id;
  const setupRuns = filteredRuns
    .filter((run) => (run.setup_id ?? "standalone") === activeSetupId)
    .sort((a, b) => String(a.window_label ?? a.run_id).localeCompare(String(b.window_label ?? b.run_id)));
  const evaluatedCount = runs.filter((run) => run.has_evaluation).length;
  const traceOnlyCount = Math.max(0, runs.length - evaluatedCount);

  return (
    <div className="retro-noise h-full select-none overflow-y-auto bg-[#fbfaf7] px-8 py-6" data-testid="runs-view">
      <div className="mb-6 flex items-center justify-between gap-4">
        <div>
          <p className="retro-label text-slate-500">Run catalog</p>
          <h1 className="mt-1 text-2xl font-black uppercase text-slate-950">[Runs]</h1>
        </div>
        <div className="flex h-10 w-10 items-center justify-center border border-[#1f2933] bg-[#eef3ff] text-teal-700">
          <Database className="h-5 w-5" aria-hidden="true" />
        </div>
      </div>

      {usingFallbackCatalog ? (
        <div className="mb-4 border border-[#d39b14] bg-[#fff0c7] p-3 text-sm font-semibold text-slate-800">
          Showing fallback catalog data because the run catalog API is unavailable.
        </div>
      ) : null}

      {catalogLoading ? (
        <div className="mb-4 border border-[#cfd5dd] bg-white p-3 text-sm font-semibold text-slate-700">
          Loading run catalog...
        </div>
      ) : null}

      <section className="retro-panel p-5">
        <div className="mb-5">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h2 className="text-sm font-black text-slate-950">Top setups</h2>
            <span className="retro-label text-slate-500">{runs.length} runs / {evaluatedCount} evaluated / {traceOnlyCount} trace-only</span>
          </div>
          <div className="grid grid-cols-5 gap-3 max-2xl:grid-cols-3 max-lg:grid-cols-2">
            {topSetups.map((setup) => (
              <button
                key={setup.id}
                type="button"
                className={`min-h-28 border border-[#1f2933] p-3 text-left transition ${
                  setup.id === activeSetupId ? "bg-[#dff5ec]" : "bg-white hover:bg-[#eef3ff]"
                }`}
                onClick={() => setSelectedSetupId(setup.id)}
              >
                <span className="block text-sm font-black text-slate-950">{setup.label}</span>
                <span className="mt-2 block text-xl font-black text-teal-800">{formatOptionalEur(setup.bestPnl)}</span>
                <span className="mt-2 block text-xs font-semibold uppercase tracking-[0.08em] text-slate-500">{setup.count} runs / {setup.evaluatedCount} evaluated</span>
                <span className="mt-1 block truncate text-xs text-slate-600">{setup.bestWindowLabel ?? setup.bestRunId ?? "No evaluated window"}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-[minmax(220px,0.36fr)_minmax(0,1fr)] gap-5 max-xl:grid-cols-1">
          <div>
            <label className="block">
              <span className="retro-label text-slate-500">Search</span>
              <span className="mt-2 flex h-10 items-center gap-2 border border-[#1f2933] bg-white px-3">
                <Search className="h-4 w-4 text-slate-500" aria-hidden="true" />
                <input
                  aria-label="Search runs"
                  className="min-w-0 flex-1 select-text bg-transparent text-sm font-semibold text-slate-950 outline-none"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
              </span>
            </label>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              {runFilters.map((item) => (
                <button
                  key={item.value}
                  type="button"
                  className={`border border-[#1f2933] px-2 py-2 font-black uppercase transition ${
                    filters[item.value] ? "bg-[#dff5ec] text-teal-800" : "bg-white text-slate-600 hover:bg-[#eef3ff]"
                  }`}
                  onClick={() => setFilters((current) => ({ ...current, [item.value]: !current[item.value] }))}
                >
                  {item.label}
                </button>
              ))}
            </div>

            <div className="mt-4 space-y-2">
              {setups.map((setup) => (
                <button
                  key={setup.id}
                  type="button"
                  className={`block w-full border border-[#1f2933] p-3 text-left transition ${
                    setup.id === activeSetupId ? "bg-[#dff5ec]" : "bg-white hover:bg-[#f7f6f2]"
                  }`}
                  onClick={() => setSelectedSetupId(setup.id)}
                >
                  <span className="block font-black text-slate-950">{setup.label}</span>
                  <span className="mt-1 block text-xs font-semibold uppercase tracking-[0.08em] text-slate-500">
                    {setup.count} runs / {setup.evaluatedCount} evaluated / best {formatOptionalEur(setup.bestPnl)}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div className="min-w-0">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h2 className="text-sm font-black text-slate-950">{setups.find((setup) => setup.id === activeSetupId)?.label ?? "No runs"}</h2>
              <span className="retro-label text-slate-500">{setupRuns.length} windows</span>
            </div>
            <div className="overflow-hidden border border-[#1f2933] bg-white">
              <div className="grid grid-cols-[1.25fr_0.45fr_0.75fr_0.55fr_0.7fr_0.55fr_0.55fr_0.7fr] gap-3 border-b border-[#1f2933] bg-[#eef3ff] px-3 py-2 text-xs font-black uppercase tracking-[0.12em] text-slate-700">
                <span>Window</span>
                <span>Ticks</span>
                <span>Controls</span>
                <span>Fcst</span>
                <span>P&L</span>
                <span>Bids</span>
                <span>MWh</span>
                <span>Status</span>
              </div>
              {setupRuns.map((run) => {
                const active = run.run_id === selectedRunId;
                return (
                  <button
                    key={run.run_id}
                    type="button"
                    className={`grid w-full grid-cols-[1.25fr_0.45fr_0.75fr_0.55fr_0.7fr_0.55fr_0.55fr_0.7fr] gap-3 border-t border-[#cfd5dd] px-3 py-3 text-left text-sm transition ${
                      active ? "bg-[#dff5ec]" : "bg-white hover:bg-[#f7f6f2]"
                    }`}
                    onClick={() => onSelectRun(run.run_id)}
                  >
                    <span className="min-w-0">
                      <span className="block truncate font-semibold text-slate-950">{run.window_label ?? run.run_id}</span>
                      <span className="mt-1 block truncate text-xs text-slate-500">{run.run_id}</span>
                    </span>
                    <span>{run.total_steps}</span>
                    <span>{run.control_mode ?? "—"}</span>
                    <span>{run.forecaster_id?.toUpperCase() ?? "—"}</span>
                    <span>{formatOptionalEur(run.pnl_eur)}</span>
                    <span>{run.bid_action_count ?? "—"}</span>
                    <span>{formatOptionalNumber(run.cleared_mwh)}</span>
                    <span className={run.has_evaluation ? "font-semibold text-teal-700" : "font-semibold text-amber-700"}>
                      {active ? "loaded" : run.has_evaluation ? "ready" : "trace-only"}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export function ResultsView({ run, snapshot }: { run: PrecomputedRun; snapshot: RunSnapshot }) {
  const accepted = run.snapshots.filter((item) => item.selected_trace.verifier_verdict.accepted).length;
  const verifierAcceptance = accepted / run.total_steps;
  const finalPnl = run.snapshots[run.snapshots.length - 1]?.health.cumulative_pnl_eur ?? snapshot.health.cumulative_pnl_eur;
  const maxExposure = Math.max(...snapshot.nodes.map((node) => Math.abs(node.open_position_mw)));
  const forecasterRows = run.forecaster_leaderboard?.length ? run.forecaster_leaderboard : [];
  const diagnostics = snapshot.forecast_diagnostics;
  const focalBaselines = run.focal_baselines ?? [];

  return (
    <div className="retro-noise h-full overflow-y-auto bg-[#fbfaf7] px-8 py-6">
      <div className="mb-6">
        <p className="retro-label text-slate-500">Run report</p>
        <h1 className="mt-1 text-2xl font-black uppercase text-slate-950">[Results]</h1>
      </div>

      <section className="grid grid-cols-4 gap-3 max-xl:grid-cols-2">
        <ResultCard label="Run P&L" value={formatEur(finalPnl)} />
        <ResultCard label="Verifier accept rate" value={`${(verifierAcceptance * 100).toFixed(1)}%`} />
        <ResultCard label="Empirical coverage" value={`${((run.forecaster_summary?.coverage ?? snapshot.health.coverage) * 100).toFixed(1)}%`} />
        <ResultCard label="Max open position" value={formatMw(maxExposure)} />
      </section>

      <section className="retro-panel mt-5 p-5">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-950">
          <Trophy className="h-4 w-4 text-teal-700" aria-hidden="true" />
          ML forecaster leaderboard
        </h2>
        <div className="mt-3 grid grid-cols-4 gap-3 text-sm max-xl:grid-cols-2">
          <ConfigRow label="Active forecaster" value={run.forecaster_summary?.active_forecaster_id?.toUpperCase() ?? diagnostics?.forecaster_id.toUpperCase() ?? "unavailable"} />
          <ConfigRow label="Selected ticks" value={`${run.forecaster_summary?.selected_tick_count ?? run.priority_accuracy?.selected_tick_count ?? 0}`} />
          <ConfigRow label="Accepted bids" value={`${((run.forecaster_summary?.accepted_bid_rate ?? snapshot.health.verifier_acceptance_rate) * 100).toFixed(1)}%`} />
          <ConfigRow label="Coverage target" value="90.0%" />
        </div>
        <div className="mt-4 overflow-hidden border border-[#1f2933] bg-white">
          <div className="grid grid-cols-[0.9fr_1.35fr_0.55fr_0.85fr_0.85fr_0.85fr_0.9fr_0.9fr] gap-3 border-b border-[#1f2933] bg-[#eef3ff] px-3 py-2 text-xs font-black uppercase tracking-[0.12em] text-slate-700">
            <span>Model</span>
            <span>Family</span>
            <span>Seeds</span>
            <span>Q10</span>
            <span>Q50</span>
            <span>Q90</span>
            <span>Mean</span>
            <span>ACI cov</span>
          </div>
          {forecasterRows.map((row) => (
            <div key={row.model_id} className="grid grid-cols-[0.9fr_1.35fr_0.55fr_0.85fr_0.85fr_0.85fr_0.9fr_0.9fr] gap-3 border-t border-[#cfd5dd] px-3 py-3 text-sm">
              <span>
                <span className="block font-semibold text-slate-950">{row.model_id}</span>
                <span className={row.status.includes("missing") ? "mt-1 block text-xs text-amber-700" : "mt-1 block text-xs text-slate-500"}>{row.status}</span>
              </span>
              <span className="font-medium text-slate-900">{row.label}</span>
              <span>{row.seed_count ?? "—"}</span>
              <span>{row.q10_pinball ?? "—"}</span>
              <span>{row.q50_pinball ?? "—"}</span>
              <span>{row.q90_pinball ?? "—"}</span>
              <span>{row.mean_pinball ?? "—"}</span>
              <span>{row.aci_coverage ?? "—"}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="retro-panel mt-5 p-5">
        <h2 className="text-sm font-semibold text-slate-950">Selected-tick forecast diagnostics</h2>
        <div className="mt-4 grid grid-cols-4 gap-3 text-sm max-xl:grid-cols-2">
          <ConfigRow label="Interval" value={formatInterval(diagnostics)} />
          <ConfigRow label="Realized price" value={formatNullablePrice(diagnostics?.realized_price_eur_mwh)} />
          <ConfigRow label="Coverage" value={diagnostics?.covered === null || diagnostics?.covered === undefined ? "unavailable" : diagnostics.covered ? "covered" : "missed"} />
          <ConfigRow label="Interval width" value={formatNullablePrice(diagnostics?.interval_width_eur_mwh)} />
          <ConfigRow label="Spot/mFRR spread" value={formatNullablePrice(diagnostics?.spot_mfrr_spread_eur_mwh)} />
          <ConfigRow label="Up edge" value={formatNullablePrice(diagnostics?.up_edge_eur_mwh)} />
          <ConfigRow label="Down edge" value={formatNullablePrice(diagnostics?.down_edge_eur_mwh)} />
          <ConfigRow label="Worst-case profit" value={diagnostics?.worst_case_profit_eur === null || diagnostics?.worst_case_profit_eur === undefined ? "unavailable" : formatEur(diagnostics.worst_case_profit_eur)} />
        </div>
        <div className="mt-4 h-4 overflow-hidden border border-[#1f2933] bg-[#f7f6f2]">
          <div
            className={diagnostics?.covered ? "h-full bg-[#0f8f7e]" : "h-full bg-[#d39b14]"}
            style={{ width: `${forecastBandWidth(diagnostics?.interval_width_eur_mwh)}%` }}
            title="Relative interval width"
          />
        </div>
      </section>

      <section className="retro-panel mt-5 p-5">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-950">
          <Trophy className="h-4 w-4 text-teal-700" aria-hidden="true" />
          Focal policy baseline leaderboard
        </h2>
        <p className="mt-1 text-xs text-slate-500">
          Built only from evaluated baseline runs on disk; no figures are hand-entered. Empty when no baseline evaluations are present.
        </p>
        {focalBaselines.length ? (
          <div className="mt-4 overflow-hidden border border-[#1f2933] bg-white">
            <div className="grid grid-cols-[1.6fr_0.9fr_0.9fr_0.9fr_0.7fr_0.5fr] gap-3 border-b border-[#1f2933] bg-[#eef3ff] px-3 py-2 text-xs font-black uppercase tracking-[0.12em] text-slate-700">
              <span>Method</span>
              <span>P&L</span>
              <span>Realized</span>
              <span>CVaR(95%)</span>
              <span>Fill</span>
              <span>n</span>
            </div>
            {focalBaselines.map((row) => (
              <div key={row.run_id} className="grid grid-cols-[1.6fr_0.9fr_0.9fr_0.9fr_0.7fr_0.5fr] gap-3 border-t border-[#cfd5dd] px-3 py-3 text-sm">
                <span className="min-w-0">
                  <span className="block truncate font-semibold text-slate-950">{row.label}</span>
                  <span className={row.kind === "ablation" ? "mt-1 block text-xs text-amber-700" : "mt-1 block text-xs text-slate-500"}>{row.status}</span>
                </span>
                <span className="font-medium text-slate-900">{formatOptionalEur(row.profit_eur)}</span>
                <span>{formatOptionalEur(row.realized_profit_eur)}</span>
                <span>{formatOptionalEur(row.cvar_95_eur)}</span>
                <span>{row.fill_rate === null ? "—" : `${(row.fill_rate * 100).toFixed(0)}%`}</span>
                <span>{row.n_runs}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-4 border border-[#cfd5dd] bg-[#f7f6f2] p-3 text-sm text-slate-600">
            No baseline evaluations found in <span className="font-mono">evaluations/*baseline*</span>.
          </p>
        )}
      </section>

      <section className="retro-panel mt-5 p-5">
        <h2 className="text-sm font-semibold text-slate-950">Verifier outcome ledger</h2>
        <p className="mt-1 text-xs text-slate-500">
          Accepted actions must pass physical feasibility and conformal worst-case-profit checks; rejected actions return structured retry guidance.
        </p>
        <div className="mt-4 space-y-2">
          {run.snapshots.slice(0, 8).map((item) => (
            <div key={item.step} className="grid grid-cols-[90px_1fr_96px] items-center gap-3 border border-[#cfd5dd] bg-[#f7f6f2] px-3 py-2 text-sm">
              <span className="text-slate-500">{formatDateTime(item.market.timestamp)}</span>
              <span className="truncate text-slate-700">{item.selected_trace.proposed_action.direction} {formatMw(item.selected_trace.proposed_action.quantity_mw)} at {formatPrice(item.selected_trace.proposed_action.price_eur_per_mwh)}</span>
              <span className={item.selected_trace.verifier_verdict.accepted ? "flex items-center gap-1 font-medium text-teal-700" : "flex items-center gap-1 font-medium text-rose-600"}>
                <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
                {item.selected_trace.verifier_verdict.accepted ? "accepted" : "rejected"}
              </span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function adjustCustomCount(
  setCustomCounts: React.Dispatch<React.SetStateAction<Record<string, number>>>,
  templateId: string,
  delta: number
) {
  setCustomCounts((current) => ({
    ...current,
    [templateId]: Math.max(0, (current[templateId] ?? 0) + delta)
  }));
}

function CountStepperButton({
  label,
  onClick,
  disabled = false,
  children
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      disabled={disabled}
      className="flex h-8 w-8 items-center justify-center border border-[#1f2933] bg-white text-slate-700 transition hover:bg-[#dff5ec] hover:text-teal-700 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {children}
    </button>
  );
}

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-[#cfd5dd] bg-[#f7f6f2] p-3">
      <p className="retro-label text-slate-500">{label}</p>
      <p className="mt-1 font-medium text-slate-900">{value}</p>
    </div>
  );
}

function formatNullablePrice(value: number | null | undefined) {
  return value === null || value === undefined ? "unavailable" : formatPrice(value);
}

function formatInterval(diagnostics: RunSnapshot["forecast_diagnostics"]) {
  if (!diagnostics || diagnostics.interval_low_eur_mwh === null || diagnostics.interval_high_eur_mwh === null) {
    return "unavailable";
  }
  return `${formatPrice(diagnostics.interval_low_eur_mwh)} to ${formatPrice(diagnostics.interval_high_eur_mwh)}`;
}

function forecastBandWidth(width: number | null | undefined) {
  if (width === null || width === undefined || !Number.isFinite(width)) {
    return 0;
  }
  return Math.max(6, Math.min(100, Math.round(width)));
}

function toDateTimeLocalValue(timestamp: string) {
  return timestamp.slice(0, 16);
}

function calculateEndTime(startTime: string, tickCount: number) {
  const start = new Date(startTime);
  if (Number.isNaN(start.getTime())) {
    return "Invalid start time";
  }
  const end = new Date(start.getTime() + Math.max(0, tickCount - 1) * 15 * 60 * 1000);
  return `${end.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric"
  })}, ${end.toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  })}`;
}

function clampNumber(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, Math.round(value)));
}

function buildComposition(
  agentCount: number,
  selectedTypes: PersonaArchetype[],
  initialCounts: Record<PersonaArchetype, number>
) {
  if (selectedTypes.length === 0) {
    return [];
  }

  const initialTotal = selectedTypes.reduce((sum, type) => sum + initialCounts[type], 0);
  const rawCounts =
    initialTotal > 0
      ? selectedTypes.map((type) => ({
          id: type,
          count: (initialCounts[type] / initialTotal) * agentCount
        }))
      : selectedTypes.map((type) => ({ id: type, count: agentCount / selectedTypes.length }));
  const floored = rawCounts.map((item) => ({
    id: item.id,
    count: Math.floor(item.count),
    remainder: item.count - Math.floor(item.count)
  }));
  let remaining = agentCount - floored.reduce((sum, item) => sum + item.count, 0);
  for (const item of [...floored].sort((a, b) => b.remainder - a.remainder)) {
    if (remaining <= 0) {
      break;
    }
    item.count += 1;
    remaining -= 1;
  }

  return floored.map((item) => ({
    id: item.id,
    label: agentTypeOptions.find((option) => option.id === item.id)?.label ?? item.id,
    count: item.count
  }));
}

function groupRunsBySetup(runs: RunCatalogEntry[]) {
  const grouped = new Map<
    string,
    {
      id: string;
      label: string;
      count: number;
      evaluatedCount: number;
      bestPnl: number | null;
      bestRunId: string | null;
      bestWindowLabel: string | null;
    }
  >();
  for (const run of runs) {
    const id = run.setup_id ?? "standalone";
    const current = grouped.get(id);
    const pnl = typeof run.pnl_eur === "number" ? run.pnl_eur : null;
    if (!current) {
      grouped.set(id, {
        id,
        label: run.setup_label ?? id,
        count: 1,
        evaluatedCount: run.has_evaluation ? 1 : 0,
        bestPnl: pnl,
        bestRunId: pnl === null ? null : run.run_id,
        bestWindowLabel: pnl === null ? null : run.window_label ?? run.run_id
      });
    } else {
      current.count += 1;
      current.evaluatedCount += run.has_evaluation ? 1 : 0;
      if (pnl !== null && (current.bestPnl === null || pnl > current.bestPnl)) {
        current.bestPnl = pnl;
        current.bestRunId = run.run_id;
        current.bestWindowLabel = run.window_label ?? run.run_id;
      }
    }
  }
  return [...grouped.values()].sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}

function runMatchesFilter(
  run: RunCatalogEntry,
  filters: RunFilters,
  search: string
) {
  const haystack = [
    run.run_id,
    run.setup_id,
    run.setup_label,
    run.window_label,
    run.control_mode,
    run.forecaster_id
  ].join(" ").toLowerCase();
  const needle = search.trim().toLowerCase();
  if (needle && !haystack.includes(needle)) {
    return false;
  }
  const durationFilters = [filters.fullDay, filters.screen];
  const controlFilters = [filters.real, filters.proxy];
  const fullDay = run.total_steps >= 96 || haystack.includes("full");
  const screen = run.total_steps < 96 || haystack.includes("screen");
  const real = haystack.includes("real controls");
  const proxy = haystack.includes("proxy controls");
  if (durationFilters.some(Boolean) && !((filters.fullDay && fullDay) || (filters.screen && screen))) {
    return false;
  }
  if (controlFilters.some(Boolean) && !((filters.real && real) || (filters.proxy && proxy))) {
    return false;
  }
  return true;
}

function formatOptionalEur(value: number | null | undefined) {
  return typeof value === "number" ? formatEur(value) : "—";
}

function formatOptionalNumber(value: number | null | undefined) {
  return typeof value === "number" ? `${Number.isInteger(value) ? value : value.toFixed(2)}` : "—";
}

function ResultCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="retro-panel p-4">
      <p className="retro-label text-slate-500">{label}</p>
      <p className="mt-2 text-xl font-black text-slate-950">{value}</p>
    </div>
  );
}

export function MarketTicker({ snapshot }: { snapshot: RunSnapshot }) {
  const items = [
    {
      icon: Clock,
      label: "UTC",
      value: formatDateTime(snapshot.market.timestamp),
      className: "bg-[#f7f6f2]",
      valueClassName: "w-[176px]"
    },
    {
      icon: Activity,
      label: "mFRR",
      value: formatPrice(snapshot.market.mfrr_price_eur_per_mwh),
      className: "bg-[#eaf6f2]",
      valueClassName: "w-[116px]"
    },
    {
      icon: CheckCircle2,
      label: "Filled",
      value: `${snapshot.health.filled_count ?? 0}/${snapshot.health.bid_count ?? snapshot.nodes.length}`,
      className: "bg-[#eef3ff]",
      valueClassName: "w-[78px]"
    },
    {
      icon: AlertTriangle,
      label: "mFRR gate",
      value: `${snapshot.market.gate_closure_minutes} min`,
      className: snapshot.market.gate_closure_minutes <= 15 ? "bg-[#fff0c7]" : "bg-[#f7f6f2]",
      valueClassName: "w-[76px]"
    },
    {
      icon: FastForward,
      label: "Tick P&L",
      value: formatEur(snapshot.health.tick_pnl_eur ?? 0),
      className: (snapshot.health.tick_pnl_eur ?? 0) < 0 ? "bg-[#ffe7df]" : "bg-[#dff5ec]",
      valueClassName: "w-[92px]"
    },
    {
      icon: BrainCircuit,
      label: "Run P&L",
      value: formatEur(snapshot.health.cumulative_pnl_eur),
      className:
        snapshot.health.cumulative_pnl_eur < 0
          ? "bg-[#ffe7df]"
          : snapshot.health.cumulative_pnl_eur > 0
            ? "bg-[#dff5ec]"
            : "bg-[#f4eef8]",
      iconClassName:
        snapshot.health.cumulative_pnl_eur < 0
          ? "text-[#d22f2f]"
          : snapshot.health.cumulative_pnl_eur > 0
            ? "text-[#0f8f7e]"
            : "text-[#0f8f7e]",
      valueClassName:
        snapshot.health.cumulative_pnl_eur < 0
          ? "w-[92px] text-slate-900"
          : snapshot.health.cumulative_pnl_eur > 0
            ? "w-[92px] text-slate-900"
            : "w-[92px] text-slate-900"
    }
  ];

  return (
    <div
      className="flex h-[50px] min-w-0 shrink-0 items-stretch overflow-hidden border border-[#cfd5dd] bg-white text-xs max-md:hidden"
      aria-label="Market status ticker"
      data-testid="market-ticker"
    >
      {items.map((item, index) => {
        const Icon = item.icon;
        return (
          <div
            key={item.label}
            className={`flex min-w-0 items-center gap-2 px-3 ${item.className} ${index > 0 ? "border-l border-[#cfd5dd]" : ""}`}
          >
            <Icon className={`h-3.5 w-3.5 shrink-0 ${item.iconClassName ?? "text-[#0f8f7e]"}`} aria-hidden="true" />
            <div className="min-w-0">
              <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-500">{item.label}</div>
              <div
                className={`tabular-stable mt-0.5 truncate text-base font-black leading-none ${item.valueClassName}`}
                title={item.value}
              >
                {item.value}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function RetroSelect({
  label,
  options,
  value,
  onChange,
  className = "mt-2 h-10 w-full text-base",
  placement = "bottom"
}: {
  label: string;
  options: Array<{ value: string | number; label: string }>;
  value: string | number;
  onChange: (value: string) => void;
  className?: string;
  placement?: "top" | "bottom";
}) {
  const [open, setOpen] = useState(false);
  const selected = options.find((option) => String(option.value) === String(value)) ?? options[0];

  return (
    <div className={`relative ${className}`}>
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={label}
        className="flex h-full w-full items-center justify-between border border-[#1f2933] bg-white px-3 font-black text-slate-950 transition hover:bg-[#f7f6f2]"
        onClick={() => setOpen((current) => !current)}
      >
        <span className="truncate">{selected.label}</span>
        <ChevronDown className="h-4 w-4 shrink-0 text-slate-500" aria-hidden="true" />
      </button>
      {open ? (
        <div
          role="listbox"
          aria-label={`${label} options`}
          className={`absolute left-0 right-0 z-50 border border-[#1f2933] bg-white shadow-[4px_4px_0_rgb(31_41_51_/_0.12)] ${
            placement === "top" ? "bottom-[calc(100%+4px)]" : "top-[calc(100%+4px)]"
          }`}
        >
          {options.map((option) => {
            const active = String(option.value) === String(value);
            return (
              <button
                key={option.value}
                type="button"
                role="option"
                aria-selected={active}
                className={`block w-full border-b border-[#cfd5dd] px-3 py-2 text-left font-black last:border-b-0 ${
                  active ? "bg-[#dff5ec] text-teal-800" : "bg-white text-slate-900 hover:bg-[#eef3ff]"
                }`}
                onClick={() => {
                  onChange(String(option.value));
                  setOpen(false);
                }}
              >
                {option.label}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

export function HelpView() {
  const guideCards = [
    {
      icon: RadioTower,
      title: "What this shows",
      body: "Heimdall is a verifier-guarded LLM society of balance-responsible-party agents for the DK1/DK2 15-minute mFRR market. The focal P2H market maker proposes bids only after physical and conformal profit checks."
    },
    {
      icon: Eye,
      title: "Reading the graph",
      body: "Node color is archetype, node size is open position, the white ring marks focus/selection, and edges show bilateral negotiations such as PROPOSE, COUNTER, ACCEPT, REJECT, or WITHDRAW."
    },
    {
      icon: FastForward,
      title: "Replay controls",
      body: "Use play, pause, interval step, speed, and the scrubber to move through 15-minute market intervals. Event ticks mark bid decisions, rejected actions, price spikes, and gate-closure moments."
    },
    {
      icon: ListChecks,
      title: "Right rail",
      body: "The activity rail is the model-chat trace: agent tool calls, verifier decisions, OTC trades, and market events. Click entries or graph edges to inspect the related agent or negotiation thread."
    }
  ];

  const walkthrough = [
    "Start on Live run and identify the focal P2H node with the white ring.",
    "Scrub to a later interval and point out how accepted/rejected bids change the activity feed.",
    "Open Config to show market timing, verifier assumptions, frozen seeds, and society composition.",
    "Open Results to compare the focal policy against the baseline leaderboard."
  ];

  return (
    <div className="retro-noise h-full overflow-y-auto bg-[#fbfaf7] px-8 py-6" data-testid="help-view">
      <div className="mb-6 flex items-center justify-between gap-4">
        <div>
          <p className="retro-label text-slate-500">Guide</p>
          <h1 className="mt-1 text-2xl font-black text-slate-950">How to read Heimdall</h1>
        </div>
        <div className="flex h-10 w-10 items-center justify-center border border-[#1f2933] bg-[#eef3ff] text-slate-700">
          <CircleHelp className="h-5 w-5" aria-hidden="true" />
        </div>
      </div>

      <section className="grid grid-cols-2 gap-3 max-xl:grid-cols-1">
        {guideCards.map((card) => {
          const Icon = card.icon;
          return (
            <article key={card.title} className="retro-panel bg-white p-5">
              <h2 className="flex items-center gap-2 text-sm font-black text-slate-950">
                <Icon className="h-4 w-4 text-teal-700" aria-hidden="true" />
                {card.title}
              </h2>
              <p className="mt-3 text-sm leading-6 text-slate-700">{card.body}</p>
            </article>
          );
        })}
      </section>

      <section className="retro-panel mt-5 bg-white p-5">
        <h2 className="text-sm font-black text-slate-950">Demo walkthrough</h2>
        <ol className="mt-4 grid gap-2 text-sm text-slate-700">
          {walkthrough.map((step, index) => (
            <li key={step} className="grid grid-cols-[36px_1fr] items-start gap-3 border border-[#cfd5dd] bg-[#f7f6f2] p-3">
              <span className="tabular-stable font-black text-teal-700">{String(index + 1).padStart(2, "0")}</span>
              <span>{step}</span>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
