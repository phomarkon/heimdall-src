import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import {
  deleteAgentTemplate,
  fetchAgentHistory,
  getAgentTrace,
  getPrecomputedRun,
  listAgentTemplates,
  listRunCatalog,
  saveAgentTemplate,
  saveSocietySpec
} from "@/lib/api/run-adapter";
import { getPrecomputedRun as getMockPrecomputedRun } from "@/lib/mock-data/generate-run";

const API = "http://localhost";
const server = setupServer();

beforeAll(() => server.listen({ onUnhandledRequest: "bypass" }));
afterEach(() => {
  server.resetHandlers();
  vi.unstubAllEnvs();
});
afterAll(() => server.close());

describe("API fallback when unconfigured", () => {
  it("returns mock data and flags the fallback catalog when no API base is set", async () => {
    const result = await listRunCatalog();
    expect(result.usingFallbackCatalog).toBe(true);
    expect(result.runs.length).toBeGreaterThan(0);
    const run = await getPrecomputedRun();
    expect(run.run_id).toBe(getMockPrecomputedRun().run_id);
  });
});

describe("fetchJson retry before mock fallback", () => {
  it("retries transient failures and then succeeds against the API", async () => {
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    let calls = 0;
    server.use(
      http.get(`${API}/v1/runs/:runId/precomputed`, () => {
        calls += 1;
        if (calls < 3) {
          return new HttpResponse(null, { status: 503 });
        }
        return HttpResponse.json({ ...getMockPrecomputedRun(), run_id: "from-api-after-retry" });
      })
    );
    const run = await getPrecomputedRun("any");
    expect(calls).toBe(3);
    expect(run.run_id).toBe("from-api-after-retry");
  });

  it("falls back to mock after exhausting retries", async () => {
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    server.use(http.get(`${API}/v1/runs/:runId/precomputed`, () => new HttpResponse(null, { status: 500 })));
    const run = await getPrecomputedRun("missing");
    expect(run.run_id).toBe(getMockPrecomputedRun().run_id);
  });
});

describe("saveSocietySpec", () => {
  it("reports unavailable when no API is configured", async () => {
    const result = await saveSocietySpec({ society_id: "draft-1" });
    expect(result.status).toBe("unavailable");
  });

  it("reports saved on 200", async () => {
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    server.use(http.post(`${API}/v1/society-specs`, () => HttpResponse.json({ society_spec: { society_id: "x" } })));
    expect((await saveSocietySpec({ society_id: "x" })).status).toBe("saved");
  });

  it("reports unavailable on 503 (no database) and error on 500", async () => {
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    server.use(http.post(`${API}/v1/society-specs`, () => new HttpResponse(null, { status: 503 })));
    expect((await saveSocietySpec({ society_id: "x" })).status).toBe("unavailable");
    server.use(http.post(`${API}/v1/society-specs`, () => new HttpResponse(null, { status: 500 })));
    expect((await saveSocietySpec({ society_id: "x" })).status).toBe("error");
  });
});

describe("agent templates", () => {
  it("returns built-in templates and database-offline when no API is configured", async () => {
    const result = await listAgentTemplates();
    expect(result.databaseAvailable).toBe(false);
    expect(result.usingFallback).toBe(true);
    expect(result.templates.length).toBe(7);
    expect(result.templates.every((template) => template.is_builtin)).toBe(true);
  });

  it("flags database availability from the API response", async () => {
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    server.use(
      http.get(`${API}/v1/agent-templates`, () =>
        HttpResponse.json({
          database: "available",
          templates: [
            { template_id: "p2h", label: "P2H operator", category: "action", archetype: "p2h", is_builtin: true },
            { template_id: "custom-x", label: "Custom X", category: "action", archetype: "custom", is_builtin: false }
          ]
        })
      )
    );
    const result = await listAgentTemplates();
    expect(result.databaseAvailable).toBe(true);
    expect(result.templates.some((template) => template.template_id === "custom-x")).toBe(true);
  });

  it("falls back to built-ins (database offline) when the API errors", async () => {
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    server.use(http.get(`${API}/v1/agent-templates`, () => new HttpResponse(null, { status: 500 })));
    const result = await listAgentTemplates();
    expect(result.databaseAvailable).toBe(false);
    expect(result.templates.length).toBe(7);
  });

  it("saveAgentTemplate reports unavailable without an API and saved on 200", async () => {
    expect((await saveAgentTemplate({ template_id: "a", label: "A", category: "action", archetype: "p2h" })).status).toBe(
      "unavailable"
    );
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    server.use(
      http.post(`${API}/v1/agent-templates`, () =>
        HttpResponse.json({ template: { template_id: "a", label: "A", category: "action", archetype: "p2h", is_builtin: false } })
      )
    );
    const result = await saveAgentTemplate({ template_id: "a", label: "A", category: "action", archetype: "p2h" });
    expect(result.status).toBe("saved");
    expect(result.template?.template_id).toBe("a");
  });

  it("saveAgentTemplate maps 503 to unavailable and 500 to error", async () => {
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    server.use(http.post(`${API}/v1/agent-templates`, () => new HttpResponse(null, { status: 503 })));
    expect((await saveAgentTemplate({ template_id: "a", label: "A", category: "action", archetype: "p2h" })).status).toBe(
      "unavailable"
    );
    server.use(http.post(`${API}/v1/agent-templates`, () => new HttpResponse(null, { status: 500 })));
    expect((await saveAgentTemplate({ template_id: "a", label: "A", category: "action", archetype: "p2h" })).status).toBe(
      "error"
    );
  });

  it("deleteAgentTemplate reports unavailable without an API and deleted on 200", async () => {
    expect((await deleteAgentTemplate("a")).status).toBe("unavailable");
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    server.use(http.delete(`${API}/v1/agent-templates/:templateId`, () => HttpResponse.json({ status: "deleted", template_id: "a" })));
    expect((await deleteAgentTemplate("a")).status).toBe("deleted");
  });

  it("deleteAgentTemplate maps 503 to unavailable and 404/500 to error", async () => {
    vi.stubEnv("NEXT_PUBLIC_HEIMDALL_API_URL", API);
    server.use(http.delete(`${API}/v1/agent-templates/:templateId`, () => new HttpResponse(null, { status: 503 })));
    expect((await deleteAgentTemplate("a")).status).toBe("unavailable");
    server.use(http.delete(`${API}/v1/agent-templates/:templateId`, () => new HttpResponse(null, { status: 404 })));
    expect((await deleteAgentTemplate("a")).status).toBe("error");
  });
});

describe("getAgentTrace resolution + history fallback", () => {
  it("returns the selected trace for the focal agent", () => {
    const run = getMockPrecomputedRun();
    const trace = getAgentTrace(run, "agent-p2h-focal", 4);
    expect(trace.agent_id).toBe("agent-p2h-focal");
    expect(trace.verifier_verdict).toBeDefined();
  });

  it("synthesizes a trace for a peer without its own decision trace", () => {
    const run = getMockPrecomputedRun();
    const peer = run.snapshots[4].nodes.find((node) => !node.is_focal)!;
    const trace = getAgentTrace(run, peer.id, 4);
    expect(trace.agent_id).toBe(peer.id);
    expect(trace.tool_calls.length).toBeGreaterThan(0);
  });

  it("builds a full mock agent history when no API is configured", async () => {
    const history = await fetchAgentHistory(undefined, "agent-p2h-focal");
    expect(history.total_records).toBeGreaterThan(0);
    expect(history.records[0].agent_id).toBe("agent-p2h-focal");
    expect(history.records[0].decision).toHaveProperty("action");
  });
});
