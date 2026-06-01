# Frontend test suite

A full testing pyramid: type safety → unit → component → end-to-end, with coverage
reporting. All layers run green and are CI-ready.

| Layer | Tool | Command | Count |
|---|---|---|---|
| Types / lint | `tsc --noEmit` | `bun run lint` | strict, 0 errors |
| Unit + component | Vitest + Testing Library + MSW (jsdom) | `bun run test` | 57 tests / 14 files |
| Coverage | `@vitest/coverage-v8` | `bun run test:coverage` | ~91% stmts |
| Integration (FE↔BE contract) | pytest + Vitest | see below | 1 + 3 tests |
| End-to-end | Playwright (Desktop Chrome + Pixel 7) | `bun run test:e2e` | 8 tests × projects |
| Full-stack e2e (opt-in) | Playwright + real run-view | `HEIMDALL_E2E_FULLSTACK=1 bun run test:e2e` | 1 test |

## What each layer covers

**Unit** (`src/lib`, `src/stores`): `utils` formatters; `theme` maps; the `run-playback`
store (clamping, the play interval with fake timers, reset) and `selection` store; the
`run-adapter` API boundary, mock fallback, **fetchJson retry-before-fallback**,
`saveSocietySpec` (saved / unavailable / error), and trace/history resolution, all behind
**MSW**; the mock-run generator.

**Component** (`src/components`): `AppShell` navigation + market ticker + **empty/error state**;
`ConfigView` form + **save flow (saved/unavailable/error)**; `RunsView` grouping/filters;
`ResultsView` leaderboards + diagnostics; `HelpView`; `HealthStrip`; `RunProgressRail`;
`MarketTimeline` scrub + value-tier legend + speed; `SocietyGraph` agent card, **edge card**,
**info-agent digest + consensus strip**, history drawer, and the **DOM fallback** when WebGL is
unavailable; `ActivitySidebar` feed derivation.

**Integration (FE↔BE contract)**, the seam the other layers stub out:
- `apps/run-view/tests/test_frontend_contract.py` (pytest): builds a realistic run (bids +
  realized outcomes + broadcast comm + a baseline eval) and asserts every endpoint
  (`/precomputed`, `/society`, `/agents/history`, `/runs`) emits exactly the fields/types the
  frontend types require, including `edges` (consensus+broadcast), `focal_baselines`, and
  `priority_signal.grounding`. Run with `HEIMDALL_WRITE_CONTRACT_FIXTURES=1` to (re)write the
  fixtures below.
- `src/lib/api/contract.integration.test.tsx` (vitest): loads the **real backend-produced**
  fixtures in `src/test-fixtures/` (`precomputed.contract.json`, `agent-history.contract.json`),
  drives them through the real adapter and renders MarketTimeline / ResultsView / ActivitySidebar /
  SocietyGraph, proving backend output parses and renders end to end.

**E2E** (`e2e/dashboard.spec.ts`): graph + activity rail load and replay scrub; mobile drawer;
view switching (Config/Results/Live); Runs-view selection; Config save banner; Help view; focal
inspector card + value-tier legend.

**Full-stack e2e** (`e2e/fullstack.spec.ts`, opt-in): boots the real `run-view` (uvicorn) + the
frontend wired to it and asserts real run data renders (graph, value-tier timeline, data-driven
baselines). Enable with `HEIMDALL_E2E_FULLSTACK=1` (needs `uv`); overridable via
`HEIMDALL_E2E_API_PORT` / `HEIMDALL_E2E_RUN`. The default suite stays python-free.

## Running e2e

Needs `bun` and Playwright browsers (`bun run playwright install --with-deps chromium`). The
config honours `PW_PORT` (default 3000) so e2e can run on a free port next to a live dev server:

```bash
PW_PORT=3100 bun run test:e2e
```

The spawned server inherits the environment, leave `NEXT_PUBLIC_HEIMDALL_API_URL` unset so the
specs exercise the mock-data fixtures.

## Conventions

- Pure logic is unit-tested directly; React pieces use Testing Library with role/label queries.
- Network is mocked with MSW at the `fetch` boundary, never by stubbing internals.
- Sigma (WebGL) is mocked; the DOM fallback path is asserted separately.
- Tests are deterministic (fake timers for intervals; frozen mock fixtures).
