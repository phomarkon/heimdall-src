import { defineConfig, devices } from "@playwright/test";

// Port is overridable (PW_PORT) so the e2e server can run alongside a dev server
// on 3000 without clashing. The spawned server inherits the current env, so leave
// NEXT_PUBLIC_HEIMDALL_API_URL unset to exercise the mock-data fixtures the specs expect.
const port = process.env.PW_PORT ?? "3000";
const baseURL = `http://127.0.0.1:${port}`;

// Opt-in full-stack mode (HEIMDALL_E2E_FULLSTACK=1): boots the real run-view backend and
// points the frontend at it, then runs ONLY e2e/fullstack.spec.ts against real run data.
// Needs `uv` + the python env; the default mock-mode suite stays python-free.
const fullstack = Boolean(process.env.HEIMDALL_E2E_FULLSTACK);
const apiPort = process.env.HEIMDALL_E2E_API_PORT ?? "8091";
const fullstackRun = process.env.HEIMDALL_E2E_RUN ?? "msa-screen-mixed18-apr03-1430-24";

const webServer = fullstack
  ? [
      {
        command: `uv run uvicorn heimdall_run_view.service:app --host 127.0.0.1 --port ${apiPort}`,
        cwd: "../..",
        url: `http://127.0.0.1:${apiPort}/healthz`,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000
      },
      {
        command: `bun run dev -- --hostname 127.0.0.1 --port ${port}`,
        url: baseURL,
        reuseExistingServer: !process.env.CI,
        env: {
          NEXT_PUBLIC_HEIMDALL_API_URL: `http://127.0.0.1:${apiPort}`,
          NEXT_PUBLIC_HEIMDALL_RUN_ID: fullstackRun
        }
      }
    ]
  : {
      command: `bun run dev -- --hostname 127.0.0.1 --port ${port}`,
      url: baseURL,
      reuseExistingServer: !process.env.CI
    };

export default defineConfig({
  testDir: "./e2e",
  testMatch: fullstack ? "**/fullstack.spec.ts" : "**/dashboard.spec.ts",
  timeout: fullstack ? 90_000 : 30_000,
  expect: {
    timeout: 5_000
  },
  webServer,
  use: {
    baseURL,
    trace: "on-first-retry"
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    },
    {
      name: "mobile",
      use: { ...devices["Pixel 7"] }
    }
  ]
});
