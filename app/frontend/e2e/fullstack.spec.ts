import { expect, test } from "@playwright/test";

// Full-stack integration: the real run-view backend + the frontend, over the wire, on real
// run data. Enabled only when HEIMDALL_E2E_FULLSTACK=1 (needs uv + the python env); the
// playwright config then boots uvicorn + next dev wired together and runs only this file.
const enabled = Boolean(process.env.HEIMDALL_E2E_FULLSTACK);

test.describe("full stack (run-view + frontend, real data)", () => {
  test.skip(!enabled, "Set HEIMDALL_E2E_FULLSTACK=1 to run the wired backend+frontend suite.");

  test("renders real run data: graph, value-tier timeline and data-driven baselines", async ({ page, isMobile }) => {
    test.skip(isMobile, "Right rail / full nav density is asserted on desktop.");
    await page.goto("/");

    // graph loads from the live backend (real precomputed payload can take a moment)
    const graph = page.getByTestId("society-graph");
    await expect(graph).toBeVisible({ timeout: 30_000 });

    // the realized-value timeline legend is present (real priority signal)
    await expect(page.getByText("Top value")).toBeVisible({ timeout: 30_000 });

    // activity rail is populated from real traces
    await expect(page.getByTestId("activity-feed")).toBeVisible();

    // Results view shows the data-driven baseline leaderboard from real evaluations
    await page.getByLabel("Results view").click();
    await expect(page.getByRole("heading", { name: /Results/ })).toBeVisible();
    await expect(page.getByText("Focal policy baseline leaderboard")).toBeVisible();
    await expect(page.getByText(/Profit-guard baseline|Verifier ablation/).first()).toBeVisible();
  });
});
