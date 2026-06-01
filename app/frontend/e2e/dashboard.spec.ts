import { expect, test } from "@playwright/test";

test("loads the graph dashboard and activity rail", async ({ page, isMobile }) => {
  test.skip(isMobile, "The activity rail is intentionally hidden on true mobile.");
  await page.goto("/");

  await expect(page.getByText("Heimdall", { exact: true }).first()).toBeVisible();
  const graph = page.getByTestId("society-graph");
  await expect(graph).toBeVisible();
  await expect(graph.locator("canvas").first()).toBeVisible();
  await expect(page.getByTestId("activity-feed")).toContainText("Bid certified for mFRR submission");

  await page.getByLabel("Next interval").click();
  await expect(page.getByTestId("activity-feed")).toContainText("00:15");
});

test("keeps the mobile dashboard usable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  await expect(page.getByTestId("society-graph")).toBeVisible();
  await expect(page.getByLabel("Simulation timeline")).toBeVisible();
  await expect(page.getByLabel("Play replay")).toBeVisible();

  // The activity rail is hidden on phones but reachable through the drawer toggle.
  // dispatchEvent fires the DOM click directly, sidestepping the transient Next.js
  // dev-overlay portal that can intercept hit-testing in dev mode; the visibility
  // assertions still prove the drawer genuinely opens.
  await expect(page.getByTestId("activity-feed")).toBeHidden();
  await page.getByLabel("Toggle activity rail").dispatchEvent("click");
  await expect(page.getByTestId("activity-feed")).toBeVisible();
});

test("switches between live, config, and results views", async ({ page, isMobile }) => {
  test.skip(isMobile, "The full navigation rail is tested on desktop.");
  await page.goto("/");

  await page.getByLabel("Config view").click();
  await expect(page.getByRole("heading", { name: /Configuration/ })).toBeVisible();
  await expect(page.getByText("Simulation setup")).toBeVisible();
  await expect(page.getByRole("button", { name: /Save society spec/ })).toBeVisible();

  await page.getByLabel("Results view").click();
  await expect(page.getByRole("heading", { name: /Results/ })).toBeVisible();
  await expect(page.getByText("Focal policy baseline leaderboard")).toBeVisible();
  await expect(page.getByText("Verifier outcome ledger")).toBeVisible();

  await page.getByLabel("Live run view").click();
  await expect(page.getByTestId("society-graph")).toBeVisible();
});

test("runs view lists setups and selecting a run returns to the live view", async ({ page, isMobile }) => {
  test.skip(isMobile, "Navigation rail is tested on desktop.");
  await page.goto("/");

  await page.getByLabel("Runs view").click();
  await expect(page.getByTestId("runs-view")).toBeVisible();
  await expect(page.getByText("Top setups")).toBeVisible();
  await expect(page.getByText("Mixed-20 full days").first()).toBeVisible();

  await page.getByText("mixed20-apr02-96-proxy-controls").click();
  await expect(page.getByTestId("society-graph")).toBeVisible();
});

test("config view saves a society spec and surfaces a status banner", async ({ page, isMobile }) => {
  test.skip(isMobile, "Navigation rail is tested on desktop.");
  await page.goto("/");

  await page.getByLabel("Config view").click();
  await page.getByRole("button", { name: /Save society spec/ }).click();
  // Mock-mode dev server has no API configured, so the save reports unavailable.
  await expect(page.getByRole("status")).toBeVisible();
  await expect(page.getByText(/No run-view API configured/i)).toBeVisible();
});

test("help view explains how to read the dashboard", async ({ page, isMobile }) => {
  test.skip(isMobile, "Navigation rail is tested on desktop.");
  await page.goto("/");

  await page.getByLabel("Help view").click();
  await expect(page.getByTestId("help-view")).toBeVisible();
  await expect(page.getByRole("heading", { name: "How to read Heimdall" })).toBeVisible();
  await expect(page.getByText("Demo walkthrough")).toBeVisible();
});

test("live view shows the focal agent inspector card and the value-tier legend", async ({ page, isMobile }) => {
  test.skip(isMobile, "Right-rail/legend density is tested on desktop.");
  await page.goto("/");

  await expect(page.getByTestId("selected-agent-card")).toBeVisible();
  await expect(page.getByText("Top value")).toBeVisible();
  await expect(page.getByTestId("activity-feed")).toBeVisible();
});
