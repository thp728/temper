import { expect, test } from "@playwright/test";
import { E2E_BACKEND_PORT } from "../src/lib/backend";
import { requireFakeProvider } from "./helpers";

// The zero-cost tier's honesty rule (spec 012): every page in that mode is
// marked as a demonstration, and the marking cannot be turned off in that
// mode. The journeys boot both halves with TEMPER_FAKE_PROVIDER
// (playwright.config.ts), so the banner is part of every page these tests
// render; this spec pins that it is actually there, on the routes a reviewer
// would land on, and that nothing on the page offers to dismiss it.

const backend =
  process.env.TEMPER_BACKEND_URL ?? `http://127.0.0.1:${E2E_BACKEND_PORT}`;

test.beforeAll(async ({ playwright }) => {
  await requireFakeProvider(() => playwright.request.newContext(), backend);
});

test("every page in the zero-cost mode carries the demonstration marking", async ({
  page,
}) => {
  // The banner lives in the root layout, so it rides every route: the upload
  // front door, the job list, and the calibration aggregate.
  for (const path of ["/", "/jobs", "/calibration"]) {
    await page.goto(path);
    const banner = page.getByLabel("Demonstration mode");
    await expect(banner).toBeVisible();
    await expect(banner).toContainText("Demonstration mode");
    await expect(banner).toContainText(/simulated machine/i);
    await expect(banner).toContainText(/nothing here touches/i);
  }
});

test("the marking cannot be turned off in that mode", async ({ page }) => {
  await page.goto("/");
  const banner = page.getByLabel("Demonstration mode");
  await expect(banner).toBeVisible();

  // No dismiss control of any kind: a banner a visitor can dismiss is a
  // banner that can be missing from the screenshot meant to prove the mode
  // was labelled.
  await expect(banner.getByRole("button")).toHaveCount(0);
  await expect(banner.getByRole("link")).toHaveCount(0);
});

test("the zero-cost tier starts with something to show", async ({ page }) => {
  // The seed half of spec 012: on a fresh database (the journeys boot the
  // control plane with TEMPER_DB_RESET) the tier plants a sample dataset and
  // one completed run, so the job list is never empty on first open. This
  // pins that the demonstration content itself, not just the marking,
  // cannot rot, and that it rides under the same banner.
  await page.goto("/jobs");
  const row = page.getByRole("row", { name: /sample-chat\.jsonl/ });
  await expect(row).toBeVisible();
  await expect(row).toContainText("complete");
  await expect(page.getByLabel("Demonstration mode")).toBeVisible();
});
