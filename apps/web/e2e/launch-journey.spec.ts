import { expect, type Page, test } from "@playwright/test";
import { BACKEND_PORT } from "../src/lib/backend";
import {
  chat,
  jsonl,
  pairedValue,
  requireFakeProvider,
  tempFile,
  upload,
  uploadRows,
} from "./helpers";

// The launch journey, ported into the shell (#38): choose a base model from
// the catalog, read everything the job will train with, and start it with one
// action. A good test here drives the application the way a person does --
// find a control by its accessible name, act on it, assert on what the user
// can then see.
//
// The control plane is booted with TEMPER_FAKE_PROVIDER (playwright.config.ts),
// so a launched job runs to `complete` against the app's own fake provider:
// no hardware, no credentials, no way to reach the billing account. The
// journeys still refuse to run against a backend where that did not take
// effect -- launching against the wrong process could provision real
// machines, and that is never a test failure worth risking money over.
const backendHealth =
  process.env.TEMPER_BACKEND_URL ?? `http://127.0.0.1:${BACKEND_PORT}`;

test.beforeAll(async ({ playwright }) => {
  await requireFakeProvider(() => playwright.request.newContext(), backendHealth);
});

// Stat cards and definition lists pair a visible label with a value; read
// them as pairs -- see helpers.ts.
async function uploadValidatedRows(page: Page, rows: object[]) {
  await uploadRows(page, rows);
  await expect(page.getByText("Validation passed")).toBeVisible();
}

async function continueToLaunch(page: Page) {
  await page.getByRole("link", { name: "Choose a model and continue" }).click();
  await expect(
    page.getByRole("heading", { name: "Choose a base model" }),
  ).toBeVisible();
}

test("a job is chosen, reviewed and launched from the shell", async ({
  page,
}) => {
  await uploadValidatedRows(
    page,
    Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`)),
  );
  await continueToLaunch(page);

  // The choice carries its terms: licence and pinned revision beside the
  // model they belong to, not buried behind a click.
  const qwen8b = page.getByRole("radio", { name: /Qwen\/Qwen3-8B/ });
  await expect(qwen8b).toBeVisible();
  const card = page.locator("label").filter({ has: qwen8b });
  await expect(card).toContainText("Apache-2.0");
  await expect(card.getByText(/^[a-f0-9]{40}$/)).toBeVisible();

  // The settings are visible before launching, and say they freeze...
  await expect(page.getByText(/frozen at launch/i)).toBeVisible();
  await expect(page.getByText(/cannot be changed afterwards/i)).toBeVisible();
  for (const key of ["lora_r", "lora_alpha", "learning_rate", "num_epochs"]) {
    await expect(page.getByText(key, { exact: true })).toBeVisible();
  }

  // ...and the screen offers exactly one obvious action.
  await expect(page.getByRole("button", { name: "Launch job" })).toHaveCount(1);

  await qwen8b.check();
  await page.getByRole("button", { name: "Launch job" }).click();

  // The job appears immediately afterwards, already carrying the choice.
  await expect(page).toHaveURL(/\/jobs\/job_/);
  await expect(page.getByText("qwen3-8b")).toBeVisible();

  // The watch screen is not ported yet (#39/#40): it is proxied through this
  // origin, stylesheet included -- an unstyled page is a broken page, whatever
  // its headings say.
  const sheet = await page.request.get("/static/styles.css");
  expect(sheet.status()).toBe(200);
  expect(sheet.headers()["content-type"]).toContain("text/css");

  // The fake provider runs the job to completion in moments; the watch page
  // follows it there on its own.
  await expect(pairedValue(page, "State")).toHaveText("complete", {
    timeout: 20_000,
  });
  await expect(
    page.getByRole("link", { name: /Download the adapter/ }),
  ).toBeVisible();
});

test("a feasibility warning arrives before the launch, while it can still be acted on", async ({
  page,
}) => {
  // Enough rows that the measured-throughput estimate plainly exceeds the
  // 24-hour ceiling (~34,300 rows would; 36,000 leaves margin).
  const rows = Array.from({ length: 36_000 }, (_, i) =>
    chat(`question ${i}`, `answer ${i}`),
  );
  await uploadValidatedRows(page, rows);
  await continueToLaunch(page);

  // Scoped: Next's own route announcer is also an alert.
  const warning = page.getByRole("main").getByRole("alert");
  await expect(warning).toBeVisible();
  await expect(warning).toContainText("duration_feasibility");
  await expect(warning).toContainText(/estimate/i);

  // A warning, never a refusal: the launch is still offered.
  await expect(
    page.getByRole("button", { name: "Launch job" }),
  ).toBeEnabled();
});

test("the launch screen is keyboard-operable end to end", async ({ page }) => {
  await uploadValidatedRows(
    page,
    Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`)),
  );

  // Reachable by keyboard alone: tab from the page's start until the action
  // has focus. The cap is a failure guard, not an assertion about the page.
  await continueToLaunch(page);
  const launch = page.getByRole("button", { name: "Launch job" });
  const maxTabStops = 6;
  for (
    let i = 0;
    i < maxTabStops && !(await launch.evaluate((el) => el === document.activeElement));
    i++
  ) {
    await page.keyboard.press("Tab");
  }
  await expect(launch).toBeFocused();

  // Shift+Tab back into the model group and choose with the arrow keys:
  // a radio group moves selection without leaving the keyboard.
  await page.keyboard.press("Shift+Tab");
  await page.keyboard.press("ArrowDown");
  await expect(
    page.getByRole("radio", { name: /Qwen\/Qwen3-8B/ }),
  ).toBeChecked();

  // Launch from the keyboard alone.
  await page.keyboard.press("Tab");
  await page.keyboard.press("Enter");

  await expect(page).toHaveURL(/\/jobs\/job_/);
  await expect(page.getByText("qwen3-8b")).toBeVisible();
});

test("an unusable dataset never reaches a launchable form", async ({
  page,
}) => {
  // A rejected dataset: its report blocks proceeding, and typing the launch
  // address by hand refuses with the same stable code the launch would raise.
  // An empty assistant turn is an error (empty_target), so the dataset is invalid.
  await upload(
    page,
    await tempFile(jsonl([chat("q", "   ")]), "bad.jsonl"),
  );
  await expect(page.getByText("This dataset was rejected")).toBeVisible();
  const datasetId = new URL(page.url()).pathname.split("/").pop() ?? "";

  await page.goto(`/jobs/new?dataset_id=${datasetId}`);
  await expect(page.getByText("Dataset not usable")).toBeVisible();
  await expect(page.getByText("dataset_invalid")).toBeVisible();
  await expect(page.getByRole("button", { name: "Launch job" })).toHaveCount(0);

  // An unknown dataset id refuses the same way, with its own code.
  await page.goto("/jobs/new?dataset_id=ds_nope");
  await expect(page.getByText("Dataset not usable")).toBeVisible();
  await expect(page.getByText("not_found").first()).toBeVisible();
});
