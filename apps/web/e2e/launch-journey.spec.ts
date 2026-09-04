import { expect, type Page, test } from "@playwright/test";
import { E2E_BACKEND_PORT } from "../src/lib/backend";
import {
  chat,
  goToLaunch,
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
  process.env.TEMPER_BACKEND_URL ?? `http://127.0.0.1:${E2E_BACKEND_PORT}`;

test.beforeAll(async ({ playwright }) => {
  await requireFakeProvider(() => playwright.request.newContext(), backendHealth);
});

// Stat cards and definition lists pair a visible label with a value; read
// them as pairs -- see helpers.ts.
async function uploadValidatedRows(
  page: Page,
  rows: object[],
  // Validation is asynchronous since #31 and streams the whole file, so how
  // long the "Ready" badge takes is a function of the row count, not a
  // constant. Twelve rows land inside Playwright's 5s default; the 36,000-row
  // feasibility case does not, and failed on a loaded CI runner while passing
  // on a quiet one from the same commit. Scale the wait with the input rather
  // than raising the default for every journey.
  timeout = 5_000,
) {
  await uploadRows(page, rows);
  await expect(page.getByText("Ready")).toBeVisible({ timeout });
}

async function continueToLaunch(page: Page) {
  await goToLaunch(page);
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

  // The settings are visible before launching, and say they freeze... The
  // advanced-settings disclosure (issue #80) also names the same fields, so
  // read the specification section's own copy.
  await expect(page.getByText(/frozen at launch/i)).toBeVisible();
  await expect(page.getByText(/cannot be changed afterwards/i)).toBeVisible();
  const spec = page.getByLabel("The job specification");
  for (const key of ["lora_r", "lora_alpha", "learning_rate", "num_epochs"]) {
    await expect(spec.getByText(key, { exact: true })).toBeVisible();
  }

  // ...and the cost-and-time estimate is shown before anything is spent:
  // a duration range (never a point) and a per-phase cost breakdown, in the
  // account's currency, labelled an estimate. It is fetched for the selected
  // model after the page renders -- an estimate never blocks the surface it
  // appears on -- so this assertion waits for it to arrive.
  await expect(
    page.getByRole("heading", { name: "Cost and time estimate" }),
  ).toBeVisible();
  await expect(page.getByText(/never blocks a launch/i)).toBeVisible();
  for (const phase of [
    "provisioning",
    "readiness",
    "image_pull",
    "model_download",
    "training",
    "teardown",
  ]) {
    await expect(page.getByText(phase, { exact: true })).toBeVisible();
  }

  // ...and each decision the predictor made is shown with its reason visible
  // by default, its alternatives one interaction away (issue #76). The fake
  // provider only has an L4 free, so the device-count decision is the one
  // with two alternatives.
  await expect(
    page.getByRole("heading", { name: "Why this configuration" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "method" }),
  ).toBeVisible();
  // The chosen value is the method decision's strong; the method override
  // control also carries an identical option label (issue #79), so target
  // the reason, not the whole page.
  await expect(
    page.locator("strong").filter({ hasText: /^qlora$/ }),
  ).toBeVisible();
  await expect(
    page.getByText(/cheapest card currently available/i),
  ).toBeVisible();
  // The disclosure summaries are the one interaction away: several
  // decisions carry two alternatives each, so the text is not unique.
  await expect(
    page.getByText("Alternatives considered (2)").first(),
  ).toBeVisible();

  // ...and the screen offers exactly one obvious action.
  await expect(page.getByRole("button", { name: "Launch job" })).toHaveCount(1);

  await qwen8b.check();
  await page.getByRole("button", { name: "Launch job" }).click();

  // The job appears immediately afterwards, already carrying the choice.
  await expect(page).toHaveURL(/\/jobs\/job_/);
  await expect(page.getByText("qwen3-8b")).toBeVisible();

  // The fake provider runs the job to completion in moments; the running-job
  // view follows it there on its own.
  await expect(pairedValue(page, "State")).toHaveText("complete", {
    timeout: 20_000,
  });
  await expect(
    page.getByRole("link", { name: /Download the artifact/ }),
  ).toBeVisible();

  // The same explanation survives the job: the quote frozen at launch still
  // carries each decision and its reason after the run has finished (issue
  // #76) -- a completed job explains itself like a planned one.
  await expect(
    page.getByRole("heading", { name: "Why this configuration" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "method" }),
  ).toBeVisible();
  await expect(page.getByText("qlora", { exact: true })).toBeVisible();
});

// An override on the plan (issue #79): a decision the predictor made can be
// changed from the same surface that explains it, and changing one re-requests
// the plan -- the recomputation rules live on the server. An override that
// the trainer cannot run yet is still describable on the plan, and the launch
// is where that description stops being a quote.
test("a plan decision can be overridden and the launch refuses what the trainer cannot run", async ({
  page,
}) => {
  await uploadValidatedRows(
    page,
    Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`)),
  );
  await continueToLaunch(page);
  await expect(
    page.getByRole("heading", { name: "Why this configuration" }),
  ).toBeVisible();

  // The control sits beside the explanation it edits: one surface, not a
  // beginner mode and an expert mode.
  const method = page.getByRole("combobox", { name: "method override" });
  await expect(method).toBeVisible();

  // Override method to lora: the plan recomputes from the server, marks the
  // decision overridden, and shows lora as what a launch would freeze.
  await method.selectOption("lora");
  await expect(page.getByText("you changed this")).toBeVisible();
  await expect(page.getByText(/you chose lora/i)).toBeVisible();

  // The trainer executes only QLoRA today, so the launch refuses -- with the
  // stable code, never by silently running something else.
  await page.getByRole("button", { name: "Launch job" }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "not_executable" }),
  ).toBeVisible();
  await expect(page).not.toHaveURL(/\/jobs\/job_/);
});

// An override that the trainer can run is frozen into the job spec (issue
// #79): the sequence-length decision is folded into the hyperparameters the
// job trains with, and the frozen quote still marks the decision as
// overridden after the run has finished -- a completed job explains what it
// actually used.
test("a runnable override is frozen into the job and still marked after the run", async ({
  page,
  request,
}) => {
  await uploadValidatedRows(
    page,
    Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`)),
  );
  await continueToLaunch(page);
  await expect(
    page.getByRole("heading", { name: "Why this configuration" }),
  ).toBeVisible();

  const sequence = page.getByRole("spinbutton", {
    name: "sequence length override",
  });
  await sequence.fill("4096");
  await sequence.blur();
  await expect(page.getByText("you changed this")).toBeVisible();

  await page.getByRole("button", { name: "Launch job" }).click();
  await expect(page).toHaveURL(/\/jobs\/job_/);

  // Since #39 the job page opens on the live running view and hands back to
  // the finished record only at a terminal state, so asserting the frozen
  // marks straight after the launch is a race with however fast this runner
  // gets the job to an ending. It passed on one CI runner and failed on a
  // slower one from the same commit. Wait for the ending itself rather than
  // widening a timeout, which would only make the race rarer.
  const jobId = page.url().split("/jobs/")[1];
  await expect
    .poll(
      async () => {
        const rec = await request.get(`${backendHealth}/v1/jobs/${jobId}`);
        return (await rec.json()).status;
      },
      { timeout: 60_000 },
    )
    .toMatch(/^(complete|failed|cancelled)$/);

  // The frozen quote carries the override's mark and its value onto the
  // finished record -- the explanation survives the job like the spec does.
  await expect(page.getByText("you changed this")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "sequence length" }),
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
  await uploadValidatedRows(page, rows, 90_000);
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
  // has focus. The cap is a failure guard, not an assertion about the page --
  // the decisions' disclosure widgets and the advanced-settings disclosure's
  // summary (issue #80) are legitimate tab stops, so the count is deliberately
  // generous. The decisions load after the page draws (an estimate never
  // blocks it), so wait for them before counting stops.
  await continueToLaunch(page);
  const launch = page.getByRole("button", { name: "Launch job" });
  await expect(
    page.getByRole("heading", { name: "Why this configuration" }),
  ).toBeVisible();
  const maxTabStops = 24;
  for (
    let i = 0;
    i < maxTabStops && !(await launch.evaluate((el) => el === document.activeElement));
    i++
  ) {
    await page.keyboard.press("Tab");
  }
  await expect(launch).toBeFocused();

  // Shift+Tab back into the model group -- past the decisions' disclosures
  // and their override controls -- and choose with the arrow keys. Only the
  // checked radio is in the tab order (roving tabindex), so the first radio
  // reached going backwards is the checked model, and ArrowDown moves the
  // choice on from there. The loop targets a radio specifically: the
  // decisions' free-form controls are number inputs (issue #79), which are
  // also INPUTs.
  for (
    let i = 0;
    i < maxTabStops &&
    !(await page.evaluate(
      () => document.activeElement?.getAttribute("type") === "radio",
    ));
    i++
  ) {
    await page.keyboard.press("Shift+Tab");
  }
  await page.keyboard.press("ArrowDown");
  await expect(
    page.getByRole("radio", { name: /Qwen\/Qwen3-8B/ }),
  ).toBeChecked();

  // Launch from the keyboard alone: back past the disclosures to the action.
  for (
    let i = 0;
    i < maxTabStops && !(await launch.evaluate((el) => el === document.activeElement));
    i++
  ) {
    await page.keyboard.press("Tab");
  }
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

// The advanced surface (issue #80): every dial the pinned trainer exposes,
// behind an explicit disclosure, each naming the specific thing that goes
// wrong. An override is re-requested from the server and frozen into the job
// spec, which the finished run shows.
test("the advanced surface is behind a disclosure, names its failure modes, and an override is frozen onto the finished run", async ({
  page,
}) => {
  await uploadValidatedRows(
    page,
    Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`)),
  );
  await continueToLaunch(page);
  await expect(
    page.getByRole("heading", { name: "Cost and time estimate" }),
  ).toBeVisible();

  // The settings are behind an explicit disclosure, not a wall of dials: the
  // disclosure is named, and no override control is reachable until it is
  // opened.
  const advanced = page.getByLabel("Advanced settings");
  await expect(advanced).toBeVisible();
  await expect(
    page.getByRole("spinbutton", { name: "learning_rate override" }),
  ).not.toBeVisible();

  // Opening it is a deliberate act; each setting then carries the specific
  // thing that goes wrong, inline -- not a general caution.
  await advanced.locator("summary").click();
  await expect(
    page.getByRole("spinbutton", { name: "learning_rate override" }),
  ).toBeVisible();
  await expect(
    page.getByText(/diverges to NaN partway through a paid run/i),
  ).toBeVisible();
  // The distinction is explained, with both refused-input examples named.
  await expect(
    page.getByText(/Why some settings are adjustable and others are refused/i),
  ).toBeVisible();
  await expect(
    page.getByText(/mixes reasoning traces with plain answers/i),
  ).toBeVisible();

  // A trainer setting this product does not offer is visible with its reason,
  // searchable rather than absent.
  await page
    .getByRole("searchbox", {
      name: "Search the trainer's settings Temper does not offer",
    })
    .fill("wandb_project");
  await expect(
    page.getByText("wandb_project", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByText(/experiment-tracking integration/i),
  ).toBeVisible();

  // An override is re-requested from the server and marked, and the frozen
  // specification the page previews reflects it.
  const lr = page.getByRole("spinbutton", { name: "learning_rate override" });
  await lr.fill("0.0001");
  await lr.blur();
  await expect(page.getByText("you changed this").first()).toBeVisible();
  await expect(
    page.getByText("0.0001", { exact: true }).first(),
  ).toBeVisible();

  await page.getByRole("button", { name: "Launch job" }).click();
  await expect(page).toHaveURL(/\/jobs\/job_/);

  // The finished run shows the settings the user froze into the job spec: the
  // run says what it actually used, after it is over.
  await expect(pairedValue(page, "State")).toHaveText("complete", {
    timeout: 20_000,
  });
  await expect(
    page.getByRole("heading", { name: "Settings you changed" }),
  ).toBeVisible();
  await expect(page.getByText("learning_rate", { exact: true })).toBeVisible();
  await expect(page.getByText("0.0001", { exact: true })).toBeVisible();
});

// An override that makes the job infeasible is refused before anything is
// spent, with the same arithmetic the predictor used, and the control reverts
// rather than leaving a lie in the form.
test("an advanced override that makes the job infeasible is refused before launch", async ({
  page,
}) => {
  await uploadValidatedRows(
    page,
    Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`)),
  );
  await continueToLaunch(page);
  await expect(
    page.getByRole("heading", { name: "Cost and time estimate" }),
  ).toBeVisible();

  const advanced = page.getByLabel("Advanced settings");
  await advanced.locator("summary").click();
  const mbs = page.getByRole("spinbutton", {
    name: "micro_batch_size override",
  });
  await expect(mbs).toBeVisible();

  // A micro batch this large cannot fit anything the provider has free; the
  // recompute refuses with the stable code and the arithmetic, and the control
  // reverts to the default.
  await mbs.fill("512");
  await mbs.blur();
  const alert = page.getByRole("main").getByRole("alert");
  await expect(alert).toContainText("configuration_does_not_fit");
  await expect(mbs).toHaveValue("1");

  // Still here, still able to act: the refusal did not navigate away or
  // disable the launch.
  await expect(
    page.getByRole("button", { name: "Launch job" }),
  ).toBeEnabled();
  await expect(page).not.toHaveURL(/\/jobs\/job_/);
});
