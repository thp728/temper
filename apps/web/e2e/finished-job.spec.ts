import { expect, type Page, test } from "@playwright/test";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { E2E_BACKEND_PORT } from "../src/lib/backend";
import {
  chat,
  pairedValue,
  requireFakeProvider,
  uploadRows,
} from "./helpers";

// The finished-job journey (#40): a user who comes back tomorrow finds their
// job in the list, reads how it ended, and takes what it produced. Driven the
// way a person drives it -- controls by accessible name, assertions on what
// is then visible.
//
// The control plane is booted with TEMPER_FAKE_PROVIDER
// (playwright.config.ts); the fake runs every job to a terminal state in
// moments, so no journey here needs hardware or credentials. Like every
// launch-driving spec, this one refuses to run against a backend where that
// did not take effect (see helpers.requireFakeProvider).
const backend = process.env.TEMPER_BACKEND_URL ?? `http://127.0.0.1:${E2E_BACKEND_PORT}`;

test.beforeAll(async ({ playwright }) => {
  await requireFakeProvider(() => playwright.request.newContext(), backend);
});

// Twelve rows validate cleanly and launch with the defaults.
function twelveRows() {
  return Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`));
}

async function launchFromTheShell(page: Page): Promise<string> {
  await uploadRows(page, twelveRows());
  await expect(page.getByText("Validation passed")).toBeVisible();
  await page.getByRole("link", { name: "Choose a model and continue" }).click();
  await expect(
    page.getByRole("heading", { name: "Choose a base model" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Launch job" }).click();

  // The job appears immediately, and its record follows it to the end: the
  // page re-renders itself until the job is terminal.
  await expect(page).toHaveURL(/\/jobs\/job_/);
  const jobId = new URL(page.url()).pathname.split("/").pop() ?? "";
  await expect(pairedValue(page, "State")).toHaveText("complete", {
    timeout: 30_000,
  });
  return jobId;
}

test("a user comes back tomorrow, finds the job, and collects the adapter", async ({
  page,
}) => {
  const jobId = await launchFromTheShell(page);

  // Leave entirely -- close the tab, as it were -- and come back through the
  // shell's own navigation rather than browser history.
  await page.getByRole("link", { name: "Jobs", exact: true }).click();
  await expect(page).toHaveURL(/\/jobs$/);
  const row = page.getByRole("row", { name: new RegExp(jobId) });
  await expect(row).toBeVisible();
  // Enough detail to identify the job without opening it...
  await expect(row).toContainText("complete");
  await expect(row).toContainText("qwen3-4b");
  await expect(row).toContainText("d.jsonl");

  // ...and one link to the full record, where the ending is stated.
  await row.getByRole("link", { name: jobId }).click();
  await expect(page).toHaveURL(new RegExp(`/jobs/${jobId}$`));
  await expect(pairedValue(page, "State")).toHaveText("complete");
  await expect(
    page.getByRole("heading", { name: "Your adapter" }),
  ).toBeVisible();

  // What it produced can be downloaded, and is an archive when it lands.
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: /Download the adapter/ }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe(`${jobId}-adapter.zip`);
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "temper-e2e-dl-"));
  const target = path.join(dir, "adapter.zip");
  await download.saveAs(target);
  const bytes = await fs.readFile(target);
  // A zip begins with "PK"; whatever else the artifact holds, it unzips.
  expect(bytes.subarray(0, 2).toString()).toBe("PK");
});

test("a finished job compares its prediction against what happened, and the aggregate shows it", async ({
  page,
}) => {
  // Issue #77: the finished record shows what was predicted against what
  // happened -- duration, peak memory and cost, each marked measured or
  // derived -- and the aggregate view makes the same comparison across runs.
  const jobId = await launchFromTheShell(page);

  await expect(
    page.getByRole("heading", { name: "Prediction vs what happened" }),
  ).toBeVisible();
  // The measured figures state their basis (spec 005's measured-vs-derived).
  await expect(page.getByText("measured on the machine")).toBeVisible();
  await expect(
    page.getByText("derived from measured duration × frozen rate"),
  ).toBeVisible();
  // The run's own stages are shown, measured from its state transitions.
  await expect(page.getByText("packaging")).toBeVisible();

  // The aggregate is one link away, and this run is in it -- an outlier can
  // be named rather than pointed at.
  await page
    .getByRole("link", { name: "Predictions vs actuals across runs" })
    .click();
  await expect(page).toHaveURL(/\/calibration$/);
  await expect(
    page.getByRole("heading", { name: "Predictions vs what happened" }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: jobId })).toBeVisible();
});

test("a failed job says why in plain language, and keeps its stable code", async ({
  request,
  page,
}) => {
  // Set up through the API: what this journey tests is reading a failure,
  // not causing one. The reserved key asks the simulated machine to end with
  // a named code -- see fake_provider.py (issue #24 will grow honest fault
  // injection).
  const payload =
    twelveRows()
      .map((r) => JSON.stringify(r))
      .join("\n") + "\n";
  const uploaded = await request.post(`${backend}/v1/datasets`, {
    multipart: {
      file: {
        name: "d.jsonl",
        mimeType: "application/octet-stream",
        buffer: Buffer.from(payload, "utf8"),
      },
    },
  });
  // Validation runs in the background; the upload answers with the dataset's
  // id while it works. A job needs the finished report, so wait for it.
  expect(uploaded.status()).toBe(202);
  const datasetId = (await uploaded.json()).id;
  await expect
    .poll(
      async () => {
        const rec = await request.get(`${backend}/v1/datasets/${datasetId}`);
        return (await rec.json()).status;
      },
      { timeout: 30_000 },
    )
    .not.toBe("validating");

  const launched = await request.post(`${backend}/v1/jobs`, {
    data: {
      dataset_id: datasetId,
      hyperparameters: { simulated_failure_code: "gpu_stalled" },
    },
  });
  expect(launched.status()).toBe(201);
  const jobId = (await launched.json()).id;

  // A user landing directly on a failed job's record:
  await page.goto(`/jobs/${jobId}`);
  await expect(pairedValue(page, "State")).toHaveText("failed");

  // Why, in plain language, beside the stable code that names it precisely.
  const alert = page.getByRole("main").getByRole("alert");
  await expect(alert).toContainText("gpu_stalled");
  await expect(alert).toContainText(/stopped by a safety limit/);
  await expect(alert).toContainText(/no adapter was produced/i);

  // What the job was doing before it died stays readable in the history.
  await expect(page.getByRole("log")).toContainText("building trainer image");

  // And nothing offers a download that does not exist.
  await expect(page.getByRole("link", { name: /Download the adapter/ })).toHaveCount(
    0,
  );
});

test("the list survives a small screen", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 720 });
  await page.goto("/jobs");
  await expect(
    page.getByRole("heading", { name: "Jobs" }),
  ).toBeVisible();
  // Nothing forces horizontal scrolling on a phone-sized viewport; the table
  // scrolls inside its own container instead of taking the page with it.
  const overflow = await page.evaluate(
    () =>
      document.documentElement.scrollWidth -
      document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
