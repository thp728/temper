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

test("a user comes back tomorrow, finds the job, and collects the artifact", async ({
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
    page.getByRole("heading", { name: "Your artifact" }),
  ).toBeVisible();

  // The finished record keeps how the run got there (issue #49): the pulls'
  // progress with a measured rate, and the raw lines kept as collapsed detail
  // rather than scattered into the log.
  await expect(
    page.getByRole("progressbar", { name: "image pull" }),
  ).toBeVisible();
  await expect(
    page.getByRole("progressbar", { name: "model download" }),
  ).toHaveAttribute("aria-valuenow", "100");
  await expect(page.getByText(/MB\/s/).first()).toBeVisible();

  // What it produced can be downloaded, and is an archive when it lands.
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: /Download the artifact/ }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe(`${jobId}-artifact.zip`);
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "temper-e2e-dl-"));
  const target = path.join(dir, "artifact.zip");
  await download.saveAs(target);
  const bytes = await fs.readFile(target);
  // A zip begins with "PK"; whatever else the artifact holds, it unzips.
  expect(bytes.subarray(0, 2).toString()).toBe("PK");
});

test("a finished job shows which checkpoint was chosen and why, and keeps the rest downloadable", async ({
  page,
}) => {
  // Issue #62: the best checkpoint is chosen by held-out loss and the choice
  // is recorded on the run, so the finished record states it in plain
  // language -- and a checkpoint that was NOT chosen still downloads, so a
  // user is never locked out of their own run's history. The journey fake's
  // checkpoints deliberately make the last one not the best.
  const jobId = await launchFromTheShell(page);

  await expect(
    page.getByRole("heading", { name: "Checkpoints" }),
  ).toBeVisible();
  const checkpoints = page.getByRole("region", { name: "Checkpoints" });
  // The chosen checkpoint and its recorded why (scoped to the section: the
  // same sentence also appears in the output log's own event).
  await expect(
    checkpoints.getByText("Best checkpoint: step 20 (held-out loss 0.39)"),
  ).toBeVisible();
  await expect(
    checkpoints.getByText(
      "Step 20 has the lowest held-out loss (0.39) of 3 retained checkpoint(s).",
    ),
  ).toBeVisible();
  await expect(checkpoints.getByText("chosen result")).toBeVisible();

  // Every retained checkpoint is offered; the unchosen one downloads.
  for (const step of [10, 20, 30]) {
    await expect(
      checkpoints.getByRole("link", { name: "Download" }).nth(step / 10 - 1),
    ).toHaveAttribute(
      "href",
      `/v1/jobs/${jobId}/checkpoints/${step}`,
    );
  }
  const downloadPromise = page.waitForEvent("download");
  await checkpoints
    .getByRole("link", { name: "Download" })
    .first()
    .click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("checkpoint-10.tar");
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "temper-e2e-ckpt-"));
  const target = path.join(dir, "ckpt.tar");
  await download.saveAs(target);
  expect((await fs.readFile(target)).length).toBeGreaterThan(0);
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

test("a finished job offers each requested delivery format with its purpose", async ({
  page,
}) => {
  // Issue #74: a launch can ask for a merged single-file model and a
  // quantised local-inference format, produced on the machine at export time
  // and offered on the finished record by what each is for -- a user chooses
  // a format without knowing what a merge is.
  await uploadRows(page, twelveRows());
  await expect(page.getByText("Validation passed")).toBeVisible();
  await page.getByRole("link", { name: "Choose a model and continue" }).click();
  await expect(
    page.getByRole("heading", { name: "Choose a base model" }),
  ).toBeVisible();
  await page.getByRole("checkbox", { name: /Merged model/ }).check();
  await page
    .getByRole("checkbox", { name: /Quantised local format/ })
    .check();
  await page.getByRole("button", { name: "Launch job" }).click();

  await expect(page).toHaveURL(/\/jobs\/job_/);
  const jobId = new URL(page.url()).pathname.split("/").pop() ?? "";
  await expect(pairedValue(page, "State")).toHaveText("complete", {
    timeout: 30_000,
  });

  // The finished record offers the canonical artifact and each delivery
  // format, named by what it is for.
  await expect(
    page.getByRole("link", { name: "Download the artifact" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Download merged" }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Download quantised" }),
  ).toBeVisible();
  // Each format states what it is for, in plain language (issue #74).
  await expect(page.getByText(/self-contained model you can serve directly/)).toBeVisible();
  await expect(page.getByText(/run it on your own machine/)).toBeVisible();

  // Each format downloads through the artifact route with its own format.
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download merged" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe(`${jobId}-merged.zip`);
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "temper-e2e-merged-"));
  const target = path.join(dir, "merged.zip");
  await download.saveAs(target);
  expect((await fs.readFile(target)).subarray(0, 2).toString()).toBe("PK");

  // The quantised format is offered too, and downloads.
  const downloadPromise2 = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download quantised" }).click();
  const download2 = await downloadPromise2;
  expect(download2.suggestedFilename()).toBe(`${jobId}-quantised.zip`);
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

  // Wait for the ending before reading the record. Since #39 the job page
  // opens on the live running view and hands back to the finished record only
  // at a terminal state, so navigating straight after creating the job races
  // whatever speed this runner gets it to one. It passed on one CI runner and
  // failed on another from the same commit.
  await expect
    .poll(
      async () => {
        const rec = await request.get(`${backend}/v1/jobs/${jobId}`);
        return (await rec.json()).status;
      },
      { timeout: 60_000 },
    )
    .toMatch(/^(complete|failed|cancelled)$/);

  // A user landing directly on a failed job's record:
  await page.goto(`/jobs/${jobId}`);
  await expect(pairedValue(page, "State")).toHaveText("failed");

  // Why, in plain language, beside the stable code that names it precisely.
  const alert = page.getByRole("main").getByRole("alert");
  await expect(alert).toContainText("gpu_stalled");
  await expect(alert).toContainText(/stopped by a safety limit/);
  await expect(alert).toContainText(/no artifact was produced/i);

  // What the job was doing before it died stays readable in the history.
  await expect(page.getByRole("log")).toContainText("building trainer image");

  // And nothing offers a download that does not exist.
  await expect(page.getByRole("link", { name: /Download the artifact/ })).toHaveCount(
    0,
  );
});

test("a finished job shows the side-by-side comparison of both models", async ({
  page,
}) => {
  // Issue #69: the same held-out prompt is answered by the base model and by
  // the checkpoint the run chose, side by side, with the decoding settings
  // recorded so a reader can tell whether two outputs are comparable. The
  // simulated machine's result carries the comparison, so the journey
  // asserts the surface renders it (the generation itself is the trainer's
  // hardware path, exercised by the hardware-marked trainer test).
  await launchFromTheShell(page);

  const section = page.getByRole("region", {
    name: "Base model vs your tuned model",
  });
  await expect(section).toBeVisible();
  // Both sides of the one held-out prompt the fake recorded.
  await expect(section.getByText("q0")).toBeVisible();
  await expect(
    section.getByText("The base model's answer to the held-out question."),
  ).toBeVisible();
  await expect(
    section.getByText("The tuned model's answer to the held-out question."),
  ).toBeVisible();
  // The tuned side names the checkpoint the run chose (the fake's best, step
  // 20), and the fixed decoding settings are shown beside the answers.
  await expect(
    section.getByText("Tuned model (chosen checkpoint, step 20)"),
  ).toBeVisible();
  await expect(section.getByText(/temperature 0.7/)).toBeVisible();
  await expect(section.getByText(/up to 128 new tokens/)).toBeVisible();
});

test("a finished job shows the general-capability smoke test, labelled as such", async ({
  page,
}) => {
  // Issue #73: the same fixed set of general questions answered by the base
  // model and the tuned model, reported as a change with the sample size
  // beside the number and the uncertainty stated -- a smoke test for
  // catastrophic forgetting, never a benchmark, and the interface says so. A
  // large regression (the fake's canned slice has one) is surfaced
  // prominently from the recorded flag. The simulated machine's result
  // carries the slice, so the journey asserts the surface renders it (the
  // slice itself is the trainer's hardware path).
  await launchFromTheShell(page);

  const section = page.getByRole("region", { name: "General capability" });
  await expect(section).toBeVisible();
  // The interface labels it a smoke test, not a benchmark.
  await expect(
    section.getByText(/smoke test for catastrophic forgetting, not a benchmark/i),
  ).toBeVisible();
  // The sample size sits beside each side's number, and the change and its
  // uncertainty are stated.
  await expect(section.getByText("6 of 8 (75%)")).toBeVisible();
  await expect(section.getByText("4 of 8 (50%)")).toBeVisible();
  await expect(section.getByText("−2 of 8 (−25%)")).toBeVisible();
  await expect(
    section.getByText(/standard error of the change is ±1.2 questions/i),
  ).toBeVisible();
  // The tuned side names the checkpoint the run chose (the fake's best, step
  // 20).
  await expect(
    section.getByText("Tuned model (chosen checkpoint, step 20)"),
  ).toBeVisible();
  // A large regression is surfaced prominently, with its threshold read from
  // the record.
  await expect(
    page.getByRole("alert").filter({ hasText: "Large regression" }),
  ).toBeVisible();
  await expect(
    page.getByRole("alert").filter({ hasText: "at least 2 fewer" }),
  ).toBeVisible();
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
