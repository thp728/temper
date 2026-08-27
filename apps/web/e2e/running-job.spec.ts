import { expect, type Page, test } from "@playwright/test";
import { E2E_BACKEND_PORT } from "../src/lib/backend";
import { chat, jsonl, pairedValue, requireFakeProvider } from "./helpers";

// The running-job journey (#39): watch a live job to completion, survive a
// dropped connection, and cancel one. The control plane is booted with
// TEMPER_FAKE_PROVIDER=1 and TEMPER_FAKE_LINE_DELAY_S (playwright.config.ts),
// so a launched job runs for a few seconds -- long enough to watch output
// arrive without a refresh and to act on a run that is still going -- while
// still costing no hardware and reaching no billing account.
//
// These journeys launch through the API rather than the shell's own form for
// one reason: the fake job is short-lived, and the point of the test is what
// happens on the running page, not the upload-and-launch walk (which
// launch-journey.spec covers). Launching through the API puts the page in
// front of a run that is still live.
const backend =
  process.env.TEMPER_BACKEND_URL ?? `http://127.0.0.1:${E2E_BACKEND_PORT}`;

test.beforeAll(async ({ playwright }) => {
  await requireFakeProvider(() => playwright.request.newContext(), backend);
});

function twelveRows() {
  return Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`));
}

async function launchViaApi(request: import("@playwright/test").APIRequestContext): Promise<string> {
  const payload = jsonl(twelveRows());
  const uploaded = await request.post(`${backend}/v1/datasets`, {
    multipart: {
      file: {
        name: "d.jsonl",
        mimeType: "application/octet-stream",
        buffer: Buffer.from(payload, "utf8"),
      },
    },
  });
  expect(uploaded.status()).toBe(201);
  const datasetId = (await uploaded.json()).id;
  const launched = await request.post(`${backend}/v1/jobs`, {
    data: { dataset_id: datasetId },
  });
  expect(launched.status()).toBe(201);
  return (await launched.json()).id;
}

test("a running job is watched live to completion, without a refresh", async ({
  page,
  request,
}) => {
  const jobId = await launchViaApi(request);
  await page.goto(`/jobs/${jobId}`);

  // While it runs, the live view is the one shown: the state is prominent,
  // the cancel control is present and clearly destructive, and the elapsed
  // clock is ticking.
  await expect(
    page.getByRole("heading", { name: "Cancel this job?" }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Cancel job" })).toBeVisible();
  await expect(page.getByText(/no adapter will be produced/i)).toBeVisible();

  // Output arrives without the page being refreshed: this line is recorded
  // only after the page was served, so its appearance proves the stream
  // delivered it. The finished record shows the same history, so the
  // assertion holds whichever half of the page is on screen when it lands.
  await expect(page.getByRole("log")).toContainText("running training", {
    timeout: 15_000,
  });

  // The latest measured loss appears as it becomes available.
  await expect(pairedValue(page, "Latest loss")).toContainText("0.6931", {
    timeout: 15_000,
  });

  // The job runs to completion on its own and the page hands back to the
  // finished record, which offers the adapter.
  await expect(pairedValue(page, "State")).toHaveText("complete", {
    timeout: 30_000,
  });
  await expect(
    page.getByRole("link", { name: /Download the adapter/ }),
  ).toBeVisible();
});

test("a dropped connection reconnects on its own", async ({ page, request }) => {
  const jobId = await launchViaApi(request);

  // Drop the first stream connection the page opens; the browser's own
  // reconnection must recover without any user action. Only the first is
  // aborted, so the retry that follows goes through.
  let drops = 0;
  await page.route("**/v1/jobs/*/stream*", async (route) => {
    if (drops === 0) {
      drops += 1;
      await route.abort();
    } else {
      await route.continue();
    }
  });

  await page.goto(`/jobs/${jobId}`);

  // The page still shows live output and reaches the terminal record.
  await expect(page.getByRole("log")).toContainText("running training", {
    timeout: 15_000,
  });
  await expect(pairedValue(page, "State")).toHaveText("complete", {
    timeout: 30_000,
  });
});

test("cancelling a running job is destructive and stops it", async ({
  page,
  request,
}) => {
  const jobId = await launchViaApi(request);
  await page.goto(`/jobs/${jobId}`);

  // The consequence is stated beside the control before the request exists.
  await expect(page.getByRole("button", { name: "Cancel job" })).toBeVisible();
  await expect(page.getByText(/cannot be undone/i)).toBeVisible();

  await page.getByRole("button", { name: "Cancel job" }).click();

  // The job ends cancelled -- the user's own decision, not a defect -- and
  // nothing offers a download that does not exist.
  await expect(pairedValue(page, "State")).toHaveText("cancelled", {
    timeout: 30_000,
  });
  await expect(
    page.getByRole("heading", { name: "Cancelled" }),
  ).toBeVisible();
  await expect(
    page.getByText(/that is what cancelling means here/i),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: /Download the adapter/ })).toHaveCount(
    0,
  );
});
