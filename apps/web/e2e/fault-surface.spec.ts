import { expect, test } from "@playwright/test";
import { E2E_BACKEND_PORT } from "../src/lib/backend";
import { pairedValue, requireFakeProvider } from "./helpers";

// The fault surface (#24): a reviewer turns on a fault and watches what
// happens, named as deliberately broken. Since #35 the trainer-side `oom`
// fault demonstrates the memory recovery: the first machine exhausts memory,
// the job retries automatically with the effective batch preserved, and it
// completes -- while the deliberately broken first attempt still carries its
// `simulated_oom` code and the run's own history still names the fault, so a
// deliberately broken run can never be mistaken for a real one, and a person
// should be able to see that in the UI, not only in tests.
//
// The control plane is booted with TEMPER_FAKE_PROVIDER (playwright.config.ts),
// which is also what lets the fault spec past the creation guard: the
// zero-cost tier is the fake provider, and that is the tier being watched.
const backend = process.env.TEMPER_BACKEND_URL ?? `http://127.0.0.1:${E2E_BACKEND_PORT}`;

test.beforeAll(async ({ playwright }) => {
  await requireFakeProvider(() => playwright.request.newContext(), backend);
});

function chat(user: string, assistant: string) {
  return {
    messages: [
      { role: "user", content: user },
      { role: "assistant", content: assistant },
    ],
  };
}

function twelveRows() {
  return Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`));
}

test("an oom fault is watched through the memory recovery it proves", async ({
  page,
  request,
}) => {
  const payload = twelveRows().map((r) => JSON.stringify(r)).join("\n") + "\n";
  const uploaded = await request.post(`${backend}/v1/datasets`, {
    multipart: {
      file: {
        name: "d.jsonl",
        mimeType: "application/octet-stream",
        buffer: Buffer.from(payload, "utf8"),
      },
    },
  });
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
      hyperparameters: { simulated_failure_code: { name: "oom" } },
    },
  });
  expect(launched.status()).toBe(201);
  const jobId = (await launched.json()).id;

  await expect
    .poll(
      async () => {
        const rec = await request.get(`${backend}/v1/jobs/${jobId}`);
        return (await rec.json()).status;
      },
      { timeout: 60_000 },
    )
    .toMatch(/^(complete|failed|cancelled)$/);

  await page.goto(`/jobs/${jobId}`);
  // The recovery, not a broken ending: the oom fault exhausts the first
  // machine and the job retries automatically with the effective batch
  // preserved, so it completes.
  await expect(pairedValue(page, "State")).toHaveText("complete");

  // The run's own history names the injected fault, in the words a reviewer
  // is looking for.
  await expect(page.getByRole("log")).toContainText(
    "Simulated fault injected: oom",
  );
  await expect(page.getByRole("log")).toContainText("deliberately broken");

  // The user is told a recovery happened and what changed -- and the
  // deliberately broken first attempt keeps the fault's `simulated_oom` code,
  // so the marker that makes the break identifiable survives the recovery.
  const alert = page.getByTestId("memory-recovery");
  await expect(alert).toContainText("Memory recovery");
  await expect(alert).toContainText(/effective batch/);
  await expect(alert).toContainText("simulated_oom");

  // The recovered job finished training, so its artifact is downloadable.
  await expect(
    page.getByRole("link", { name: /Download the artifact/ }),
  ).toHaveCount(1);
});
