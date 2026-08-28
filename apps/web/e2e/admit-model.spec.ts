import { expect, type Page, test } from "@playwright/test";
import { E2E_BACKEND_PORT } from "../src/lib/backend";
import {
  chat,
  requireFakeProvider,
  uploadRows,
} from "./helpers";

// The boundary coming down (issue #58): a model outside the catalog is usable
// once a compatibility probe reports on it, and the probe's result is shown
// in front of the user before a job can be created. This journey probes a
// model -- against the app's own fake model-facts resolver, so nothing here
// touches the network -- and launches with the imported model.
//
// The backend is booted with TEMPER_FAKE_PROVIDER (playwright.config.ts), so
// the model-facts seam is the fake seeded with the catalog's real facts: the
// journey probes a catalog model as an *imported* one, which is exactly what
// the flow does for anything else, without depending on network reachability.
const backendHealth =
  process.env.TEMPER_BACKEND_URL ?? `http://127.0.0.1:${E2E_BACKEND_PORT}`;

test.beforeAll(async ({ playwright }) => {
  await requireFakeProvider(() => playwright.request.newContext(), backendHealth);
});

async function continueToLaunch(page: Page) {
  await page.getByRole("link", { name: "Choose a model and continue" }).click();
  await expect(
    page.getByRole("heading", { name: "Choose a base model" }),
  ).toBeVisible();
}

async function uploadValidatedRows(page: Page, rows: object[]) {
  // Validation is asynchronous since #31; the report (and the launch link)
  // only exist once it has passed.
  await uploadRows(page, rows);
  await expect(page.getByText("Validation passed")).toBeVisible();
}

test("a model outside the catalog is probed, shown and launched", async ({
  page,
}) => {
  await uploadValidatedRows(
    page,
    Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`)),
  );
  await continueToLaunch(page);

  // The disclosure holds the boundary: use any model repository at a pinned
  // revision, probed in front of the user.
  await page.getByText("Use a model outside the catalog").click();
  await page.getByLabel("Public repository").fill("Qwen/Qwen3-4B");
  // The pinned revision the catalog itself carries: a probe refuses anything
  // but a resolved commit, never a branch name.
  await page
    .getByLabel("Pinned revision")
    .fill("1cfa9a7208912126459214e8b04321603b3df60c");
  await page.getByRole("button", { name: "Probe and admit" }).click();

  // The result is shown, not merely enforced: the imported model appears
  // under Imported models -- the catalog's own Qwen/Qwen3-4B is a separate
  // radio, so scope to the imported group -- and it is selected.
  const imported = page
    .getByRole("group", { name: "Imported models" })
    .getByRole("radio", { name: /Qwen\/Qwen3-4B/ });
  await expect(imported).toBeVisible();
  await expect(imported).toBeChecked();
  await expect(
    page.getByText("Compatibility probe:", { exact: false }),
  ).toBeVisible();
  await expect(page.getByText("Usable", { exact: true }).first()).toBeVisible();

  // Launching uses the imported model; the finished job says what it trained
  // against (the imported model's id, not a catalog id).
  const importedId = await imported.getAttribute("value");
  await page.getByRole("button", { name: "Launch job" }).click();
  await expect(page).toHaveURL(/\/jobs\/job_/);
  await expect(page.getByText(importedId ?? "")).toBeVisible();
});

test("an unpinned revision is refused with its stable code", async ({
  page,
}) => {
  await uploadValidatedRows(
    page,
    Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`)),
  );
  await continueToLaunch(page);

  await page.getByText("Use a model outside the catalog").click();
  await page.getByLabel("Public repository").fill("Qwen/Qwen3-4B");
  await page.getByLabel("Pinned revision").fill("main");
  await page.getByRole("button", { name: "Probe and admit" }).click();

  // Next's route announcer is also an alert, so scope to main.
  const refusal = page.getByRole("main").getByRole("alert");
  await expect(refusal).toContainText("unpinned_revision");
  await expect(refusal).toContainText("not a pinned revision");
});
