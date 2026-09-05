import { expect, type APIRequestContext, type Page } from "@playwright/test";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

// Shared fixtures for the journeys: a dataset on disk, and the one way every
// journey enters the product. Kept beside the specs they serve; a helper no
// second spec wants does not belong in here.

function chat(user: string, assistant: string) {
  return {
    messages: [
      { role: "user", content: user },
      { role: "assistant", content: assistant },
    ],
  };
}

async function tempFile(content: string, name = "d.jsonl"): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "temper-e2e-"));
  const file = path.join(dir, name);
  await fs.writeFile(file, content, "utf8");
  return file;
}

function jsonl(rows: object[]): string {
  return rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
}

async function upload(page: Page, filePath: string) {
  await page.goto("/datasets");
  await page.getByLabel("Dataset file (.jsonl)").setInputFiles(filePath);
  await page.getByRole("button", { name: "Upload and validate" }).click();
}

async function uploadRows(page: Page, rows: object[]) {
  await upload(page, await tempFile(jsonl(rows)));
}

// The report page's own way into the launch screen: the "Associated jobs"
// panel's button, scoped to the dataset already on the page. Driven by its
// accessible name, the same as a person would click it.
async function goToLaunch(page: Page) {
  await page.getByRole("link", { name: "New job with this dataset" }).click();
  await expect(
    page.getByRole("heading", { name: "Model & dataset" }),
  ).toBeVisible();
}

// Stat cards and definition lists pair a visible label with a value; read
// them as pairs rather than matching bare words that repeat across the page.
// A record keeps several pairs in one list, so take the value that
// immediately follows the named term.
function pairedValue(page: Page, label: string) {
  return page
    .getByText(label, { exact: true })
    .locator("xpath=following-sibling::dd[1]");
}

// ADR-0024's guard, defined once: every spec that drives launches refuses to
// begin against a backend where TEMPER_FAKE_PROVIDER did not take effect --
// reuseExistingServer would otherwise make a plain dev server look identical
// to the one Playwright booted, and launching against it could provision
// real machines.
async function requireFakeProvider(
  newRequestContext: () => Promise<APIRequestContext>,
  backendUrl: string,
) {
  const context = await newRequestContext();
  const health = await context.get(`${backendUrl}/health`);
  const body = await health.json();
  await context.dispose();
  if (body.provider !== "fake") {
    throw new Error(
      `The control plane on port ${new URL(backendUrl).port} is not running ` +
        "with TEMPER_FAKE_PROVIDER=1 -- driving launches against it could " +
        "provision real machines. Stop that process and let Playwright " +
        "boot its own.",
    );
  }
}

export {
  chat,
  tempFile,
  jsonl,
  upload,
  uploadRows,
  goToLaunch,
  pairedValue,
  requireFakeProvider,
};
