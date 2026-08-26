import { expect, type Page } from "@playwright/test";
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
  await page.goto("/");
  await page.getByLabel("Dataset file (.jsonl)").setInputFiles(filePath);
  await page.getByRole("button", { name: "Upload and validate" }).click();
}

async function uploadRows(page: Page, rows: object[]) {
  await upload(page, await tempFile(jsonl(rows)));
}

export { chat, tempFile, jsonl, upload, uploadRows };
