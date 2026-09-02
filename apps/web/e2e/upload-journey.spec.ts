import { expect, type Page } from "@playwright/test";
import { test } from "@playwright/test";
import { chat, tempFile, jsonl, upload, uploadRows } from "./helpers";

// A good test here drives the application the way a person does: find a
// control by its accessible name, act on it, assert on what the user can
// then see. No internal structure, no class names.

// Stat cards pair a visible label with a value; read them as pairs.
function statValue(page: Page, name: string) {
  return page
    .getByText(name, { exact: true })
    .locator("xpath=following-sibling::dd");
}

test("an accepted dataset reaches its report and offers to proceed", async ({
  page,
}) => {
  const rows = Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`));
  await uploadRows(page, rows);

  await expect(page.getByText("Validation passed")).toBeVisible();
  await expect(statValue(page, "Rows found")).toHaveText("12");
  await expect(statValue(page, "Usable rows")).toHaveText("12");
  // The schema is named...
  await expect(statValue(page, "Schema")).toHaveText("chat");
  // ...and thinking mode is stated in plain language, not as a boolean.
  await expect(page.getByText("Not detected")).toBeVisible();
  await expect(
    page.getByText(/trained to answer directly/),
  ).toBeVisible();

  // The token count is produced by the phase that runs after validation and
  // lands on this same report (issue #42). It arrives asynchronously, so the
  // expectation waits for it.
  await expect(page.getByText("Token count", { exact: true })).toBeVisible();
  await expect(statValue(page, "Would be truncated")).toHaveText("0");

  // The preview shows how the first rows were understood.
  await expect(page.getByText(/user:/).first()).toBeVisible();
  await expect(page.getByText("q0").first()).toBeVisible();
  await expect(page.getByText("a0").first()).toBeVisible();

  // Twelve rows is below the recommended fifty: a warning appears...
  await expect(page.getByText("few_rows")).toBeVisible();
  // ...and does not block proceeding.
  await expect(
    page.getByRole("link", { name: "Choose a model and continue" }),
  ).toBeVisible();
});

test("continuing hands over to the ported launch screen in this shell", async ({
  page,
}) => {
  // Since #38 the next screen is part of this application, not a proxy
  // handoff. The ported screen is recognised by what only it says -- the
  // frozen-spec statement that replaces the form post.
  const rows = Array.from({ length: 12 }, (_, i) => chat(`q${i}`, `a${i}`));
  await uploadRows(page, rows);

  await page
    .getByRole("link", { name: "Choose a model and continue" })
    .click();

  await expect(
    page.getByRole("heading", { name: "Choose a base model" }),
  ).toBeVisible();
  await expect(page.getByText(/cannot be changed afterwards/i)).toBeVisible();
});

test("a rejected dataset names each problem against its line", async ({
  page,
}) => {
  const lines = [
    ...Array.from({ length: 11 }, (_, i) => JSON.stringify(chat(`q${i}`, `a${i}`))),
    "{not json}",
    JSON.stringify(chat("q", "   ")),
  ];
  await upload(page, await tempFile(lines.join("\n") + "\n", "bad.jsonl"));

  await expect(page.getByText("This dataset was rejected")).toBeVisible();
  // The offending lines are named, with their stable codes.
  await expect(page.getByText("Line 12")).toBeVisible();
  await expect(page.getByText("invalid_json")).toBeVisible();
  await expect(page.getByText("Line 13")).toBeVisible();
  await expect(page.getByText("empty_target")).toBeVisible();

  // A rejected dataset cannot proceed.
  await expect(
    page.getByRole("link", { name: "Choose a model and continue" }),
  ).toHaveCount(0);
});

test("a file the picker allows but the API refuses keeps its stable code", async ({
  page,
}) => {
  const txt = await tempFile("not a dataset at all", "notes.txt");

  // setInputFiles bypasses the picker's accept filter, as a drag-and-drop
  // would; the refusal must come back typed rather than as a generic error.
  await upload(page, txt);

  // Scoped: Next's own route announcer is also an alert.
  const alert = page.getByRole("main").getByRole("alert");
  await expect(alert).toContainText("unsupported_extension");
});

test("the journey is keyboard-reachable and survives a small screen", async ({
  page,
}) => {
  await page.setViewportSize({ width: 375, height: 720 });
  await page.goto("/datasets");

  // Every control carries an accessible name.
  await expect(page.getByLabel("Dataset file (.jsonl)")).toBeVisible();
  const submit = page.getByRole("button", { name: "Upload and validate" });
  await expect(submit).toBeVisible();

  // The controls are reachable by keyboard alone: tab from the page's start
  // until the action has focus. The cap is a failure guard, not an assertion
  // about the page -- the sidebar carries the brand link, four nav items and
  // the new-job action before the form.
  const maxTabStops = 8;
  for (let i = 0; i < maxTabStops && !(await submit.evaluate((el) => el === document.activeElement)); i++) {
    await page.keyboard.press("Tab");
  }
  await expect(submit).toBeFocused();

  // Nothing forces horizontal scrolling on a phone-sized viewport.
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});

// --- import from a public repository (issue #45) -----------------------------
// The journeys boot the control plane with TEMPER_FAKE_PROVIDER, which swaps
// the remote-dataset seam for a canned fake -- importing reaches no network,
// exactly like a launch reaches no GPU. The fake knows three references: one
// that imports to a valid report, one that fails validation, and one whose
// split resolves to nothing.

async function importRepo(page: Page, repo: string) {
  await page.goto("/datasets");
  await page.getByLabel("Public repository").fill(repo);
  await page.getByRole("button", { name: "Import and validate" }).click();
}

test("an imported repository reaches its validation report", async ({ page }) => {
  await importRepo(page, "acme/demo-chat");

  // The imported rows went through the identical validation path: the same
  // report an upload of the same rows would produce.
  await expect(page.getByText("Validation passed")).toBeVisible();
  await expect(statValue(page, "Rows found")).toHaveText("12");
  await expect(statValue(page, "Usable rows")).toHaveText("12");
  await expect(
    page.getByRole("link", { name: "Choose a model and continue" }),
  ).toBeVisible();
});

test("an import that fails validation is kept with its report", async ({
  page,
}) => {
  await importRepo(page, "acme/demo-broken");

  await expect(page.getByText("This dataset was rejected")).toBeVisible();
  // The same line-numbered errors an upload of these rows would carry.
  await expect(page.getByText("Line 1")).toBeVisible();
  await expect(page.getByText("missing_messages")).toBeVisible();
  await expect(page.getByText("Line 2")).toBeVisible();
  await expect(page.getByText("no_assistant_turn")).toBeVisible();
});

test("a split that resolves to nothing is refused with its reason", async ({
  page,
}) => {
  await importRepo(page, "acme/empty-split");

  // The refusal keeps its stable code on the form, like every other refusal.
  const alert = page.getByRole("main").getByRole("alert");
  await expect(alert).toContainText("split_empty");
  await expect(alert).toContainText("no rows");
});
