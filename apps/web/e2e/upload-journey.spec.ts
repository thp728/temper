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
  await page.goto("/");

  // Every control carries an accessible name.
  await expect(page.getByLabel("Dataset file (.jsonl)")).toBeVisible();
  const submit = page.getByRole("button", { name: "Upload and validate" });
  await expect(submit).toBeVisible();

  // The controls are reachable by keyboard alone: tab from the page's start
  // until the action has focus. The cap is a failure guard, not an assertion
  // about the page -- the header carries two links before the form.
  const maxTabStops = 6;
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
