import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LaunchForm from "@/components/LaunchForm";
import type {
  CatalogEntry,
  JobSpecPreview,
  ModelCatalog,
} from "@/lib/api/generated/client";

const push = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
}));

// The generated client is the seam: these tests prove what the user does and
// sees, with the transport mocked out. The real client is exercised by the
// Playwright journeys.
const createJobMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api/generated/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/generated/client")>()),
  createJobV1JobsPost: createJobMock,
}));

import { ApiError } from "@/lib/api/mutator";

function entry(overrides: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    id: "qwen3-4b",
    repo: "Qwen/Qwen3-4B",
    revision: "1cfa9a7208912126459214e8b04321603b3df60c",
    params_b: 4.0,
    license: "Apache-2.0",
    license_url: "https://huggingface.co/Qwen/Qwen3-4B",
    context_length: 40960,
    good_for: "Fast iteration and smaller datasets. The default.",
    min_gpu: "L4 (24 GB)",
    est_peak_vram_gb: 5.3,
    ...overrides,
  };
}

const catalog: ModelCatalog = {
  models: [
    entry(),
    entry({
      id: "qwen3-8b",
      repo: "Qwen/Qwen3-8B",
      revision: "b968826d9c46dd6066d109eabc6255188de91218",
      params_b: 8.2,
      context_length: 32768,
      good_for: "Higher quality when the dataset justifies it.",
      est_peak_vram_gb: 9.5,
    }),
  ],
  default: "qwen3-4b",
};

function preview(overrides: Partial<JobSpecPreview> = {}): JobSpecPreview {
  return {
    dataset: {
      id: "ds_abc123",
      filename: "d.jsonl",
      created_at: 1756160400,
      status: "valid",
    },
    hyperparameters: {
      lora_r: 16,
      lora_alpha: 32,
      learning_rate: 0.0002,
      num_epochs: 3,
    },
    warning: null,
    ...overrides,
  };
}

function specValue(name: string): string | null | undefined {
  return screen.getByText(name).closest("dt")?.nextElementSibling?.textContent;
}

// A model option is one labelled control; its card carries the facts about
// that choice. Read as the user reads: pick the option by its name, then read
// the text beside it.
function optionCard(repo: string): HTMLElement {
  const radio = screen.getByRole("radio", {
    name: new RegExp(repo.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
  });
  const card = radio.closest("label");
  if (!card) throw new Error(`no label card for ${repo}`);
  return card;
}

beforeEach(() => {
  createJobMock.mockReset();
  push.mockReset();
});

describe("LaunchForm", () => {
  it("offers every catalog model as a named choice, with licence and pinned revision", () => {
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    for (const m of catalog.models) {
      const card = optionCard(m.repo);
      expect(screen.getByRole("radio", { name: new RegExp(m.repo) })).toBeVisible();
      // The terms and the pin, beside the choice they belong to.
      expect(card).toHaveTextContent(`Licence ${m.license}`);
      expect(card).toHaveTextContent(m.revision);
    }
  });

  it("preselects the catalog default", () => {
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    expect(
      screen.getByRole("radio", { name: /Qwen\/Qwen3-4B/ }),
    ).toBeChecked();
    expect(
      screen.getByRole("radio", { name: /Qwen\/Qwen3-8B/ }),
    ).not.toBeChecked();
  });

  it("shows the specification the job will freeze, before launching", () => {
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    expect(specValue("lora_r")).toBe("16");
    expect(specValue("num_epochs")).toBe("3");
    // The commitment is stated where the numbers are read...
    expect(screen.getByText(/frozen at launch/i)).toBeVisible();
    // ...alongside the fact that it cannot be undone later.
    expect(screen.getByText(/cannot be changed afterwards/i)).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Launch job" }),
    ).toBeVisible();
  });

  it("launches with the chosen model and opens the job's own page", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    createJobMock.mockResolvedValueOnce({ id: "job_abc123" });
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalledWith({
      dataset_id: "ds_abc123",
      base_model: "qwen3-4b",
      hyperparameters: {},
    });
    await vi.waitFor(() =>
      expect(push).toHaveBeenCalledWith("/jobs/job_abc123"),
    );
  });

  it("launches with a model chosen after arrival", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    await user.click(screen.getByRole("radio", { name: /Qwen\/Qwen3-8B/ }));
    createJobMock.mockResolvedValueOnce({ id: "job_def456" });
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalledWith(
      expect.objectContaining({ base_model: "qwen3-8b" }),
    );
  });

  it("keeps a refused launch's stable code on the page", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    createJobMock.mockRejectedValueOnce(
      new ApiError(
        400,
        "unknown_model",
        "'gpt-9' is not in the catalog.",
      ),
    );
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("unknown_model");
    expect(alert).toHaveTextContent("not in the catalog");
    expect(push).not.toHaveBeenCalled();
    // Still here, still able to act: pick the other model and retry.
    expect(
      screen.getByRole("radio", { name: /Qwen\/Qwen3-8B/ }),
    ).toBeEnabled();
  });

  it("survives an unreachable server without losing the refusal", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    createJobMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(screen.getByRole("alert")).toHaveTextContent("network_error");
  });

  it("disables the action while the launch is in flight", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    let resolve!: (v: unknown) => void;
    createJobMock.mockReturnValueOnce(
      new Promise((res) => {
        resolve = res;
      }),
    );
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(
      screen.getByRole("button", { name: /launching/i }),
    ).toBeDisabled();
    resolve({});
  });
});
