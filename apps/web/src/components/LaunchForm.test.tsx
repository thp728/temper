import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LaunchForm from "@/components/LaunchForm";
import type {
  CatalogEntry,
  JobSpecPreview,
  ModelCatalog,
  Quote,
} from "@/lib/api/generated/client";

const push = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
}));

// The generated client is the seam: these tests prove what the user does and
// sees, with the transport mocked out. The real client is exercised by the
// Playwright journeys.
const createJobMock = vi.hoisted(() => vi.fn());
const getQuoteMock = vi.hoisted(() => vi.fn());
const recomputeQuoteMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api/generated/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/generated/client")>()),
  createJobV1JobsPost: createJobMock,
  getQuoteV1QuotesGet: getQuoteMock,
  recomputeQuoteV1QuotesPost: recomputeQuoteMock,
}));

import { ApiError } from "@/lib/api/mutator";

function peakMemory(
  overrides: Partial<CatalogEntry["peak_memory"]> = {},
): CatalogEntry["peak_memory"] {
  return {
    weights_gb: 2.01,
    gradients_gb: 0.07,
    optimizer_gb: 0.4,
    activations_gb: 0.38,
    overhead_gb: 2.5,
    total_gb: 5.35,
    trainable_params: 33030144,
    tolerance: 0.15,
    gpu_type: "L4",
    gpu_capacity_gb: 24,
    headroom_gb: 18.65,
    ...overrides,
  };
}

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
    peak_memory: peakMemory(),
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
      peak_memory: peakMemory({
        total_gb: 9.6,
        trainable_params: 43646976,
        headroom_gb: 14.4,
      }),
    }),
  ],
  default: "qwen3-4b",
};

function quote(overrides: Partial<Quote> = {}): Quote {
  return {
    currency: "INR",
    minor_unit: 100,
    dataset_id: "ds_abc123",
    dataset_created_at: 1756160400,
    base_revision: "1cfa9a7208912126459214e8b04321603b3df60c",
    token_count: 12345,
    expires_at: 1756160400 + 86400,
    phases: [
      {
        name: "provisioning",
        duration_low_s: 13,
        duration_high_s: 17,
        cost_low_minor: 15,
        cost_high_minor: 20,
      },
      {
        name: "readiness",
        duration_low_s: 40,
        duration_high_s: 68,
        cost_low_minor: 46,
        cost_high_minor: 78,
      },
      {
        name: "image_pull",
        duration_low_s: 87,
        duration_high_s: 183,
        cost_low_minor: 100,
        cost_high_minor: 210,
      },
      {
        name: "model_download",
        duration_low_s: 13,
        duration_high_s: 51,
        cost_low_minor: 15,
        cost_high_minor: 59,
      },
      {
        name: "training",
        duration_low_s: 80,
        duration_high_s: 800,
        cost_low_minor: 92,
        cost_high_minor: 918,
      },
      {
        name: "teardown",
        duration_low_s: 5,
        duration_high_s: 20,
        cost_low_minor: 6,
        cost_high_minor: 23,
      },
    ],
    duration_low_s: 238,
    duration_high_s: 1139,
    cost_low_minor: 274,
    cost_high_minor: 1308,
    storage_cost_usd_per_hour: 0.0137,
    storage_cost_usd_total_low_minor: 1,
    storage_cost_usd_total_high_minor: 4,
    is_estimate: true,
    decisions: [
      {
        decision: "hardware",
        chosen: "L4",
        constraint:
          "predicted peak is 5.4 GB; the L4 (24.0 GB) is the cheapest card currently available that holds it with 18.6 GB to spare.",
        alternatives: [],
        overridden: false,
      },
      {
        decision: "method",
        chosen: "qlora",
        constraint:
          "the trainer can execute qlora today, and selection picks the cheapest executable method that fits.",
        alternatives: [
          {
            value: "lora",
            cost: "needs 11.4 GB peak",
            constraint: "the trainer cannot execute LoRA yet.",
          },
        ],
        overridden: false,
      },
    ],
    override_options: {
      method: ["qlora", "lora", "full"],
      hardware: ["A100-80GB", "H100", "H200", "L4", "RTX-PRO6000"],
      precision: ["nf4 (4-bit)", "bf16 (no quantisation)"],
    },
    ...overrides,
  };
}

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
  getQuoteMock.mockReset();
  recomputeQuoteMock.mockReset();
  getQuoteMock.mockResolvedValue(null);
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

  it("shows the predicted peak memory and headroom for every model, before launch", () => {
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    const card4b = optionCard("Qwen/Qwen3-4B");
    expect(card4b).toHaveTextContent("5.35");
    expect(card4b).toHaveTextContent("18.65");
    expect(card4b).toHaveTextContent("L4");
    const card8b = optionCard("Qwen/Qwen3-8B");
    expect(card8b).toHaveTextContent("9.6");
    expect(card8b).toHaveTextContent("14.4");
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
      overrides: [],
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

  it("fetches and shows the selected model's quote: a duration range, not a point", async () => {
    getQuoteMock.mockResolvedValue(quote());
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    // The default model's quote is fetched after the page renders, with its
    // ranges.
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Cost and time estimate" }),
      ).toBeVisible(),
    );
    expect(getQuoteMock).toHaveBeenCalledWith({
      dataset_id: "ds_abc123",
      base_model: "qwen3-4b",
    });
    expect(screen.getByText(/never blocks a launch/i)).toBeVisible();
    expect(screen.getByText(/3m 58s–18m 59s/)).toBeVisible();
    expect(screen.getByText(/INR 2\.74 – INR 13\.08/)).toBeVisible();
  });

  it("shows the per-phase breakdown of cost and duration", async () => {
    getQuoteMock.mockResolvedValue(quote());
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Cost and time estimate" }),
      ).toBeVisible(),
    );
    for (const phase of ["provisioning", "readiness", "image_pull", "model_download", "training", "teardown"]) {
      expect(screen.getByText(phase)).toBeVisible();
    }
    expect(screen.getByText(/1m 27s–3m 03s/)).toBeVisible(); // image pull
  });

  it("re-fetches the quote when the selected model changes", async () => {
    getQuoteMock.mockResolvedValue(quote());
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Cost and time estimate" }),
      ).toBeVisible(),
    );
    getQuoteMock.mockResolvedValue(
      quote({
        duration_low_s: 400,
        duration_high_s: 2000,
        cost_low_minor: 460,
        cost_high_minor: 2300,
      }),
    );
    await user.click(screen.getByRole("radio", { name: /Qwen\/Qwen3-8B/ }));
    await waitFor(() =>
      expect(getQuoteMock).toHaveBeenLastCalledWith({
        dataset_id: "ds_abc123",
        base_model: "qwen3-8b",
      }),
    );
    expect(screen.getByText(/6m 40s–33m 20s/)).toBeVisible();
    expect(screen.getByText(/INR 4\.60 – INR 23\.00/)).toBeVisible();
  });

  it("still offers the launch when a quote is absent", async () => {
    getQuoteMock.mockResolvedValue(null);
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    // The estimate is shown as unavailable, and the launch is still offered:
    // an estimate warns, it never blocks (spec 005).
    await waitFor(() =>
      expect(screen.getByText(/could not be computed/i)).toBeVisible(),
    );
    createJobMock.mockResolvedValueOnce({ id: "job_abc123" });
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalled();
  });

  // --- the plan is editable (issue #79) -------------------------------------

  it("re-requests the plan from the server when a decision is overridden", async () => {
    getQuoteMock.mockResolvedValue(quote());
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Cost and time estimate" }),
      ).toBeVisible(),
    );
    // The change is re-requested, not applied locally: the client sends the
    // pinned decision and the server recomputes the rest.
    recomputeQuoteMock.mockResolvedValue(
      quote({
        decisions: [
          {
            decision: "method",
            chosen: "lora",
            constraint: "you chose lora. The trainer executes only 'qlora' today.",
            alternatives: [],
            overridden: true,
          },
        ],
      }),
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "method override" }),
      "lora",
    );
    await waitFor(() =>
      expect(recomputeQuoteMock).toHaveBeenCalledWith({
        dataset_id: "ds_abc123",
        base_model: "qwen3-4b",
        overrides: [{ decision: "method", value: "lora" }],
      }),
    );
    // The recomputed plan is what is shown, and the override is marked.
    expect(screen.getByText("you changed this")).toBeVisible();
  });

  it("shows a refusal beside the plan when an override cannot be honoured", async () => {
    getQuoteMock.mockResolvedValue(quote());
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Cost and time estimate" }),
      ).toBeVisible(),
    );
    recomputeQuoteMock.mockRejectedValue(
      new ApiError(
        400,
        "configuration_does_not_fit",
        "full on a L4 predicts 66.9 GB peak, which the 24.0 GB L4 cannot hold.",
      ),
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "method override" }),
      "full",
    );
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("configuration_does_not_fit");
    expect(alert).toHaveTextContent("66.9 GB peak");
    // The plan reverts to the last valid configuration: the method control is
    // back on the predictor's choice, not the refused value.
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
      ).toHaveValue(""),
    );
  });

  it("launches with the pinned decisions frozen into the request", async () => {
    getQuoteMock.mockResolvedValue(quote());
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} />);
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Cost and time estimate" }),
      ).toBeVisible(),
    );
    recomputeQuoteMock.mockResolvedValue(quote());
    await user.selectOptions(
      screen.getByRole("combobox", { name: "method override" }),
      "lora",
    );
    await waitFor(() => expect(recomputeQuoteMock).toHaveBeenCalled());
    createJobMock.mockResolvedValueOnce({ id: "job_abc123" });
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalledWith({
      dataset_id: "ds_abc123",
      base_model: "qwen3-4b",
      hyperparameters: {},
      overrides: [{ decision: "method", value: "lora" }],
    });
  });
});
