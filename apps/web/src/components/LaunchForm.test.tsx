import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LaunchForm from "@/components/LaunchForm";
import type {
  AdvancedSurface as AdvancedSurfaceModel,
  CatalogEntry,
  DatasetRecord,
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
const getPreviewMock = vi.hoisted(() => vi.fn());
const uploadDatasetMock = vi.hoisted(() => vi.fn());
const getDatasetMock = vi.hoisted(() => vi.fn());
const listJobsMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api/generated/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/generated/client")>()),
  createJobV1JobsPost: createJobMock,
  getQuoteV1QuotesGet: getQuoteMock,
  recomputeQuoteV1QuotesPost: recomputeQuoteMock,
  getJobSpecPreviewV1JobsSpecGet: getPreviewMock,
  uploadDatasetV1DatasetsPost: uploadDatasetMock,
  getDatasetV1DatasetsDatasetIdGet: getDatasetMock,
  listJobsV1JobsGet: listJobsMock,
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
      {
        decision: "sequence length",
        chosen: "2048",
        constraint: "the trainer default (2048, trainer-defaults.json).",
        alternatives: [],
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
    delivery_formats: [
      {
        id: "merged",
        what_for:
          "The base model with your trained change built into its full weights. Serve it directly.",
      },
      {
        id: "quantised",
        what_for:
          "A compact local-inference version of the merged model. Run it on your own machine.",
      },
    ],
    ...overrides,
  };
}

// The generated advanced surface (issue #80), as `GET /v1/surface` publishes
// it: a couple of exposed fields with their failure modes and a
// known-but-unsupported setting.
function surface(): AdvancedSurfaceModel {
  return {
    source_image: "axolotlai/axolotl:main@sha256:abc",
    axolotl_version: "0.19.0.dev0",
    config_model: "AxolotlInputConfig",
    known_keys: ["learning_rate", "num_epochs", "lora_r", "wandb_project"],
    platform_internal_keys: ["simulated_failure_code"],
    overrideable_keys: ["learning_rate", "num_epochs"],
    counts: {
      calculated: 0,
      exposed_with_named_failure_mode: 2,
      known_but_unsupported: 1,
    },
    tiers: {
      calculated: {},
      exposed_with_named_failure_mode: {
        learning_rate: {
          tier: "exposed_with_named_failure_mode",
          reason: "The peak learning rate for the cosine schedule.",
          failure_mode:
            "Too high and the loss diverges to NaN partway through a paid run.",
          type: "float",
        },
        num_epochs: {
          tier: "exposed_with_named_failure_mode",
          reason: "How many passes over the training set.",
          failure_mode: "Too few and the adapter underfits.",
          type: "float",
        },
      },
      known_but_unsupported: {
        wandb_project: {
          tier: "known_but_unsupported",
          reason: "experiment-tracking integration is not offered.",
        },
      },
    },
    runtime_only_validators: {
      model_validators: ["check_fsdp_deepspeed"],
      field_validator_count: 27,
    },
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
  getPreviewMock.mockReset();
  getPreviewMock.mockResolvedValue(preview());
  uploadDatasetMock.mockReset();
  getDatasetMock.mockReset();
  listJobsMock.mockReset();
  listJobsMock.mockResolvedValue({ jobs: [] });
  push.mockReset();
});

// The launch surface is a four-step wizard (Model & dataset >
// Hyperparameters > Compute & hardware > Review): model and dataset choice
// live on step 1, the spec and training-shape decisions on step 2, the
// estimate and where-it-runs decisions on step 3, delivery and Launch on
// step 4. Tests drive it the way a person does — Continue through the steps,
// then assert.
async function goToHyperparameters(user: ReturnType<typeof userEvent.setup>) {
  await user.click(
    screen.getByRole("button", { name: "Continue to hyperparameters" }),
  );
}

async function goToHardware(user: ReturnType<typeof userEvent.setup>) {
  await goToHyperparameters(user);
  await user.click(
    screen.getByRole("button", { name: "Continue to hardware" }),
  );
}

async function goToReview(user: ReturnType<typeof userEvent.setup>) {
  await goToHardware(user);
  await user.click(
    screen.getByRole("button", { name: "Continue to review" }),
  );
}

describe("LaunchForm", () => {
  it("offers every catalog model as a named choice, with licence and pinned revision", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    for (const m of catalog.models) {
      const card = optionCard(m.repo);
      expect(screen.getByRole("radio", { name: new RegExp(m.repo) })).toBeVisible();
      // The terms beside the choice they belong to; the forty-character
      // revision shows truncated, with the exact hash opening in a tooltip
      // on hover (and staying in the label for screen readers).
      expect(card).toHaveTextContent(`Licence ${m.license}`);
      const code = card.querySelector("code")!;
      expect(
        code.querySelector("[aria-hidden='true']")?.textContent,
      ).not.toBe(m.revision);
      await user.hover(code);
      await screen.findAllByText(m.revision);
      await user.unhover(code);
    }
  });

  it("shows params and context on every model card, without the memory estimate", () => {
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    // The memory arithmetic lives beside the hardware choice on step 3, not
    // on the card: the card says what the model is, not where it fits.
    const card4b = optionCard("Qwen/Qwen3-4B");
    expect(card4b).toHaveTextContent("4B");
    expect(card4b).toHaveTextContent("40960");
    expect(card4b).not.toHaveTextContent("headroom");
    expect(card4b).not.toHaveTextContent("Predicted peak VRAM");
    const card8b = optionCard("Qwen/Qwen3-8B");
    expect(card8b).toHaveTextContent("8.2");
    expect(card8b).toHaveTextContent("32768");
    expect(card8b).not.toHaveTextContent("headroom");
  });

  it("preselects the catalog default", () => {
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    expect(
      screen.getByRole("radio", { name: /Qwen\/Qwen3-4B/ }),
    ).toBeChecked();
    expect(
      screen.getByRole("radio", { name: /Qwen\/Qwen3-8B/ }),
    ).not.toBeChecked();
  });

  it("shows the specification the job will freeze, before launching", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    await goToHyperparameters(user);
    expect(specValue("lora_r")).toBe("16");
    expect(specValue("num_epochs")).toBe("3");
    // The commitment is stated where the numbers are read...
    expect(screen.getByText(/frozen at launch/i)).toBeVisible();
    // ...alongside the fact that it cannot be undone later.
    expect(screen.getByText(/cannot be changed afterwards/i)).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Continue to hardware" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Continue to review" }),
    );
    expect(
      screen.getByRole("button", { name: "Launch job" }),
    ).toBeVisible();
  });

  // An unpublished trainer image is refused by the orchestrator before it
  // provisions anything, so pressing Launch can only produce a failed job.
  // The review step says so while the user can still act on it.
  it("refuses to launch when the trainer image is not published", async () => {
    const user = userEvent.setup();
    render(
      <LaunchForm
        catalog={catalog}
        preview={preview({ trainer_image_published: false })}
        surface={null}
      />,
    );
    await goToReview(user);
    // The refusal keeps its stable code on the page.
    expect(screen.getByText(/image_not_published/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Launch job" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).not.toHaveBeenCalled();
  });

  // The other end of the same rule: a store the machine cannot upload to
  // means a run that is billed in full and delivers nothing.
  it("refuses to launch when the artifact could not be delivered", async () => {
    const user = userEvent.setup();
    render(
      <LaunchForm
        catalog={catalog}
        preview={preview({ artifact_deliverable: false })}
        surface={null}
      />,
    );
    await goToReview(user);
    expect(screen.getByText(/artifact_undeliverable/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Launch job" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).not.toHaveBeenCalled();
  });

  // A control plane that does not publish the field must not block a launch:
  // an absent check passes open, like the other four.
  it("launches when the published field is absent", async () => {
    const user = userEvent.setup();
    render(
      <LaunchForm
        catalog={catalog}
        preview={preview({ trainer_image_published: undefined })}
        surface={null}
      />,
    );
    await goToReview(user);
    expect(screen.getByRole("button", { name: "Launch job" })).toBeEnabled();
  });

  it("launches with the chosen model and opens the job's own page", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    createJobMock.mockResolvedValueOnce({ id: "job_abc123" });
    await goToReview(user);
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalledWith({
      dataset_id: "ds_abc123",
      base_model: "qwen3-4b",
      hyperparameters: {},
      overrides: [],
      delivery: [],
    });
    await vi.waitFor(() =>
      expect(push).toHaveBeenCalledWith("/jobs/job_abc123"),
    );
  });

  it("launches with the delivery formats the user asked for (issue #74)", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    createJobMock.mockResolvedValueOnce({ id: "job_abc123" });
    await goToReview(user);
    // The delivery choices are offered by what each format is for.
    await user.click(
      screen.getByRole("checkbox", { name: /Merged model/ }),
    );
    await user.click(
      screen.getByRole("checkbox", { name: /Quantised local format/ }),
    );
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalledWith(
      expect.objectContaining({ delivery: ["merged", "quantised"] }),
    );
  });

  it("greys out a format the published image cannot produce (issue #74 follow-up)", async () => {
    const user = userEvent.setup();
    render(
      <LaunchForm
        catalog={catalog}
        preview={preview({
          delivery_formats: [
            {
              id: "merged",
              what_for:
                "The base model with your trained change built into its full weights. Serve it directly.",
              producible: true,
            },
            {
              id: "quantised",
              what_for:
                "A compact local-inference version of the merged model. Run it on your own machine.",
              producible: false,
            },
          ],
        })}
        surface={null}
      />,
    );
    createJobMock.mockResolvedValueOnce({ id: "job_abc123" });
    await goToReview(user);

    const merged = screen.getByRole("checkbox", { name: /Merged model/ });
    const quantised = screen.getByRole("checkbox", {
      name: /Quantised local format/,
    });
    expect(merged).toBeEnabled();
    expect(quantised).toBeDisabled();
    expect(
      screen.getByText(/cannot produce this format/),
    ).toBeInTheDocument();

    // Ticking the producible format alone still launches -- the gate blocks
    // one checkbox, not the whole step.
    await user.click(merged);
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalledWith(
      expect.objectContaining({ delivery: ["merged"] }),
    );
  });

  it("launches with a model chosen after arrival", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    await user.click(screen.getByRole("radio", { name: /Qwen\/Qwen3-8B/ }));
    await goToReview(user);
    createJobMock.mockResolvedValueOnce({ id: "job_def456" });
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalledWith(
      expect.objectContaining({ base_model: "qwen3-8b" }),
    );
  });

  it("keeps a refused launch's stable code on the page", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    createJobMock.mockRejectedValueOnce(
      new ApiError(
        400,
        "unknown_model",
        "'gpt-9' is not in the catalog.",
      ),
    );
    await goToReview(user);
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("unknown_model");
    expect(alert).toHaveTextContent("not in the catalog");
    expect(push).not.toHaveBeenCalled();
    // Still here, still able to act: go back and pick the other model.
    await user.click(screen.getByRole("button", { name: "Back to hardware" }));
    await user.click(screen.getByRole("button", { name: "Back to hyperparameters" }));
    await user.click(screen.getByRole("button", { name: "Back to sources" }));
    expect(
      screen.getByRole("radio", { name: /Qwen\/Qwen3-8B/ }),
    ).toBeEnabled();
  });

  it("survives an unreachable server without losing the refusal", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    createJobMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    await goToReview(user);
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(screen.getByRole("alert")).toHaveTextContent("network_error");
  });

  it("disables the action while the launch is in flight", async () => {
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    let resolve!: (v: unknown) => void;
    createJobMock.mockReturnValueOnce(
      new Promise((res) => {
        resolve = res;
      }),
    );
    await goToReview(user);
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(
      screen.getByRole("button", { name: /launching/i }),
    ).toBeDisabled();
    resolve({});
  });

  it("fetches and shows the selected model's quote: a duration range, not a point", async () => {
    getQuoteMock.mockResolvedValue(quote());
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    await goToHardware(user);
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
    expect(screen.getByText(/00:03:58–00:18:59/)).toBeVisible();
    expect(screen.getByText(/INR 2\.74 – INR 13\.08/)).toBeVisible();
  });

  it("shows the per-phase breakdown of cost and duration", async () => {
    getQuoteMock.mockResolvedValue(quote());
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    await goToHardware(user);
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Cost and time estimate" }),
      ).toBeVisible(),
    );
    for (const phase of ["provisioning", "readiness", "image_pull", "model_download", "training", "teardown"]) {
      expect(screen.getByText(phase)).toBeVisible();
    }
    expect(screen.getByText(/00:01:27–00:03:03/)).toBeVisible(); // image pull
  });

  it("re-fetches the quote when the selected model changes", async () => {
    getQuoteMock.mockResolvedValue(quote());
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    getQuoteMock.mockResolvedValue(
      quote({
        duration_low_s: 400,
        duration_high_s: 2000,
        cost_low_minor: 460,
        cost_high_minor: 2300,
      }),
    );
    await user.click(screen.getByRole("radio", { name: /Qwen\/Qwen3-8B/ }));
    await goToHardware(user);
    await waitFor(() =>
      expect(getQuoteMock).toHaveBeenLastCalledWith({
        dataset_id: "ds_abc123",
        base_model: "qwen3-8b",
      }),
    );
    await waitFor(() =>
      expect(screen.getByText(/00:06:40–00:33:20/)).toBeVisible(),
    );
    expect(screen.getByText(/INR 4\.60 – INR 23\.00/)).toBeVisible();
  });

  it("still offers the launch when a quote is absent", async () => {
    getQuoteMock.mockResolvedValue(null);
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    // The estimate is shown as unavailable on the hardware step, and the
    // launch is still offered on review: an estimate warns, it never blocks.
    await goToHardware(user);
    await waitFor(() =>
      expect(screen.getByText(/could not be computed/i)).toBeVisible(),
    );
    await user.click(
      screen.getByRole("button", { name: "Continue to review" }),
    );
    createJobMock.mockResolvedValueOnce({ id: "job_abc123" });
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalled();
  });

  // --- the plan is editable (issue #79) -------------------------------------

  it("re-requests the plan from the server when a decision is overridden", async () => {
    getQuoteMock.mockResolvedValue(quote());
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
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
        hyperparameters: {},
      }),
    );
    // The recomputed plan is what is shown, and the override is marked.
    expect(screen.getByText("you changed this")).toBeVisible();
  });

  it("shows a refusal beside the plan when an override cannot be honoured", async () => {
    getQuoteMock.mockResolvedValue(quote());
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
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
      ).toHaveValue("qlora"),
    );
  });

  it("launches with the pinned decisions frozen into the request", async () => {
    getQuoteMock.mockResolvedValue(quote());
    const user = userEvent.setup();
    render(<LaunchForm catalog={catalog} preview={preview()} surface={null} />);
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
      ).toBeVisible(),
    );
    recomputeQuoteMock.mockResolvedValue(quote());
    await user.selectOptions(
      screen.getByRole("combobox", { name: "method override" }),
      "lora",
    );
    await waitFor(() => expect(recomputeQuoteMock).toHaveBeenCalled());
    createJobMock.mockResolvedValueOnce({ id: "job_abc123" });
    await user.click(
      screen.getByRole("button", { name: "Continue to hardware" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Continue to review" }),
    );
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalledWith({
      dataset_id: "ds_abc123",
      base_model: "qwen3-4b",
      hyperparameters: {},
      overrides: [{ decision: "method", value: "lora" }],
      delivery: [],
    });
  });

  // --- the advanced surface (issue #80) ------------------------------------

  it("re-requests the plan when an advanced setting is changed", async () => {
    const user = userEvent.setup();
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={surface()} />,
    );
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
      ).toBeVisible(),
    );
    // The edit happens in place on the hyperparameter card: the advanced
    // disclosure below only lists what Temper does not offer yet.
    recomputeQuoteMock.mockResolvedValue(quote());
    const lr = screen.getByRole("spinbutton", { name: "learning_rate override" });
    await user.clear(lr);
    await user.type(lr, "0.0001");
    await user.tab();
    await waitFor(() =>
      expect(recomputeQuoteMock).toHaveBeenCalledWith({
        dataset_id: "ds_abc123",
        base_model: "qwen3-4b",
        overrides: [],
        hyperparameters: { learning_rate: "0.0001" },
      }),
    );
    // The frozen-spec preview reflects the override, so what the page says a
    // launch will freeze is what will actually freeze: the card input holds
    // the edited value.
    expect(
      screen.getByRole("spinbutton", { name: "learning_rate override" }),
    ).toHaveValue(0.0001);
  });

  it("edits an exposed hyperparameter in place and offers the way back", async () => {
    const user = userEvent.setup();
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={surface()} />,
    );
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
      ).toBeVisible(),
    );
    // num_epochs is exposed, so its card row is an input populated with the
    // default; editing re-requests the plan and marks the row.
    recomputeQuoteMock.mockResolvedValue(quote());
    const epochs = screen.getByRole("spinbutton", {
      name: "num_epochs override",
    });
    expect(epochs).toHaveValue(3);
    await user.clear(epochs);
    await user.type(epochs, "5");
    await user.tab();
    await waitFor(() =>
      expect(recomputeQuoteMock).toHaveBeenCalledWith({
        dataset_id: "ds_abc123",
        base_model: "qwen3-4b",
        overrides: [],
        hyperparameters: { num_epochs: "5" },
      }),
    );
    expect(screen.getByText("you changed this")).toBeVisible();
    // The way back clears the override and re-requests the plain quote.
    await user.click(screen.getByRole("button", { name: "Use the default" }));
    await waitFor(() =>
      expect(getQuoteMock).toHaveBeenLastCalledWith({
        dataset_id: "ds_abc123",
        base_model: "qwen3-4b",
      }),
    );
    await waitFor(() =>
      expect(
        screen.getByRole("spinbutton", { name: "num_epochs override" }),
      ).toHaveValue(3),
    );
  });

  it("keeps calculated keys read-only with the value shown", async () => {
    const user = userEvent.setup();
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={surface()} />,
    );
    await goToHyperparameters(user);
    // lora_r is not in the fixture surface's exposed tier, so its card row
    // is a shown value, never a control the launch would refuse.
    expect(
      screen.queryByRole("spinbutton", { name: "lora_r override" }),
    ).not.toBeInTheDocument();
    const card = screen.getByRole("region", {
      name: "LoRA & Adapter Architecture",
    });
    expect(within(card).getByText("16")).toBeVisible();
  });

  it("explains a hyperparameter behind a '?' tooltip", async () => {
    const user = userEvent.setup();
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={surface()} />,
    );
    await goToHyperparameters(user);
    // The reason and the failure mode stay off the page until asked for.
    expect(
      screen.queryByText(/The peak learning rate for the cosine schedule/i),
    ).not.toBeInTheDocument();
    await user.hover(screen.getByRole("button", { name: "About learning_rate" }));
    expect(
      await screen.findByText(/The peak learning rate for the cosine schedule/i),
    ).toBeVisible();
    expect(
      screen.getByText(/diverges to NaN partway through a paid run/i),
    ).toBeVisible();
  });

  it("explains a plan decision behind a '?' dialog", async () => {
    const user = userEvent.setup();
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={surface()} />,
    );
    await goToHyperparameters(user);
    // The card keeps heading, value and control; the reason opens on demand.
    expect(screen.getByRole("heading", { name: "method" })).toBeVisible();
    expect(
      screen.getByRole("combobox", { name: "method override" }),
    ).toBeVisible();
    expect(
      screen.queryByText(/the trainer can execute qlora today/i),
    ).not.toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "About the method decision" }),
    );
    expect(
      await screen.findByText(/the trainer can execute qlora today/i),
    ).toBeVisible();
  });

  it("nests the training decisions inside the form cards, with no decisions section", async () => {
    const user = userEvent.setup();
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={surface()} />,
    );
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
      ).toBeVisible(),
    );
    // No dedicated section: method and precision sit in the adapter card,
    // sequence length in the sequence card.
    expect(
      screen.queryByRole("heading", { name: "Why this configuration" }),
    ).not.toBeInTheDocument();
    const adapter = screen.getByRole("region", {
      name: "LoRA & Adapter Architecture",
    });
    expect(
      within(adapter).getByRole("combobox", { name: "method override" }),
    ).toBeVisible();
    const sequence = screen.getByRole("region", {
      name: "Sequence & Context Length",
    });
    expect(
      within(sequence).getByRole("spinbutton", {
        name: "sequence length override",
      }),
    ).toBeVisible();
  });

  it("resets every override to the smart defaults in one act", async () => {
    const user = userEvent.setup();
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={surface()} />,
    );
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
      ).toBeVisible(),
    );
    // Pin a decision and a hyperparameter first, so the reset has something
    // to clear.
    recomputeQuoteMock.mockResolvedValue(
      quote({
        decisions: [
          {
            decision: "method",
            chosen: "lora",
            constraint: "you chose lora.",
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
    const lr = screen.getByRole("spinbutton", { name: "learning_rate override" });
    await user.clear(lr);
    await user.type(lr, "0.0001");
    await user.tab();
    await waitFor(() => expect(recomputeQuoteMock).toHaveBeenCalled());
    expect(screen.getAllByText("you changed this").length).toBe(2);
    // Reset returns to the smart defaults: one clean quote request, no
    // markers, inputs back on their defaults.
    await user.click(screen.getByRole("button", { name: "Reset defaults" }));
    await waitFor(() =>
      expect(getQuoteMock).toHaveBeenLastCalledWith({
        dataset_id: "ds_abc123",
        base_model: "qwen3-4b",
      }),
    );
    expect(screen.queryByText("you changed this")).not.toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: "method override" }),
    ).toHaveValue("qlora");
    await waitFor(() =>
      expect(
        screen.getByRole("spinbutton", { name: "learning_rate override" }),
      ).toHaveValue(0.0002),
    );
  });

  it("overrides the hardware from its option cards", async () => {
    const user = userEvent.setup();
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={surface()} />,
    );
    await goToHardware(user);
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Cost and time estimate" }),
      ).toBeVisible(),
    );
    // Temper's default starts selected and badged.
    const group = screen.getByRole("radiogroup", { name: "hardware options" });
    expect(within(group).getByRole("radio", { name: /L4/ })).toBeChecked();
    expect(within(group).getByText("Recommended")).toBeVisible();
    // Picking another card pins the override and recomputes from the server.
    recomputeQuoteMock.mockResolvedValue(
      quote({
        decisions: [
          {
            decision: "hardware",
            chosen: "H100",
            constraint: "you chose H100.",
            alternatives: [],
            overridden: true,
          },
        ],
      }),
    );
    await user.click(within(group).getByRole("radio", { name: /H100/ }));
    await waitFor(() =>
      expect(recomputeQuoteMock).toHaveBeenCalledWith({
        dataset_id: "ds_abc123",
        base_model: "qwen3-4b",
        overrides: [{ decision: "hardware", value: "H100" }],
        hyperparameters: {},
      }),
    );
    expect(screen.getByText("you changed this")).toBeVisible();
  });

  it("reselecting the predictor's default unpins instead of re-pinning it", async () => {
    const user = userEvent.setup();
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={surface()} />,
    );
    await goToHardware(user);
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Cost and time estimate" }),
      ).toBeVisible(),
    );
    const group = screen.getByRole("radiogroup", { name: "hardware options" });
    // Pin another GPU: the recomputed quote echoes the pin as its choice.
    recomputeQuoteMock.mockResolvedValue(
      quote({
        decisions: [
          {
            decision: "hardware",
            chosen: "H100",
            constraint: "you chose H100.",
            alternatives: [],
            overridden: true,
          },
        ],
      }),
    );
    await user.click(within(group).getByRole("radio", { name: /H100/ }));
    await waitFor(() => expect(recomputeQuoteMock).toHaveBeenCalled());
    // Recommended stays on the predictor's default, and the copy names it.
    expect(
      within(group).getByRole("radio", { name: "L4 Recommended" }),
    ).toBeVisible();
    expect(
      screen.getByRole("region", { name: "Select hardware" }),
    ).toHaveTextContent(/Temper's default was/);
    // Reselecting the default clears the override: one clean quote request,
    // no marker, the default checked again.
    await user.click(
      within(group).getByRole("radio", { name: "L4 Recommended" }),
    );
    await waitFor(() =>
      expect(getQuoteMock).toHaveBeenLastCalledWith({
        dataset_id: "ds_abc123",
        base_model: "qwen3-4b",
      }),
    );
    expect(screen.queryByText("you changed this")).not.toBeInTheDocument();
    expect(within(group).getByRole("radio", { name: /L4/ })).toBeChecked();
  });

  it("reverts an advanced override the server refuses, and shows the refusal", async () => {
    const user = userEvent.setup();
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={surface()} />,
    );
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
      ).toBeVisible(),
    );
    const details = screen.getByRole("group", { name: "Advanced settings" });
    await user.click(details.querySelector("summary") as HTMLElement);
    // A hyperparameter the surface refuses (here, one that makes the job
    // infeasible) is refused with its stable code, and the control reverts to
    // the default rather than leaving a lie in the form.
    recomputeQuoteMock.mockRejectedValue(
      new ApiError(
        400,
        "configuration_does_not_fit",
        "qlora on a L4 predicts 66.9 GB peak, which the 24.0 GB L4 cannot hold.",
      ),
    );
    const epochs = screen.getByRole("spinbutton", { name: "num_epochs override" });
    await user.clear(epochs);
    await user.type(epochs, "999");
    await user.tab();
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("configuration_does_not_fit");
    await waitFor(() =>
      expect(
        screen.getByRole("spinbutton", { name: "num_epochs override" }),
      ).toHaveValue(3),
    );
  });

  it("launches with the advanced overrides frozen into the request", async () => {
    const user = userEvent.setup();
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={surface()} />,
    );
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
      ).toBeVisible(),
    );
    const details = screen.getByRole("group", { name: "Advanced settings" });
    await user.click(details.querySelector("summary") as HTMLElement);
    recomputeQuoteMock.mockResolvedValue(quote());
    const lr = screen.getByRole("spinbutton", { name: "learning_rate override" });
    await user.clear(lr);
    await user.type(lr, "0.0001");
    await user.tab();
    await waitFor(() => expect(recomputeQuoteMock).toHaveBeenCalled());
    createJobMock.mockResolvedValueOnce({ id: "job_abc123" });
    await user.click(
      screen.getByRole("button", { name: "Continue to hardware" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Continue to review" }),
    );
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalledWith({
      dataset_id: "ds_abc123",
      base_model: "qwen3-4b",
      hyperparameters: { learning_rate: "0.0001" },
      overrides: [],
      delivery: [],
    });
  });

  it("offers the launch with the defaults when the surface cannot be loaded", async () => {
    const user = userEvent.setup();
    // A surface-load failure never blocks the launch: the job is still
    // offered with the defaults, which is what a first-time user gets anyway.
    getQuoteMock.mockResolvedValue(quote());
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={null} />,
    );
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
      ).toBeVisible(),
    );
    expect(
      screen.queryByRole("group", { name: "Advanced settings" }),
    ).toBeNull();
    await user.click(
      screen.getByRole("button", { name: "Continue to hardware" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Continue to review" }),
    );
    createJobMock.mockResolvedValueOnce({ id: "job_abc123" });
    await user.click(screen.getByRole("button", { name: "Launch job" }));
    expect(createJobMock).toHaveBeenCalledWith({
      dataset_id: "ds_abc123",
      base_model: "qwen3-4b",
      hyperparameters: {},
      overrides: [],
      delivery: [],
    });
  });

  // --- the dataset is picked on the Sources step ---------------------------

  function datasetRecord(
    overrides: Partial<DatasetRecord> = {},
  ): DatasetRecord {
    return {
      id: "ds_def456",
      filename: "second.jsonl",
      created_at: 1756160500,
      updated_at: 1756160500,
      status: "valid",
      report: {
        valid: true,
        row_count: 20,
        usable_rows: 18,
        errors: [],
        warnings: [],
        preview: [],
      },
      token_count_status: "counted",
      ...overrides,
    } as DatasetRecord;
  }

  it("offers only ready datasets as named choices beside the current one", () => {
    render(
      <LaunchForm
        catalog={catalog}
        preview={preview()}
        surface={null}
        datasets={[
          datasetRecord(),
          datasetRecord({
            id: "ds_bad",
            filename: "bad.jsonl",
            report: {
              valid: false,
              row_count: 3,
              usable_rows: 0,
              errors: [],
              warnings: [],
              preview: [],
            },
          }),
        ]}
      />,
    );
    // The current dataset is the checked choice; the other ready one is
    // offered beside it, with its usable rows.
    expect(screen.getByRole("radio", { name: /^d\.jsonl/ })).toBeChecked();
    const other = screen.getByRole("radio", { name: /second\.jsonl/ });
    expect(other).not.toBeChecked();
    expect(other.closest("label")).toHaveTextContent("18 of 20 rows");
    // A dataset that needs fixes is not offered here at all — its report,
    // not this picker, is where it gets fixed.
    expect(
      screen.queryByRole("link", { name: /bad\.jsonl/ }),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/needs fixes/)).not.toBeInTheDocument();
  });

  it("switching datasets reloads the preview and resets the tune state", async () => {
    getQuoteMock.mockResolvedValue(quote());
    const user = userEvent.setup();
    const second = preview({
      dataset: {
        id: "ds_def456",
        filename: "second.jsonl",
        created_at: 1756160500,
        status: "valid",
      },
    });
    getPreviewMock.mockResolvedValue(second);
    render(
      <LaunchForm
        catalog={catalog}
        preview={preview()}
        surface={null}
        datasets={[datasetRecord()]}
      />,
    );
    // Pin something first, so the reset has something to clear.
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
      ).toBeVisible(),
    );
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
    await waitFor(() => expect(recomputeQuoteMock).toHaveBeenCalled());
    expect(screen.getByText("you changed this")).toBeVisible();

    // Back to Sources, switch datasets: the plan below is the new
    // dataset's, the override is gone, and the quote is re-requested for it.
    await user.click(screen.getByRole("button", { name: "Back to sources" }));
    await user.click(screen.getByRole("radio", { name: /second\.jsonl/ }));
    await waitFor(() =>
      expect(getPreviewMock).toHaveBeenCalledWith({
        dataset_id: "ds_def456",
      }),
    );
    await waitFor(() =>
      expect(getQuoteMock).toHaveBeenLastCalledWith({
        dataset_id: "ds_def456",
        base_model: "qwen3-4b",
      }),
    );
    await goToHyperparameters(user);
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "method override" }),
      ).toBeVisible(),
    );
    expect(screen.queryByText("you changed this")).not.toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: "method override" }),
      ).toHaveValue("qlora");
  });

  it("a refused dataset reverts with its stable code", async () => {
    const user = userEvent.setup();
    getPreviewMock.mockRejectedValueOnce(
      new ApiError(400, "dataset_invalid", "This dataset cannot be trained on."),
    );
    render(
      <LaunchForm
        catalog={catalog}
        preview={preview()}
        surface={null}
        datasets={[datasetRecord()]}
      />,
    );
    await user.click(screen.getByRole("radio", { name: /second\.jsonl/ }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("dataset_invalid");
    // Reverted to the last good dataset rather than leaving a lie checked.
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: /^d\.jsonl/ })).toBeChecked(),
    );
  });

  // --- starting with no dataset (/jobs/new with no ?dataset_id=) -----------

  it("shows the wizard with no dataset picked, and a disabled Continue", () => {
    render(
      <LaunchForm
        catalog={catalog}
        preview={null}
        surface={null}
        datasets={[datasetRecord()]}
      />,
    );
    expect(
      screen.getByRole("heading", { name: "Model & dataset" }),
    ).toBeVisible();
    expect(
      screen.getByRole("radio", { name: /second\.jsonl/ }),
    ).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Continue to hyperparameters" }),
    ).toBeDisabled();
    // No spec, no estimate, no launch without a dataset.
    expect(
      screen.queryByRole("button", { name: "Launch job" }),
    ).not.toBeInTheDocument();
  });

  it("picking a dataset loads its preview and enables Continue", async () => {
    const user = userEvent.setup();
    getPreviewMock.mockResolvedValue(
      preview({
        dataset: {
          id: "ds_def456",
          filename: "second.jsonl",
          created_at: 1756160500,
          status: "valid",
        },
      }),
    );
    render(
      <LaunchForm
        catalog={catalog}
        preview={null}
        surface={null}
        datasets={[datasetRecord()]}
      />,
    );
    const cont = screen.getByRole("button", {
      name: "Continue to hyperparameters",
    });
    expect(cont).toBeDisabled();
    await user.click(screen.getByRole("radio", { name: /second\.jsonl/ }));
    await waitFor(() =>
      expect(getPreviewMock).toHaveBeenCalledWith({
        dataset_id: "ds_def456",
      }),
    );
    await waitFor(() => expect(cont).toBeEnabled());
  });

  // --- adding a dataset without leaving Sources ----------------------------

  it("switches to an inline add pane with upload and import tabs", async () => {
    const user = userEvent.setup();
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={null} />,
    );
    await user.click(
      screen.getByRole("button", { name: "Upload / Import" }),
    );
    // No dialog, no navigation: the same forms render inline.
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(
      screen.getByLabelText("Dataset file (.jsonl)"),
    ).toBeVisible();
    await user.click(
      screen.getByRole("tab", { name: "Import from Hugging Face" }),
    );
    expect(
      screen.getByRole("button", { name: "Import and validate" }),
    ).toBeVisible();
  });

  it("an uploaded dataset joins the list, selected, once validation lands", async () => {    const user = userEvent.setup();
    const fresh = datasetRecord({
      id: "ds_fresh",
      filename: "fresh.jsonl",
      created_at: 1756160600,
    });
    uploadDatasetMock.mockResolvedValue({
      id: "ds_fresh",
      filename: "fresh.jsonl",
      status: "validating",
    });
    // Still validating on the immediate check, reported on the next poll.
    getDatasetMock
      .mockResolvedValueOnce({ ...fresh, report: null, status: "validating" })
      .mockResolvedValue(fresh);
    getPreviewMock.mockImplementation(({ dataset_id }: { dataset_id: string }) =>
      dataset_id === "ds_fresh"
        ? preview({
            dataset: {
              id: "ds_fresh",
              filename: "fresh.jsonl",
              created_at: 1756160600,
              status: "valid",
            },
          })
        : preview(),
    );
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={null} />,
    );
    await user.click(
      screen.getByRole("button", { name: "Upload / Import" }),
    );
    const file = new File(['{"messages": []}'], "fresh.jsonl", {
      type: "application/json",
    });
    await user.upload(screen.getByLabelText("Dataset file (.jsonl)"), file);
    await user.click(
      screen.getByRole("button", { name: "Upload and validate" }),
    );
    // The pane switches back and the arrival validates in the list…
    await waitFor(() =>
      expect(screen.getByText("Validating…")).toBeVisible(),
    );
    // …then joins the list, selected, with its preview loaded.
    await waitFor(
      () =>
        expect(
          screen.getByRole("radio", { name: /^fresh\.jsonl/ }),
        ).toBeChecked(),
      { timeout: 5000 },
    );
    await waitFor(() =>
      expect(getPreviewMock).toHaveBeenCalledWith({
        dataset_id: "ds_fresh",
      }),
    );
  });

  it("opens the validation report in place instead of navigating away", async () => {    const user = userEvent.setup();
    getDatasetMock.mockResolvedValue(
      datasetRecord({ id: "ds_abc123", filename: "d.jsonl" }),
    );
    render(
      <LaunchForm catalog={catalog} preview={preview()} surface={null} />,
    );
    await user.click(
      screen.getByRole("button", { name: "View validation report" }),
    );
    // The full report renders in the dialog — same component as the report
    // page — instead of navigating away.
    await screen.findByRole("heading", { name: "d.jsonl" });
    expect(push).not.toHaveBeenCalled();
    // Closing returns to the wizard where it was.
    await user.keyboard("{Escape}");
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Continue to hyperparameters" }),
      ).toBeVisible(),
    );
  });

  it("keeps the dataset order stable when switching", async () => {
    const user = userEvent.setup();
    getPreviewMock.mockResolvedValue(
      preview({
        dataset: {
          id: "ds_def456",
          filename: "second.jsonl",
          created_at: 1756160500,
          status: "valid",
        },
      }),
    );
    render(
      <LaunchForm
        catalog={catalog}
        preview={preview()}
        surface={null}
        datasets={[
          datasetRecord(),
          datasetRecord({
            id: "ds_abc123",
            filename: "d.jsonl",
            created_at: 1756160400,
            updated_at: 1756160400,
          }),
        ]}
      />,
    );
    // Newest first, with the preview's older dataset in its natural place —
    // and switching selections must not reshuffle the grid.
    const order = () =>
      Array.from(
        document.querySelectorAll('#dataset-list input[type="radio"]'),
      ).map((el) => (el as HTMLInputElement).value);
    expect(order()).toEqual(["ds_def456", "ds_abc123"]);
    await user.click(screen.getByRole("radio", { name: /second\.jsonl/ }));
    await waitFor(() =>
      expect(getPreviewMock).toHaveBeenCalledWith({
        dataset_id: "ds_def456",
      }),
    );
    expect(order()).toEqual(["ds_def456", "ds_abc123"]);
    expect(screen.getByRole("radio", { name: /second\.jsonl/ })).toBeChecked();
  });
});
