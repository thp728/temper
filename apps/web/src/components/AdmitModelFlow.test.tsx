import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LaunchForm from "@/components/LaunchForm";
import type {
  AdmittedModel,
  CatalogEntry,
  JobSpecPreview,
  ModelCatalog,
  Quote,
} from "@/lib/api/generated/client";

const push = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
}));

// The generated client is the seam: the probe and the launch are mocked here,
// and the real client is exercised by the Playwright journeys.
const createJobMock = vi.hoisted(() => vi.fn());
const getQuoteMock = vi.hoisted(() => vi.fn());
const recomputeQuoteMock = vi.hoisted(() => vi.fn());
const probeModelMock = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api/generated/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/generated/client")>()),
  createJobV1JobsPost: createJobMock,
  getQuoteV1QuotesGet: getQuoteMock,
  recomputeQuoteV1QuotesPost: recomputeQuoteMock,
  probeModelV1ModelsProbePost: probeModelMock,
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
  models: [entry()],
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
        name: "training",
        duration_low_s: 80,
        duration_high_s: 800,
        cost_low_minor: 92,
        cost_high_minor: 918,
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
    decisions: [],
    override_options: {},
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

const PINNED = "c".repeat(40);

function admitted(
  overrides: Partial<AdmittedModel> = {},
  probe: Partial<AdmittedModel["probe"]> = {},
): AdmittedModel {
  return {
    id: "m_imported",
    repo: "org/imported",
    revision: PINNED,
    created_at: 1756160400,
    probe: {
      repo: "org/imported",
      revision: PINNED,
      ok: true,
      verdict: "usable",
      architecture: "qwen3",
      params_b: 4.0,
      context_length: 40960,
      license: "apache-2.0",
      is_moe: false,
      findings: [],
      memory: {
        fits: true,
        peak_gb: 5.35,
        card: "L4",
        capacity_gb: 24,
        headroom_gb: 18.65,
      },
      ...probe,
    },
    ...overrides,
  };
}

function renderForm(admittedModels: AdmittedModel[] = []) {
  return render(
    <LaunchForm
      catalog={catalog}
      preview={preview()}
      surface={null}
      admitted={admittedModels}
    />,
  );
}

describe("admitted models on the launch screen (issue #58)", () => {
  beforeEach(() => {
    createJobMock.mockReset();
    getQuoteMock.mockReset().mockResolvedValue(quote());
    recomputeQuoteMock.mockReset();
    probeModelMock.mockReset();
  });

  it("shows an admitted model with its probe result, selectable when usable", async () => {
    renderForm([admitted()]);

    const radio = screen.getByRole("radio", { name: /org\/imported/ });
    expect(radio).toBeEnabled();
    expect(screen.getByText("Usable")).toBeInTheDocument();
    expect(screen.getByText("Compatibility probe:")).toBeInTheDocument();
    expect(screen.getByTestId("probe-result")).toHaveTextContent("5.35 GB");

    await userEvent.click(radio);
    await waitFor(() =>
      expect(getQuoteMock).toHaveBeenCalledWith({
        dataset_id: "ds_abc123",
        base_model: "m_imported",
      }),
    );
  });

  it("shows a blocked model's reasons and refuses to select it", () => {
    renderForm([
      admitted(
        {},
        {
          ok: false,
          verdict: "blocked",
          findings: [
            {
              code: "missing_chat_template",
              severity: "block",
              message:
                "this model publishes no chat template, so chat data cannot be formatted",
            },
          ],
        },
      ),
    ]);

    const radio = screen.getByRole("radio", { name: /org\/imported/ });
    expect(radio).toBeDisabled();
    expect(screen.getByText("Blocked")).toBeInTheDocument();
    expect(screen.getByText(/no chat template/)).toBeInTheDocument();
  });

  it("probes a model from the disclosure, shows the result and launches with it", async () => {
    renderForm();
    probeModelMock.mockResolvedValue(admitted());

    await userEvent.click(
      screen.getByText("Use a model outside the catalog"),
    );
    await userEvent.type(
      screen.getByLabelText("Public repository"),
      "org/imported",
    );
    await userEvent.type(
      screen.getByLabelText("Pinned revision"),
      PINNED,
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Probe and admit" }),
    );

    await waitFor(() =>
      expect(probeModelMock).toHaveBeenCalledWith({
        repo: "org/imported",
        revision: PINNED,
      }),
    );
    const radio = await screen.findByRole("radio", { name: /org\/imported/ });
    expect(radio).toBeChecked();
    expect(screen.getByText("Usable")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Launch job" }));
    await waitFor(() =>
      expect(createJobMock).toHaveBeenCalledWith({
        dataset_id: "ds_abc123",
        base_model: "m_imported",
        hyperparameters: {},
        overrides: [],
      }),
    );
  });

  it("shows the probe refusal's stable code when admission is refused", async () => {
    renderForm();
    probeModelMock.mockRejectedValue(
      new ApiError(
        400,
        "unpinned_revision",
        "'main' is not a pinned revision.",
      ),
    );

    await userEvent.click(
      screen.getByText("Use a model outside the catalog"),
    );
    await userEvent.type(
      screen.getByLabelText("Public repository"),
      "org/imported",
    );
    await userEvent.type(screen.getByLabelText("Pinned revision"), "main");
    await userEvent.click(
      screen.getByRole("button", { name: "Probe and admit" }),
    );

    await waitFor(() =>
      expect(screen.getByText("unpinned_revision")).toBeInTheDocument(),
    );
    expect(screen.getByText(/'main' is not a pinned revision\./)).toBeInTheDocument();
  });
});
