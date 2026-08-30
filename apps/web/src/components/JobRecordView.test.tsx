import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import JobRecordView from "@/components/JobRecordView";
import type { JobEvent, JobRecord, Quote } from "@/lib/api/generated/client";

// The assertions are what a user reads on a job's record: how it ended, what
// it produced, why it failed, and that a cancellation is a decision rather
// than a defect. The port's behaviour, pinned.

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    id: "job_abc123def456",
    dataset_id: "ds_xyz789",
    base_model: "qwen3-4b",
    base_revision: "a".repeat(40),
    status: "complete",
    cancel_requested: false,
    created_at: new Date(2025, 7, 26, 1, 0, 0).getTime() / 1000,
    started_at: new Date(2025, 7, 26, 1, 0, 5).getTime() / 1000,
    finished_at: new Date(2025, 7, 26, 1, 4, 35).getTime() / 1000,
    gpu_type: "L4",
    price_per_hour: 41.31,
    currency: "INR",
    warnings: [],
    result: { ok: true },
    artifact: null,
    ...overrides,
  };
}

function event(overrides: Partial<JobEvent> = {}): JobEvent {
  return {
    id: 1,
    job_id: "job_abc123def456",
    ts: new Date(2025, 7, 26, 1, 0, 5).getTime() / 1000,
    kind: "state",
    message: "queued",
    data: null,
    ...overrides,
  };
}

function quote(): Quote {
  return {
    currency: "INR",
    minor_unit: 100,
    dataset_id: "ds_xyz789",
    dataset_created_at: new Date(2025, 7, 26, 1, 0, 0).getTime() / 1000,
    base_revision: "a".repeat(40),
    token_count: null,
    expires_at: new Date(2025, 7, 27, 1, 0, 0).getTime() / 1000,
    phases: [
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
    peak_memory_gb: 5.4,
  };
}

describe("JobRecordView", () => {
  it("shows the state with its spend and latest loss beside it", () => {
    render(
      <JobRecordView
        job={job()}
        datasetFilename="support-chats.jsonl"
        events={[
          event(),
          event({
            id: 2,
            kind: "metric",
            message: "{'loss': 0.6931, 'step': 10}",
            data: { loss: 0.6931, step: 10 },
          }),
        ]}
      />,
    );

    // Label/value pairs are read as pairs; several say "—" so match precisely.
    const value = (label: string) =>
      screen.getByText(label).nextElementSibling?.textContent;
    expect(value("State")).toBe("complete");
    expect(value("Elapsed")).toBe("4m 30s");
    expect(value("Machine")).toBe("L4 at 41.31 INR/hr");
    expect(value("Latest loss")).toContain("0.6931");
    expect(value("Latest loss")).toContain("step 10");

    // The job is identifiable: dataset by name (linking to its report), model
    // and pinned revision in short form.
    expect(
      screen.getByRole("link", { name: "support-chats.jsonl" }),
    ).toHaveAttribute("href", "/datasets/ds_xyz789");
    expect(screen.getByText("qwen3-4b")).toBeVisible();
    expect(screen.getByText("a".repeat(12))).toBeVisible();
  });

  it("offers a completed job's artifact through the download route", () => {
    render(
      <JobRecordView
        job={job({
          artifact: {
            kind: "adapter",
            members: ["adapter_model.safetensors", "adapter_config.json"],
            bytes: 132,
            loading: "A PEFT adapter over the base model.",
          },
        })}
        events={[]}
      />,
    );
    const link = screen.getByRole("link", { name: /Download the artifact/ });
    expect(link).toHaveAttribute(
      "href",
      "/v1/jobs/job_abc123def456/artifact",
    );
    // The load path ships with the artifact (issue #32): the offer says how
    // to use what it hands over.
    expect(screen.getByText(/PEFT adapter over the base model/)).toBeVisible();
    expect(screen.getByText(/adapter_config\.json/)).toBeVisible();
  });

  it("offers each produced delivery format with its purpose (issue #74)", () => {
    render(
      <JobRecordView
        job={job({
          artifact: {
            kind: "adapter",
            members: ["adapter_model.safetensors"],
            bytes: 132,
            loading: "A PEFT adapter over the base model.",
          },
          delivery_formats: [
            {
              format: "merged",
              kind: "merged_model",
              what_for:
                "The base model with your trained change built into its full weights — serve it directly.",
              members: ["merged.tar.gz"],
              loading: "A complete model; load it directly.",
            },
            {
              format: "quantised",
              kind: "quantised_local",
              what_for:
                "A compact local-inference version of the merged model — run it on your own machine.",
              members: ["quantised.gguf"],
              loading: "A local-inference format for llama.cpp.",
            },
          ],
        })}
        events={[]}
      />,
    );
    // Each format is offered by what it is for, in plain language, with a
    // per-format download that carries the `format` query parameter.
    expect(screen.getByText("merged")).toBeVisible();
    expect(screen.getByText(/serve it directly/)).toBeVisible();
    expect(screen.getByRole("link", { name: "Download merged" })).toHaveAttribute(
      "href",
      "/v1/jobs/job_abc123def456/artifact?format=merged",
    );
    expect(screen.getByText("quantised")).toBeVisible();
    expect(screen.getByText(/run it on your own machine/)).toBeVisible();
    expect(
      screen.getByRole("link", { name: "Download quantised" }),
    ).toHaveAttribute(
      "href",
      "/v1/jobs/job_abc123def456/artifact?format=quantised",
    );
    // The canonical artifact download is still offered.
    expect(
      screen.getByRole("link", { name: /Download the artifact/ }),
    ).toHaveAttribute("href", "/v1/jobs/job_abc123def456/artifact");
  });

  it("names the artifact by its declared kind", () => {
    render(
      <JobRecordView
        job={job({
          artifact: {
            kind: "full_model",
            members: ["model.safetensors", "config.json"],
            bytes: 2200,
            loading: "A fully fine-tuned model: load it directly.",
          },
        })}
        events={[]}
      />,
    );
    expect(screen.getByRole("link", { name: /Download the artifact/ })).toBeVisible();
    expect(screen.getByText(/fully fine-tuned model/)).toBeVisible();
    expect(screen.getByText(/model\.safetensors/)).toBeVisible();
  });

  it("says when training finished but no artifact could be retrieved", () => {
    render(
      <JobRecordView job={job({ result: null, artifact: null })} events={[]} />,
    );
    expect(screen.getByText(/no artifact could be retrieved/i)).toBeVisible();
    expect(
      screen.queryByRole("link", { name: /Download the artifact/ }),
    ).toBeNull();
  });

  it("shows a failure's stable code with its reason in plain language", () => {
    render(
      <JobRecordView
        job={job({
          status: "failed",
          error_code: "gpu_stalled",
          error_message:
            "No output for 900.0s (stall limit); machine destroyed.",
          result: null,
        })}
        events={[
          event({ message: "[00:00:00] building trainer image", kind: "log" }),
        ]}
      />,
    );

    expect(screen.getByText("gpu_stalled")).toBeVisible();
    expect(screen.getByText(/stopped by a safety limit/)).toBeVisible();
    expect(
      screen.getByText(/No output for 900\.0s \(stall limit\)/),
    ).toBeVisible();
    // Proof that billing stopped reaches the page, in the machine's own words.
    expect(screen.getByText(/machine destroyed/)).toBeVisible();
    // What the job was doing before it died stays readable.
    expect(screen.getByText("[00:00:00] building trainer image")).toBeVisible();
    expect(
      screen.queryByRole("link", { name: /Download the artifact/ }),
    ).toBeNull();
  });

  it("explains an out-of-memory failure with the memory-recovery reason", () => {
    // Issue #35: a genuine memory exhaustion and an exhausted recovery carry
    // their own stable codes, explained in plain language -- and they are not
    // a divergence, so the divergence retry control is absent.
    render(
      <JobRecordView
        job={job({
          status: "failed",
          error_code: "memory_retries_exhausted",
          error_message:
            "The job ran out of device memory repeatedly, and every memory reduction the platform can apply has been tried.",
          result: null,
          attempts: [
            {
              attempt: 1,
              outcome: "failed",
              error_code: "training_oom",
              spec: {
                micro_batch_size: 8,
                gradient_accumulation_steps: 1,
                sequence_len: 2048,
                method: "qlora",
              },
              recovery: null,
            },
            {
              attempt: 2,
              outcome: "failed",
              error_code: "training_oom",
              spec: {
                micro_batch_size: 4,
                gradient_accumulation_steps: 2,
                sequence_len: 2048,
                method: "qlora",
              },
              recovery: null,
            },
          ],
        })}
        events={[]}
      />,
    );
    expect(screen.getByText("memory_retries_exhausted")).toBeVisible();
    // The explanation gives the recovery's plain-language reason: every
    // reduction was tried, then the run was stopped rather than retried
    // forever.
    expect(
      screen.getByText(/Try a smaller configuration, a shorter sequence length/),
    ).toBeVisible();
    // A memory exhaustion is not offered the divergence retry.
    expect(
      screen.queryByRole("button", { name: /Retry at half learning rate/ }),
    ).toBeNull();
  });

  it("tells the user a memory recovery happened and what changed", () => {
    // Issue #35: an out-of-memory failure retried automatically with the
    // effective batch preserved; the record's attempts list is what the user
    // reads for "a recovery happened, and this is what changed".
    render(
      <JobRecordView
        job={job({
          attempts: [
            {
              attempt: 1,
              outcome: "failed",
              error_code: "simulated_oom",
              spec: {
                micro_batch_size: 8,
                gradient_accumulation_steps: 1,
                sequence_len: 2048,
                method: "qlora",
              },
              recovery: {
                rung: "halve_batch",
                action:
                  "the per-step batch was halved (8->4) and accumulation doubled (1->2); the effective batch (8x1=8) is unchanged",
                effective_batch: 8,
              },
            },
            {
              attempt: 2,
              outcome: "complete",
              error_code: null,
              spec: {
                micro_batch_size: 4,
                gradient_accumulation_steps: 2,
                sequence_len: 2048,
                method: "qlora",
              },
              recovery: null,
            },
          ],
        })}
        events={[]}
      />,
    );
    expect(screen.getByText("Memory recovery")).toBeVisible();
    expect(screen.getByText(/retried automatically/)).toBeVisible();
    expect(screen.getByText(/effective batch was preserved/)).toBeVisible();
    expect(screen.getByText("Attempt 1")).toBeVisible();
    expect(screen.getByText("simulated_oom")).toBeVisible();
    expect(screen.getByText(/per-step batch was halved \(8->4\)/)).toBeVisible();
    expect(screen.getByText("Attempt 2")).toBeVisible();
    expect(screen.getAllByText("complete").length).toBeGreaterThan(0);
  });

  it("shows no memory-recovery banner for a single-attempt job", () => {
    render(<JobRecordView job={job()} events={[]} />);
    expect(
      screen.queryByRole("heading", { name: "Memory recovery" }),
    ).toBeNull();
  });

  it("does not present an exhausted recovery as a successful one", () => {
    // Every attempt failed (the ladder and the retry cap were exhausted): the
    // record reads as a failure, explained by the failure section, never as a
    // recovery that succeeded.
    render(
      <JobRecordView
        job={job({
          status: "failed",
          error_code: "memory_retries_exhausted",
          result: null,
          attempts: [
            {
              attempt: 1,
              outcome: "failed",
              error_code: "training_oom",
              recovery: null,
            },
            {
              attempt: 2,
              outcome: "failed",
              error_code: "training_oom",
              recovery: null,
            },
          ],
        })}
        events={[]}
      />,
    );
    expect(
      screen.queryByRole("heading", { name: "Memory recovery" }),
    ).toBeNull();
  });

  it("says an over-long job hit the ceiling, with its code and reason", () => {
    render(
      <JobRecordView
        job={job({
          status: "failed",
          error_code: "gpu_max_duration_exceeded",
          error_message: "Ran past the maximum duration a job may use.",
          result: null,
        })}
        events={[]}
      />,
    );

    expect(screen.getByText("gpu_max_duration_exceeded")).toBeVisible();
    // The safety limit is explained, not left as a code.
    expect(screen.getByText(/ran past the maximum duration a single job may use/i)).toBeVisible();
  });

  it("offers no cancel control once the job is terminal", () => {
    render(<JobRecordView job={job()} events={[]} />);
    expect(
      screen.queryByRole("button", { name: "Cancel job" }),
    ).toBeNull();
  });

  it("presents a cancellation as a decision rather than a defect", () => {
    render(
      <JobRecordView
        job={job({ status: "cancelled", finished_at: 1756162800, result: null })}
        events={[event({ message: "Cancelled at your request." })]}
      />,
    );

    // The decision is stated in the outcome section...
    expect(
      screen.getByText(/that is what cancelling means here/i),
    ).toBeVisible();
    // ...and the history records it in the same words the run recorded.
    expect(screen.getByRole("log")).toHaveTextContent(
      "Cancelled at your request.",
    );
    // No error code, no destructive framing: nothing on the page may read
    // the user's own decision as something that went wrong.
    expect(screen.queryByText(/^failed/i)).toBeNull();
    expect(screen.queryByText("gpu_stalled")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows the quote the job launched under, still on the finished record", () => {
    // The quote is frozen into the job spec at launch and never updated; a
    // finished job still says what it was predicted to cost (issue #72).
    render(<JobRecordView job={job({ quote: quote() })} events={[]} />);
    expect(
      screen.getByRole("heading", { name: "Cost and time estimate" }),
    ).toBeVisible();
    expect(screen.getByText(/3m 58s–18m 59s/)).toBeVisible();
    expect(screen.getByText(/INR 2\.74 – INR 13\.08/)).toBeVisible();
  });

  it("compares what was predicted against what happened on a finished job", () => {
    // Issue #77: a finished job shows each metric's prediction against the
    // measured figure, marked measured or derived, with a link to the
    // aggregate.
    render(
      <JobRecordView
        job={job({
          quote: quote(),
          actuals: {
            duration_s: 600,
            peak_memory_gb: 5.31,
              cost_minor: 400,
              currency: "INR",
              stages: [
                { name: "provisioning", duration_s: 5 },
                { name: "preparing", duration_s: 55 },
                { name: "training", duration_s: 480 },
                { name: "packaging", duration_s: 60 },
              ],
          },
        })}
        events={[]}
      />,
    );

    expect(
      screen.getByRole("heading", { name: "Prediction vs what happened" }),
    ).toBeVisible();
    // Duration: predicted 3m 58s–18m 59s (midpoint ~11m 29s), actual 10m.
    expect(screen.getByText("measured")).toBeVisible();
    expect(screen.getByText("derived from measured duration × frozen rate")).toBeVisible();
    expect(screen.getByText(/10m 00s/)).toBeVisible();
    // Peak memory: predicted 5.40 GB, measured 5.31 GB.
    expect(screen.getByText("5.40 GB")).toBeVisible();
    expect(screen.getByText("5.31 GB")).toBeVisible();
    // The measured stages are shown, marked as the run's own.
    expect(screen.getByText("packaging")).toBeVisible();
    // The aggregate is one link away.
    expect(
      screen.getByRole("link", { name: "Predictions vs actuals across runs" }),
    ).toHaveAttribute("href", "/calibration");
  });

  it("does not claim a comparison when the job has no actuals", () => {
    render(<JobRecordView job={job()} events={[]} />);
    expect(
      screen.queryByRole("heading", { name: "Prediction vs what happened" }),
    ).toBeNull();
  });

  it("keeps the full history of a finished job readable", () => {
    render(
      <JobRecordView
        job={job()}
        events={[
          event({ message: "queued" }),
          event({ id: 2, message: "Selecting a GPU" }),
        ]}
      />,
    );
    expect(screen.getByRole("log")).toHaveTextContent("queued");
    expect(screen.getByRole("log")).toHaveTextContent("Selecting a GPU");
  });

  it("charts training and held-out loss on the finished record", () => {
    // Issue #53: the overfitting signal is read after the run too, so the
    // chart lives on the finished record as well as the running view.
    render(
      <JobRecordView
        job={job()}
        events={[
          event(),
          event({
            id: 2,
            kind: "metric",
            message: "{'loss': 0.6931, 'step': 10}",
            data: { loss: 0.6931, step: 10 },
          }),
          event({
            id: 3,
            kind: "metric",
            message: "{'eval_loss': 0.52, 'epoch': 0.5}",
            data: { held_out_loss: 0.52, epoch: 0.5 },
          }),
        ]}
      />,
    );
    expect(
      screen.getByRole("img", { name: /training loss and held-out loss/i }),
    ).toBeVisible();
    expect(screen.getByText("Training loss")).toBeVisible();
    expect(screen.getByText("Held-out loss")).toBeVisible();
  });

  it("shows the pulls' progress on the finished record, with the detail kept", () => {
    // Issue #49: the finished record keeps the promoted progress and the raw
    // lines that became it, so a returned visitor sees how the run got there
    // and nothing was discarded.
    render(
      <JobRecordView
        job={job()}
        events={[]}
        progress={[
          {
            phase: "image pull",
            done: 42420000,
            total: 42420000,
            rate: 12_000_000,
            eta_s: 0,
            ts: 1,
          },
          {
            phase: "model download",
            done: 2_400_000_000,
            total: 4_000_000_000,
            rate: 28_000_000,
            eta_s: 57,
            ts: 2,
          },
        ]}
        output={[
          { id: 1, phase: "image pull", line: "9b829b73a52f: Pull complete" },
          {
            id: 2,
            phase: "model download",
            line: "model.safetensors:  60%|██████    | 2.4G/4.00G [00:30<00:20]",
          },
        ]}
      />,
    );

    expect(screen.getByRole("heading", { name: "Progress" })).toBeVisible();
    expect(
      screen.getByRole("progressbar", { name: "image pull" }),
    ).toHaveAttribute("aria-valuenow", "100");
    expect(screen.getByText("model download")).toBeVisible();
    expect(screen.getByText(/2\.4 GB of 4\.0 GB/)).toBeVisible();
    expect(screen.getByText(/28\.0 MB\/s/)).toBeVisible();
    expect(screen.getByText(/about 57s left/)).toBeVisible();
    // The raw lines are offered as collapsed detail, not scattered into the
    // event log: the promoted line is present but hidden behind the summary.
    expect(screen.getByText(/60%\|██████/)).not.toBeVisible();
  });

  it("surfaces a held-out loss that stopped improving, in plain language", () => {
    const heldOut = (id: number, loss: number) =>
      event({
        id,
        kind: "metric",
        message: `{'eval_loss': ${loss}, 'epoch': 1.0}`,
        data: { held_out_loss: loss, epoch: 1.0 },
      });
    render(
      <JobRecordView
        job={job()}
        events={[heldOut(2, 0.7), heldOut(3, 0.7), heldOut(4, 0.7)]}
      />,
    );
    const note = screen.getByText(/held-out loss has not improved/i);
    expect(note).toHaveTextContent(/overfitting/i);
  });

  it("states which checkpoint was chosen as the result and why, and offers the rest", () => {
    // Issue #62: the choice is recorded on the run, so the finished record
    // says which checkpoint won and why, flags it, and keeps every other
    // retained checkpoint downloadable.
    render(
      <JobRecordView
        job={job({
          best_checkpoint: {
            step: 20,
            held_out_loss: 0.39,
            basis: "best_held_out_loss",
            reason:
              "Step 20 has the lowest held-out loss (0.39) of 3 retained checkpoint(s).",
          },
          checkpoints: [
            { step: 10, slot: 0, held_out_loss: 0.44, verified: true },
            { step: 20, slot: 1, held_out_loss: 0.39, verified: true, selected: true },
            { step: 30, slot: 2, held_out_loss: 0.52, verified: true },
          ],
        })}
        events={[]}
      />,
    );

    // The choice, in plain language, with its recorded reason.
    expect(
      screen.getByText("Best checkpoint: step 20 (held-out loss 0.39)"),
    ).toBeVisible();
    expect(
      screen.getByText(
        "Step 20 has the lowest held-out loss (0.39) of 3 retained checkpoint(s).",
      ),
    ).toBeVisible();
    // The chosen one is flagged; every retained one is downloadable.
    expect(screen.getByText("chosen result")).toBeVisible();
    const downloads = screen.getAllByRole("link", { name: "Download" });
    expect(downloads).toHaveLength(3);
    for (const [i, step] of [10, 20, 30].entries()) {
      expect(downloads[i]).toHaveAttribute(
        "href",
        `/v1/jobs/job_abc123def456/checkpoints/${step}`,
      );
    }
    expect(screen.getByText("held-out loss 0.44")).toBeVisible();
    expect(screen.getByText("held-out loss 0.52")).toBeVisible();
  });

  it("names a checkpoint with no held-out loss, without a download that cannot succeed", () => {
    // A checkpoint that never landed (not retained) is named but not offered:
    // a download that cannot succeed is worse than none (issue #62's "any
    // other checkpoint remains downloadable" holds for what is retained).
    render(
      <JobRecordView
        job={job({
          checkpoints: [
            { step: 10, slot: 0, held_out_loss: 0.4, verified: true, selected: true },
            { step: 20, slot: 1, verified: false, superseded: true },
          ],
          best_checkpoint: {
            step: 10,
            held_out_loss: 0.4,
            basis: "best_held_out_loss",
            reason: "Step 10 has the lowest held-out loss (0.4) of 1 retained checkpoint(s).",
          },
        })}
        events={[]}
      />,
    );
    expect(screen.getByText("Step 10")).toBeVisible();
    expect(screen.getByText("Step 20")).toBeVisible();
    expect(screen.getByText("not retained")).toBeVisible();
    expect(screen.getByText("no held-out loss recorded")).toBeVisible();
    // Exactly one download is offered: the retained one.
    expect(screen.getAllByRole("link", { name: "Download" })).toHaveLength(1);
  });

  it("shows no checkpoint section when the run recorded none", () => {
    render(<JobRecordView job={job()} events={[]} />);
    expect(
      screen.queryByRole("heading", { name: "Checkpoints" }),
    ).toBeNull();
  });

  it("renders without scripting once the job is terminal", () => {
    const { container } = render(<JobRecordView job={job()} events={[]} />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector('meta[http-equiv="refresh"]')).toBeNull();
  });

  it("re-renders itself while the job is still working, and shows it as it is", () => {
    render(
      <JobRecordView
        job={job({ status: "training", finished_at: null, result: null })}
        events={[]}
      />,
    );
    // The no-JavaScript equivalent of the old page's poll-and-reload: a
    // returned visitor does not read a stale state. React 19 hoists the tag
    // into the document head, where a browser honours it.
    const refresh = document.querySelector('meta[http-equiv="refresh"]');
    expect(refresh).toHaveAttribute("content", "2");
    expect(screen.getByText("State").nextElementSibling?.textContent).toBe(
      "training",
    );
    // A working job offers no download yet.
    expect(
      screen.queryByRole("link", { name: /Download the artifact/ }),
    ).toBeNull();
  });

  // The advanced-surface overrides frozen into the job spec (issue #80) are
  // shown on the finished run: the run says what it actually used.
  it("shows the frozen settings the user changed on the finished run", () => {
    render(
      <JobRecordView
        job={job({
          hyperparameters: {
            learning_rate: 0.0001,
            num_epochs: 5,
          },
        })}
        events={[]}
      />,
    );
    expect(
      screen.getByRole("heading", { name: "Settings you changed" }),
    ).toBeVisible();
    // The overrides are shown as frozen at launch, with their values.
    expect(screen.getByText("learning_rate")).toBeVisible();
    expect(screen.getByText("0.0001")).toBeVisible();
    expect(screen.getByText("num_epochs")).toBeVisible();
    expect(screen.getByText("5")).toBeVisible();
  });

  it("does not show a settings section when nothing was changed", () => {
    render(<JobRecordView job={job()} events={[]} />);
    expect(
      screen.queryByRole("heading", { name: "Settings you changed" }),
    ).toBeNull();
  });

  it("discloses how many events are shown when not truncated", () => {
    render(
      <JobRecordView
        job={job()}
        events={[event({ id: 1, message: "a" }), event({ id: 2, message: "b" })]}
        total={2}
      />,
    );
    expect(screen.getByTestId("event-disclosure")).toHaveTextContent("Showing 2 of 2 events");
    // No pagination control when nothing is hidden.
    expect(screen.queryByRole("button", { name: /Load more events/ })).toBeNull();
  });

  it("where a limit still applies it says what it is showing and of how many, and the tail is reachable", async () => {
    // A large run exceeds the 500 cap; the page states the cut and offers pagination.
    const many = Array.from({ length: 500 }, (_, i) =>
      event({ id: i + 1, message: `log ${i + 1}` }),
    );
    render(<JobRecordView job={job()} events={many} total={734} />);
    const disclosure = screen.getByTestId("event-disclosure");
    expect(disclosure).toHaveTextContent("Showing 500 of 734 events");
    expect(disclosure).toHaveTextContent(/paginated/);
    expect(disclosure).toHaveTextContent(/full record remains reachable/);
    const btn = screen.getByRole("button", { name: /Load more events/ });
    expect(btn).toBeVisible();
    expect(btn).toHaveTextContent(/234 remaining/);
  });

  // The side-by-side comparison (issue #69): held-out prompts answered by the
  // base model and the chosen checkpoint, with the decoding settings recorded
  // so a reader can tell whether two outputs are comparable.
  it("shows both models answering the same held-out prompts", () => {
    render(
      <JobRecordView
        job={job({
          comparison: {
            ok: true,
            decoding: { temperature: 0.7, max_new_tokens: 128, do_sample: true },
            selection: {
              step: 20,
              basis: "best_held_out_loss",
              reason: "Step 20 has the lowest held-out loss.",
            },
            rows: [
              {
                prompt: [{ role: "user", content: "What is the capital of France?" }],
                base: "The base model's answer.",
                tuned: "The tuned model's answer.",
              },
            ],
          },
        })}
        events={[]}
      />,
    );
    const section = within(
      screen.getByRole("region", { name: "Base model vs your tuned model" }),
    );
    expect(section.getByText("What is the capital of France?")).toBeVisible();
    expect(section.getByText("The base model's answer.")).toBeVisible();
    expect(section.getByText("The tuned model's answer.")).toBeVisible();
    // The tuned side names the checkpoint the run chose, and the decoding
    // settings are shown -- recorded, so two outputs can be compared.
    expect(
      section.getByText("Tuned model (chosen checkpoint, step 20)"),
    ).toBeVisible();
    expect(section.getByText(/temperature 0.7/)).toBeVisible();
    expect(section.getByText(/up to 128 new tokens/)).toBeVisible();
  });

  it("states a failed comparison's reason without dressing it as a failed job", () => {
    render(
      <JobRecordView
        job={job({
          comparison: {
            ok: false,
            decoding: { temperature: 0.7 },
            reason: "RuntimeError: the tuned model could not be loaded",
          },
        })}
        events={[]}
      />,
    );
    expect(
      screen.getByRole("heading", { name: "Base model vs your tuned model" }),
    ).toBeVisible();
    expect(
      screen.getByText(/No side-by-side comparison was produced/),
    ).toBeVisible();
    expect(screen.getByText(/could not be loaded/)).toBeVisible();
    // The job itself still reads complete -- a comparison failure is not a
    // failed run.
    expect(
      screen.getByText("State").nextElementSibling?.textContent,
    ).toBe("complete");
  });

  it("shows nothing when the run recorded no comparison", () => {
    render(<JobRecordView job={job()} events={[]} />);
    expect(
      screen.queryByRole("heading", { name: "Base model vs your tuned model" }),
    ).toBeNull();
  });
});

describe("the general-capability slice (issue #73)", () => {
  const capability = (over: Record<string, unknown> = {}) => ({
    ok: true,
    version: 1,
    regression_threshold: 2,
    total: 8,
    base_correct: 6,
    tuned_correct: 4,
    base_score: 0.75,
    tuned_score: 0.5,
    delta: -0.25,
    delta_se: 0.152,
    large_regression: true,
    decoding: { temperature: 0.7, max_new_tokens: 128, do_sample: true },
    selection: {
      step: 20,
      basis: "best_held_out_loss",
      reason: "Step 20 has the lowest held-out loss.",
    },
    rows: [
      {
        prompt: [{ role: "user", content: "Which planet is the largest?" }],
        domain: "astronomy",
        answer: "A",
        base: "A",
        tuned: "C",
        base_parsed: "A",
        tuned_parsed: "C",
        base_correct: true,
        tuned_correct: false,
      },
    ],
    ...over,
  });

  it("labels it a smoke test, shows the sample size beside the numbers, the change and the uncertainty", () => {
    render(
      <JobRecordView
        job={job({ capability: capability() as JobRecord["capability"] })}
        events={[]}
      />,
    );
    const section = within(
      screen.getByRole("region", { name: "General capability" }),
    );
    // The interface says it is a smoke test, never a benchmark.
    expect(
      section.getByText(/smoke test for catastrophic forgetting, not a benchmark/i),
    ).toBeVisible();
    // The sample size sits beside each side's number, and the change and its
    // uncertainty are stated.
    expect(section.getByText("6 of 8 (75%)")).toBeVisible();
    expect(section.getByText("4 of 8 (50%)")).toBeVisible();
    expect(section.getByText("−2 of 8 (−25%)")).toBeVisible();
    expect(
      section.getByText(/each is worth 12.5% of the score/i),
    ).toBeVisible();
    expect(section.getByText(/standard error of the change is ±1.2 questions/i)).toBeVisible();
    // The tuned side names the checkpoint the run chose, and the decoding
    // settings are shown -- recorded, so the numbers are legible.
    expect(
      section.getByText("Tuned model (chosen checkpoint, step 20)"),
    ).toBeVisible();
    expect(section.getByText(/temperature 0.7/)).toBeVisible();
  });

  it("surfaces a large regression prominently, from the recorded flag", () => {
    render(
      <JobRecordView
        job={job({ capability: capability() as JobRecord["capability"] })}
        events={[]}
      />,
    );
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Large regression");
    // The threshold is read from the record, never re-decided here.
    expect(alert).toHaveTextContent("at least 2 fewer");
    expect(
      screen.getByText(/fine-tuning degraded general capability/i),
    ).toBeVisible();
  });

  it("does not claim a regression the record does not flag", () => {
    render(
      <JobRecordView
        job={job({
          capability: capability({ large_regression: false }) as JobRecord["capability"],
        })}
        events={[]}
      />,
    );
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("states a failed slice's reason without dressing it as a failed job", () => {
    render(
      <JobRecordView
        job={job({
          capability: {
            ok: false,
            version: 1,
            decoding: { temperature: 0.7 },
            reason: "RuntimeError: the tuned model could not be loaded",
          } as JobRecord["capability"],
        })}
        events={[]}
      />,
    );
    expect(
      screen.getByRole("heading", { name: "General capability" }),
    ).toBeVisible();
    expect(
      screen.getByText(/No general-capability check was produced/),
    ).toBeVisible();
    expect(screen.getByText(/could not be loaded/)).toBeVisible();
    // The job itself still reads complete -- a failed slice is not a failed
    // run.
    expect(
      screen.getByText("State").nextElementSibling?.textContent,
    ).toBe("complete");
  });

  it("shows nothing when the run recorded no capability", () => {
    render(<JobRecordView job={job()} events={[]} />);
    expect(
      screen.queryByRole("heading", { name: "General capability" }),
    ).toBeNull();
  });
});
