# Decision records

Every decision that shapes this platform, with the alternatives that were
rejected and why. Each entry is written as its change lands, not afterwards. An
entry written during the build records reasoning. An entry written after it
reconstructs reasoning, and reconstruction is what fails under questioning.

## Why the numbers are not in date order

This directory was created on 2026-08-21, about three-quarters of the way into
the build, and the in-repo numbering began there. The earlier decisions —
the choice of JarvisLabs VMs with an SSH-driven bootstrap, pinning the
trainer image by digest, detecting thinking mode from the dataset, the
reference baseline, Temporal for orchestration, and the rest — were made
between 2026-08-15 and 2026-08-21 and filed here afterwards.

Those ten were filed in this directory on 2026-08-26
([#26](https://github.com/thp728/temper/issues/26)), in their original order,
as [0013](0013-assumptions-verified-before-code.md) through
[0022](0022-phase-a-closed-from-a-browser.md). Numbers are assigned when a
record lands, never reserved, never reused, never renumbered, so records that
were decided *earlier* carry *higher* numbers than records that landed before
them. The number says when a record entered this directory. The Date column
says when the decision was made. The index below is ordered by date so the
record reads from its beginning.

Each filed entry keeps its original reasoning, with file paths updated to
where files now live. None was rewritten, because an entry that reconstructs
its reasoning after the fact is the thing this project has consistently said
fails under questioning.

## Index

| ADR | Decision | Status | Date |
| --- | --- | --- | --- |
| [0001](0001-event-channel-over-ssh-stdout.md) | The event channel is the machine's stdout, pulled over SSH | accepted | 2026-08-21 |
| [0002](0002-stall-detection-and-a-duration-ceiling.md) | A stalled job and an over-long job are stopped separately, and named separately | accepted | 2026-08-21 |
| [0003](0003-cancellation-is-destructive.md) | Cancelling destroys the machine, produces no adapter, and is not a failure | accepted, flagged for reopening (spike 8 found the provider can pause) | 2026-08-21 |
| [0004](0004-the-machine-is-a-pure-compute-node.md) | The machine is a pure compute node, and everything it receives is pushed by the control plane | accepted | 2026-08-21 |
| [0005](0005-the-dataset-size-limit-is-derived-from-measured-memory.md) | The dataset size limit is derived from measured memory, not chosen | accepted, superseded by [0036](0036-the-dataset-size-limit-is-derived-from-measured-throughput.md) | 2026-08-21 |
| [0006](0006-validation-runs-off-the-event-loop.md) | Validation runs off the event loop | accepted | 2026-08-21 |
| [0007](0007-the-feasibility-warning-is-an-estimate-and-warns-rather-than-blocks.md) | The feasibility warning is an estimate from one measured run, and warns rather than blocks | accepted | 2026-08-21 |
| [0008](0008-adapters-ship-as-fp32.md) | Adapters ship as fp32 — the artifact is exactly the weights that were trained | accepted | 2026-08-21 |
| [0009](0009-the-machine-may-write-its-own-artifact-to-a-scoped-url.md) | The machine may write its own artifact to a pre-signed URL scoped to one key | accepted, supersedes [0004](0004-the-machine-is-a-pure-compute-node.md) in part | 2026-08-23 |
| [0010](0010-the-repository-is-laid-out-as-apps-and-packages.md) | The repository is laid out as `apps/` and `packages/` | accepted | 2026-08-25 |
| [0011](0011-one-command-runs-every-task-and-one-defines-green.md) | One command runs every task, and one command defines green | accepted | 2026-08-25 |
| [0012](0012-the-repository-ships-under-mit.md) | The repository ships under MIT | accepted | 2026-08-25 |
| [0013](0013-assumptions-verified-before-code.md) | The architecture's own assumptions are verified before writing code | accepted | 2026-08-17 |
| [0014](0014-training-runs-on-jarvislabs-vms-over-ssh.md) | Training runs on JarvisLabs VMs with an SSH-driven bootstrap | accepted | 2026-08-17 |
| [0015](0015-trainer-image-pins-axolotl-by-digest.md) | The trainer image pins Axolotl by digest rather than resolving the stack | accepted | 2026-08-18 |
| [0016](0016-together-ai-is-the-reference-baseline.md) | Together AI is the reference baseline | accepted | 2026-08-18 |
| [0017](0017-the-architecture-cut-list.md) | The architecture cut list is ratified as drafted | accepted | 2026-08-18 |
| [0018](0018-thinking-mode-is-detected-from-the-dataset.md) | Thinking mode is detected from the dataset, not fixed or exposed | accepted | 2026-08-18 |
| [0019](0019-production-stack-after-the-checkpoint.md) | The product ships production-ready; scrappy only until the checkpoint | accepted | 2026-08-19 |
| [0020](0020-temporal-for-orchestration.md) | Temporal for orchestration; multi-GPU is a provisioning choice | accepted | 2026-08-19 |
| [0021](0021-the-first-assembled-run-read-before-paid.md) | The first assembled run: it works, and three defects that reading caught before the GPU did | accepted | 2026-08-19 |
| [0022](0022-phase-a-closed-from-a-browser.md) | Phase A closes from a browser, and the test double's blind spot is recorded | accepted | 2026-08-21 |
| [0023](0023-the-interface-consumes-a-client-generated-from-the-api-contract.md) | The interface consumes a client generated from the API contract, and the two interfaces never coexist | accepted | 2026-08-26 |
| [0024](0024-the-browser-journeys-launch-against-an-app-side-fake-provider.md) | The browser journeys launch against an app-side fake provider, and refuse to run against anything else | accepted | 2026-08-26 |
| [0025](0025-the-control-plane-resolves-and-the-trainer-applies.md) | The control plane resolves every hyperparameter; the trainer applies values rather than choosing them | accepted | 2026-08-26 |
| [0026](0026-the-simulated-machine-fails-when-a-journey-asks.md) | The simulated machine fails when a journey asks it to, through the real failure path | accepted | 2026-08-26 |
| [0027](0027-the-transport-is-proven-against-a-real-endpoint.md) | The transport is proven against a real connection endpoint | accepted | 2026-08-26 |
| [0028](0028-peak-memory-is-predicted-through-a-model-facts-seam.md) | Peak memory is predicted through a model-facts seam, not stored per catalog entry | accepted | 2026-08-27 |
| [0029](0029-hardware-is-selected-cheapest-fit-first-not-preferred-in-order.md) | Hardware is selected cheapest-fit-first, not preferred in order | accepted | 2026-08-27 |
| [0030](0030-disk-is-computed-not-a-platform-minimum-constant.md) | Disk is computed, not a platform-minimum constant | accepted | 2026-08-27 |
| [0031](0031-the-predictor-warns-on-time-and-cost-as-a-range-composed-per-phase.md) | The predictor warns on time and cost, as a range, composed per phase | accepted | 2026-08-27 |
| [0032](0032-every-calculated-default-carries-its-reason.md) | Every calculated default carries its reason and the alternatives that lost | accepted | 2026-08-27 |
| [0033](0033-any-plan-decision-can-be-overridden-and-the-rest-recomputes.md) | Any plan decision can be overridden, and the rest recomputes | accepted | 2026-08-27 |
| [0034](0034-the-advanced-surface-is-generated-from-the-trainers-schema.md) | The advanced surface is generated from the trainer's own schema | accepted | 2026-08-27 |
| [0035](0035-the-control-plane-verifies-a-machine-written-artifact-by-streaming-it-back.md) | The control plane verifies a machine-written artifact by streaming it back | accepted | 2026-08-27 |
| [0036](0036-the-dataset-size-limit-is-derived-from-measured-throughput.md) | The dataset size limit is a product limit derived from measured throughput, not from memory — supersedes 0005 | accepted | 2026-08-27 |
| [0037](0037-every-run-records-what-was-predicted-against-what-happened.md) | Every run records what was predicted against what happened | accepted | 2026-08-27 |
| [0038](0038-the-token-count-is-produced-by-a-phase-of-its-own-bounded-and-recorded-with-the-dataset-version.md) | The token count is produced by a phase of its own, bounded, and recorded with the dataset version | accepted | 2026-08-27 |
| [0039](0039-the-running-job-surface-consumes-a-server-pushed-event-stream.md) | The running-job surface consumes a server-pushed event stream and hands back to the finished record at a terminal state | accepted | 2026-08-27 |
| [0040](0040-an-override-names-its-failure-mode-and-a-locked-setting-is-not-a-refused-input.md) | An override names its failure mode, is recorded in the job spec, and a locked setting is not a refused input | accepted | 2026-08-27 |
| [0041](0041-checkpoints-are-written-off-the-machine-as-they-are-produced.md) | Checkpoints are written off the machine as they are produced, and verified before they are presented | accepted | 2026-08-27 |
| [0042](0042-every-export-asserts-the-training-and-artifact-templates-tokenize-identically.md) | Every export asserts the training and artifact templates tokenize identically | accepted | 2026-08-27 |
| [0043](0043-memory-blocks-at-creation-while-time-and-cost-warn.md) | Memory blocks at job creation while time and cost warn, and the two halves differ on the cost of being wrong | accepted | 2026-08-28 |
| [0044](0044-the-server-rendered-pages-are-deleted-and-the-interfaces-never-coexist.md) | The server-rendered pages are deleted, and the two interfaces never coexist | accepted | 2026-08-28 |
| [0045](0045-the-artifact-is-the-deliverable-and-the-adapter-is-one-kind-of-it.md) | The artifact is the deliverable, and the adapter is one kind of it | accepted | 2026-08-28 |
| [0046](0046-the-trainer-image-is-built-by-the-pipeline-and-referenced-by-digest.md) | The trainer image is built by the pipeline and referenced by digest, never built on the machine | accepted | 2026-08-27 |
| [0047](0047-imported-datasets-are-the-same-bytes-through-the-same-validation-path.md) | Imported datasets are the same bytes, through the same validation path as an upload | accepted | 2026-08-28 |
| [0048](0048-the-held-out-split-is-the-platforms-and-rides-the-existing-event-stream.md) | The held-out split is the platform's, and held-out loss rides the existing event stream | accepted | 2026-08-28 |
| [0049](0049-the-best-checkpoint-is-chosen-by-held-out-loss-and-the-choice-is-recorded.md) | The best checkpoint is chosen by held-out loss, and the choice is recorded on the run | accepted | 2026-08-28 |
| [0050](0050-a-model-outside-the-catalog-is-usable-once-a-probe-reports-on-it.md) | A model outside the catalog is usable once a probe reports on it, and the probe reads the predictor's own facts | accepted | 2026-08-28 |
| [0051](0051-a-failure-path-is-not-done-until-it-has-been-caused-deliberately.md) | A failure path is not done until it has been caused deliberately | accepted | 2026-08-28 |
| [0052](0052-full-fine-tuning-is-a-second-method-through-the-existing-seams.md) | Full fine-tuning is a second method through the existing seams | accepted | 2026-08-28 |
| [0053](0053-progress-is-measured-from-the-output-that-was-going-to-be-thrown-away.md) | Progress is measured from the output that was going to be thrown away | accepted | 2026-08-28 |
| [0054](0054-every-artifact-ships-with-a-generated-provenance-manifest.md) | Every artifact ships with a generated provenance manifest | accepted | 2026-08-28 |
| [0055](0055-divergence-is-detected-on-the-streamed-loss-and-offers-a-single-retry.md) | Divergence is detected on the streamed loss and offers a single retry as a choice | accepted | 2026-08-28 |
| [0056](0056-a-mixture-of-experts-model-is-usable-and-labelled-untested.md) | A mixture-of-experts model is usable and labelled untested | accepted | 2026-08-28 |
| [0057](0057-teardown-is-confirmed-across-consecutive-observations.md) | Teardown is confirmed across consecutive observations | accepted | 2026-08-28 |
| [0058](0058-the-finished-job-s-event-history-is-paginated-and-disclosed.md) | The finished job's event history is paginated and disclosed | accepted | 2026-08-28 |
| [0059](0059-delivery-formats-flow-through-the-artifact-manifest.md) | Delivery formats are produced off the correctly merged model and flow through the artifact manifest | accepted | 2026-08-28 |
| [0060](0060-out-of-memory-retries-with-the-effective-batch-preserved.md) | An out-of-memory failure retries automatically with the effective batch preserved | accepted | 2026-08-28 |
| [0061](0061-the-side-by-side-comparison-runs-on-the-warm-machine.md) | The side-by-side comparison runs on the warm machine, compares the chosen checkpoint, and never fails the run | accepted | 2026-08-28 |
| [0062](0062-the-configuration-boundary-sits-at-deployment-settings.md) | The configuration boundary sits at deployment settings, and the stack starts with one command | accepted | 2026-08-28 |
| [0063](0063-a-spend-ceiling-is-enforced-outside-the-training-process.md) | A spend ceiling is enforced outside the training process, with the shutdown ordered checkpoint-then-terminate-then-destroy | accepted | 2026-08-28 |
| [0064](0064-the-relational-store-moves-behind-the-existing-seam.md) | The relational store moves behind the existing persistence seam, and the migration's boundary held for the domain but not for the test harness's isolation strategy or the engine's operational shape | accepted | 2026-08-29 |
| [0065](0065-a-served-endpoint-stops-itself.md) | A served endpoint stops itself | accepted | 2026-08-29 |
| [0066](0066-orchestration-moves-into-a-worker-process-that-claims-work.md) | Orchestration moves into a worker process that claims queued jobs with `SELECT ... FOR UPDATE SKIP LOCKED`; the request path starts no threads | accepted | 2026-08-29 |
| [0067](0067-the-general-capability-check-is-a-small-slice-whose-limits-are-stated.md) | The general-capability check is a small slice, labelled a smoke test, whose limits are stated | accepted | 2026-08-30 |
| [0068](0068-the-reconciler-destroys-machines-no-live-job-owns.md) | The reconciler destroys machines no live job owns | accepted | 2026-08-30 |
| [0069](0069-an-interrupted-job-resumes-from-its-last-checkpoint-on-a-fresh-machine.md) | An interrupted job resumes from its last checkpoint on a fresh machine | accepted | 2026-08-30 |
| [0070](0070-events-are-persisted-before-published-and-replay-by-last-seen.md) | Events are persisted before they are published, and watchers replay by last-seen identifier | accepted | 2026-08-30 |
| [0071](0071-the-zero-cost-path-is-a-labelled-demonstration-structurally-blind-to-the-transport.md) | The zero-cost path is a labelled demonstration, seeded and structurally blind to the transport — a single setting selects it, every page is marked, and the limitation is named | accepted | 2026-08-30 |
| [0072](0072-durable-execution-recovers-the-job-while-the-reconciler-protects-the-money.md) | Durable execution recovers the job while the reconciler protects the money — a claim lease, a step cursor, and recovery that never stacks machines | accepted | 2026-08-31 |
| [0073](0073-a-bulk-import-reads-parquet-not-the-preview-endpoint.md) | A bulk import reads Parquet, not the preview endpoint — `/rows` is generated on demand and throttled, the shards sit in the published Resolvers bucket, and `/rows` stays as the fallback for an unconverted split | accepted | 2026-09-03 |
| [0074](0074-dataset-ingestion-recognises-exactly-one-schema.md) | Dataset ingestion recognises exactly one schema — the OpenAI-style chat `messages` list — on purpose, not by accident | accepted | 2026-09-04 |
| [0075](0075-hyperparameters-edit-in-place-and-explanations-open-on-demand.md) | Hyperparameters edit in place, and explanations open on demand — cards own the only editors, reasons and failure modes sit behind "?", sliders/selects rejected for lack of published bounds | accepted | 2026-09-05 |
| [0076](0076-the-served-endpoint-loads-the-model-before-it-mints-a-key.md) | The served endpoint loads the model before it mints a key — the server is pushed rather than baked, reached over SSH rather than published, and a machine whose model never loads is destroyed with the start refused | accepted | 2026-09-06 |

Spec 003 refers to the fp32 decision as "ADR-0003", its position in the
original decision ordering; in this directory it is [0008](0008-adapters-ship-as-fp32.md).

## Numbering

Numbers are assigned when a record lands, not reserved in advance. Specs used
to name the number their decisions would take. They now name them by title,
because a reservation is a plan, and a plan that slips leaves gaps in a
sequence that is supposed to mean chronology. Numbers are never reused and
never renumbered. A number therefore says when a record entered this directory,
while the index's Date column says when the decision was made. The two differ
for the entries filed later, which is why the index above is
ordered by date rather than by number.

## Format

Each record carries the context that forced the decision, the decision itself,
the alternatives considered and why each was rejected, the consequences
accepted, and how to roll it back. The alternatives section is the point of the
document. A decision recorded without its rejected options is an assertion
rather than a record.

A decision is not edited once accepted. Changing one means writing a new record
that supersedes it, so that the reasoning at the time stays legible.
