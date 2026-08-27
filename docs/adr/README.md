# Decision records

Every decision that shapes this platform, with the alternatives that were
rejected and why. Written **as each change lands, not afterwards** — an entry
written during the build records reasoning, an entry written after it
reconstructs reasoning, and reconstruction is what fails under questioning.

## Why the numbers are not in date order

This directory was created on 2026-08-21, about three-quarters of the way into
the build, and the in-repo numbering began there. The **earlier decisions were
logged in a private working vault** as they were made, between 2026-08-15
and 2026-08-21: the choice of JarvisLabs VMs with an SSH-driven bootstrap,
pinning the trainer image by digest, detecting thinking mode from the dataset,
the reference baseline, Temporal for orchestration, and the rest. That log is
closed to new entries.

**The ten platform decisions among them were copied into this directory on
2026-08-26** ([#26](https://github.com/thp728/temper/issues/26)), in their
original order, as [0013](0013-assumptions-verified-before-code.md) through
[0022](0022-phase-a-closed-from-a-browser.md). Three further vault entries are
personal project-management records — scheduling, research budgeting and the
choice of product name — and stay in the private vault. Because numbers are
assigned when a record lands — never reserved, never reused, never renumbered —
records that were decided *earlier* carry *higher* numbers than records that
landed before them. **The number says when a record entered this directory; the
Date column says when the decision was made.** The index below is ordered by
date so the record reads from its beginning.

Each copied entry keeps its original reasoning; it was edited only for a public
audience (private-vault links replaced with descriptions of what they pointed
at, file paths updated to where files now live), never rewritten — an entry that
reconstructs its reasoning after the fact is the thing this project has
consistently said fails under questioning.

Records are written for a public audience from the first entry, because the
repository becomes public at submission.

## Index

| ADR | Decision | Status | Date |
| --- | --- | --- | --- |
| [0013](0013-assumptions-verified-before-code.md) | The architecture's own assumptions are verified before writing code | accepted | 2026-08-17 |
| [0014](0014-training-runs-on-jarvislabs-vms-over-ssh.md) | Training runs on JarvisLabs VMs with an SSH-driven bootstrap | accepted | 2026-08-17 |
| [0015](0015-trainer-image-pins-axolotl-by-digest.md) | The trainer image pins Axolotl by digest rather than resolving the stack | accepted | 2026-08-18 |
| [0016](0016-together-ai-is-the-reference-baseline.md) | Together AI is the reference baseline | accepted | 2026-08-18 |
| [0017](0017-the-architecture-cut-list.md) | The architecture cut list is ratified as drafted | accepted | 2026-08-18 |
| [0018](0018-thinking-mode-is-detected-from-the-dataset.md) | Thinking mode is detected from the dataset, not fixed or exposed | accepted | 2026-08-18 |
| [0019](0019-production-stack-after-the-checkpoint.md) | The product ships production-ready; scrappy only until the checkpoint | accepted | 2026-08-19 |
| [0020](0020-temporal-for-orchestration.md) | Temporal for orchestration; multi-GPU is a provisioning choice | accepted | 2026-08-19 |
| [0021](0021-the-first-assembled-run-read-before-paid.md) | The first assembled run: it works, and three defects that reading caught before the GPU did | accepted | 2026-08-19 |
| [0001](0001-event-channel-over-ssh-stdout.md) | The event channel is the machine's stdout, pulled over SSH | accepted | 2026-08-21 |
| [0002](0002-stall-detection-and-a-duration-ceiling.md) | A stalled job and an over-long job are stopped separately, and named separately | accepted | 2026-08-21 |
| [0003](0003-cancellation-is-destructive.md) | Cancelling destroys the machine, produces no adapter, and is not a failure | accepted — **flagged for reopening** (spike 8 found the provider can pause) | 2026-08-21 |
| [0004](0004-the-machine-is-a-pure-compute-node.md) | The machine is a pure compute node, and everything it receives is pushed by the control plane | accepted | 2026-08-21 |
| [0005](0005-the-dataset-size-limit-is-derived-from-measured-memory.md) | The dataset size limit is derived from measured memory, not chosen | accepted — **corrected** (spike 9 measured 5.93×, not 4.8×) | 2026-08-21 |
| [0006](0006-validation-runs-off-the-event-loop.md) | Validation runs off the event loop | accepted | 2026-08-21 |
| [0007](0007-the-feasibility-warning-is-an-estimate-and-warns-rather-than-blocks.md) | The feasibility warning is an estimate from one measured run, and warns rather than blocks | accepted | 2026-08-21 |
| [0008](0008-adapters-ship-as-fp32.md) | Adapters ship as fp32 — the artifact is exactly the weights that were trained | accepted | 2026-08-21 |
| [0022](0022-phase-a-closed-from-a-browser.md) | Phase A closes from a browser, and the test double's blind spot is recorded | accepted | 2026-08-21 |
| [0009](0009-the-machine-may-write-its-own-artifact-to-a-scoped-url.md) | The machine may write its own artifact to a pre-signed URL scoped to one key | accepted — supersedes [0004](0004-the-machine-is-a-pure-compute-node.md) in part | 2026-08-23 |

| [0010](0010-the-repository-is-laid-out-as-apps-and-packages.md) | The repository is laid out as `apps/` and `packages/` | accepted | 2026-08-25 |
| [0011](0011-one-command-runs-every-task-and-one-defines-green.md) | One command runs every task, and one command defines green | accepted | 2026-08-25 |
| [0012](0012-the-repository-ships-under-mit.md) | The repository ships under MIT | accepted | 2026-08-25 |
| [0027](0027-the-transport-is-proven-against-a-real-endpoint.md) | The transport is proven against a real connection endpoint | accepted | 2026-08-26 |
| [0023](0023-the-interface-consumes-a-client-generated-from-the-api-contract.md) | The interface consumes a client generated from the API contract, and the two interfaces never coexist | accepted | 2026-08-26 |
| [0024](0024-the-browser-journeys-launch-against-an-app-side-fake-provider.md) | The browser journeys launch against an app-side fake provider, and refuse to run against anything else | accepted | 2026-08-26 |
| [0025](0025-the-control-plane-resolves-and-the-trainer-applies.md) | The control plane resolves every hyperparameter; the trainer applies values rather than choosing them | accepted | 2026-08-26 |
| [0026](0026-the-simulated-machine-fails-when-a-journey-asks.md) | The simulated machine fails when a journey asks it to, through the real failure path | accepted | 2026-08-26 |
| [0028](0028-peak-memory-is-predicted-through-a-model-facts-seam.md) | Peak memory is predicted through a model-facts seam, not stored per catalog entry | accepted | 2026-08-27 |
| [0029](0029-hardware-is-selected-cheapest-fit-first-not-preferred-in-order.md) | Hardware is selected cheapest-fit-first, not preferred in order | accepted | 2026-08-27 |
| [0030](0030-disk-is-computed-not-a-platform-minimum-constant.md) | Disk is computed, not a platform-minimum constant | accepted | 2026-08-27 |
| [0031](0031-the-predictor-warns-on-time-and-cost-as-a-range-composed-per-phase.md) | The predictor warns on time and cost, as a range, composed per phase | accepted | 2026-08-27 |
| [0032](0032-every-calculated-default-carries-its-reason.md) | Every calculated default carries its reason and the alternatives that lost | accepted | 2026-08-27 |
| [0033](0033-any-plan-decision-can-be-overridden-and-the-rest-recomputes.md) | Any plan decision can be overridden, and the rest recomputes | accepted | 2026-08-27 |
| [0034](0034-the-advanced-surface-is-generated-from-the-trainers-schema.md) | The advanced surface is generated from the trainer's own schema | accepted | 2026-08-27 |
| [0035](0035-the-control-plane-verifies-a-machine-written-artifact-by-streaming-it-back.md) | The control plane verifies a machine-written artifact by streaming it back | accepted | 2026-08-27 |

Spec 003 refers to the fp32 decision as "ADR-0003", its position in the
private working log; in this directory it is [0008](0008-adapters-ship-as-fp32.md).

## Numbering

Numbers are assigned **when a record lands**, not reserved in advance. Specs
used to name the number their decisions would take; they now name them by
title, because a reservation is a plan and a plan that slips leaves gaps in a
sequence that is supposed to mean chronology. Numbers are never reused and
never renumbered. A number therefore says when a record entered this directory;
the index's Date column says when the decision was made — the two differ for the
entries copied in from the vault, which is why the index above is
ordered by date rather than by number.

## Format

Each record carries: the context that forced the decision, the decision itself,
**the alternatives considered and why each was rejected**, the consequences
accepted, and how to roll it back. The alternatives section is the point of the
document — a decision recorded without its rejected options is an assertion, not
a record.

A decision is not edited once accepted. Changing one means writing a new record
that supersedes it, so that the reasoning at the time stays legible.
