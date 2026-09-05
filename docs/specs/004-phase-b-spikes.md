# Spec 004 — Phase B spikes

**Status:** run 2026-08-23 — see Outcomes
**Phase:** B (Sun 2026-08-23, before any Phase B ticket is written)
**Depends on:** nothing — these run against the live account and the pinned image
**Produces:** the measured inputs for Specs 005+ and ADR-0009 (superseding ADR-0004)
**Assumes:** ADR-0004's transport rules still hold at Phase A scale, and stop holding above it

## Problem Statement

Phase B's scope changed shape on 2026-08-23. The catalog stops being two dense
4B/8B models and becomes any model the provider can hold; full fine-tuning and
multi-GPU move from cut to in; the hyperparameter surface stops being eleven
hand-picked knobs and becomes whatever the pinned Axolotl accepts; and the
product's central claim moves from *"it trains"* to *"it computes the right
configuration for what you asked for, and tells you why."*

**Every one of those rests on numbers nobody has measured.** The largest job
ever run through this product is a 4B QLoRA on one L4 producing a 132 MB
adapter. The plan now describes 70B jobs producing artifacts a thousand times
that size, on hardware configurations never provisioned, through a transport
that holds payloads in memory as `bytes`.

Phase A's lesson was that every defect was found by running the assembled
thing, never by the suite. The corollary for Phase B is narrower and cheaper:
**find the physical limits before the tickets are written, not after the code
assumes them.** A spike that costs ₹50 on 2026-08-23 is worth a day on
2026-08-29.

These are probes, not prototypes. None of them trains anything it does not
have to. A spike that also trains cannot tell you which half broke.

## What each spike gates

| # | Question | Blocks | GPU | Est. cost |
| --- | --- | --- | --- | --- |
| 5 | How much disk can a VM have, and how fast do weights arrive? | 70B entirely; the predictor's disk output; the quote's ETA | yes | ~₹50 |
| 6 | Does the pinned image see 2 GPUs, and does Axolotl's FSDP run? | the 8B full-FT capstone; multi-GPU as a product claim | yes | ~₹15 |
| 7 | How wide is Axolotl's config schema, and can we derive from it? | Advanced mode's entire design | no | ₹0 |
| 8 | What does the JarvisLabs SDK actually expose? | whether "reflect the provider's features" is a claim or a slogan | no | ₹0 |
| 9 | How fast does validation stream a multi-GB dataset? | the 1 GB upload ceiling; token counting; the quote | no | ₹0 |

**Run 5 and 6 first.** They are the two that can kill a capstone, and finding
that out at ₹41/hr on a Sunday is the entire point. 7, 8 and 9 touch no money
and can run in parallel.

---

## Spike 5 — Disk ceiling and model download throughput

### The question

Two numbers, both currently unknown, both load-bearing.

**Disk.** `create()` enforces a 100 GB minimum on VMs — measured in Spike 1 and
recorded as a floor. Nothing has ever asked for more. Llama-3.3-70B in bf16 is
~141 GB of weights, which does not fit inside the floor, and QLoRA does not
help: the bf16 weights still land on disk before they are quantised on load,
unless a pre-quantised repo is pinned instead — which changes the provenance
story and is a separate decision. Add checkpoints and a merged output and a
70B job needs somewhere between 250 and 400 GB.

**Throughput.** The measured cold start is 2–4 minutes, of which 87–183s is a
trainer image of a few GB. The only model ever downloaded through this product
is ~8 GB. A 70B model is ~140 GB from Hugging Face onto a JarvisLabs VM at a
throughput nobody has measured. At 200 MB/s that is 12 minutes; at 50 MB/s it
is 47. **That difference is the entire ETA of the `preparing` phase**, which is
a number the quote promises to the user before they spend anything.

### Method

1. Read the SDK's create signature for the storage parameter and its bounds.
   Record whether the ceiling is documented, enforced, or discovered by
   rejection.
2. Provision the cheapest VM-capable GPU with storage well above the floor —
   400 GB if it is accepted, otherwise bisect to find what is.
3. Confirm the disk is actually present and writable at that size from inside
   the machine, not merely accepted by the API. A create call's return value is
   not evidence.
4. Record the hourly rate with and without the extra storage, from
   `account.currency()` and the pricing surface — **storage may be billed
   separately from the card, and the quote has to say so.**
5. `hf download` a large repo — the largest that is not itself an hour of
   waiting — and record sustained MB/s, wall clock, and whether throughput is
   steady or bursty. Bursty matters: a live ETA computed from a 30-second
   window will oscillate badly if it is.
6. Destroy in a `finally` block. Confirm by listing instances across
   **consecutive** samples, treating `Destroying` as not-yet-confirmed —
   correction C17 says a single absent listing is not proof.

### Kill criterion

**If disk cannot exceed ~200 GB, 70B full fine-tuning is off the table on this
provider and the spec says so in those words.** The catalog stays unbounded by
design, the predictor still computes the configuration honestly, and the README
states the largest configuration actually exercised — the same pattern already
used for multi-GPU (*"measured but unexercised — never describe it as
proven"*). That is a finding, not a failure, and it is worth ₹50 to have it on
the 23rd rather than the 29th.

### What it writes

`spike/findings-spike5.json` — storage ceiling, accepted values, rate with and
without extra storage, currency, download MB/s (mean and variance), wall clock
per GB, and the teardown confirmation trace.

### What consumes it

The predictor gains `disk_gb` as an output and a per-GB cost term. The quote's
`preparing` ETA becomes a range derived from a measured rate rather than an
extrapolation from an 8 GB model. The 70B capstone is either scheduled or
formally cut.

---

## Spike 6 — Multi-GPU device visibility and FSDP through the pinned image

### The question

`num_gpus` is a create parameter and every VM-capable type shows 8 free devices
— both measured 2026-08-19, and both are **availability facts, not execution
facts.** Nothing has ever run on more than one card through this stack. Three
things in sequence have to be true and only the first has any evidence:

1. `num_gpus=2` provisions a machine with two devices attached.
2. `docker run --gpus all` inside the **pinned image** yields
   `torch.cuda.device_count() == 2`. The image was built and digest-pinned
   before multi-GPU was in scope; nothing asserts the toolkit passes more than
   one device through.
3. Axolotl's FSDP FULL_SHARD configuration launches and takes steps under
   `accelerate` in that image, and writes a checkpoint in a format that can be
   read back.

Point 3 is where this goes wrong, and it is worth being explicit about why:
sharded checkpoints are not the checkpoint format this product has ever
written. Resume is a proven flow at single-GPU and an unproven one here.

### Method

1. Provision **2× of the cheapest VM-capable card** — 2× L4 at ~₹82/hr. Not 8,
   and not H100s. The question is whether the mechanism works at all, and the
   cheapest configuration that can answer it is the right one.
2. Pull the pinned digest. Assert `device_count()` inside the container.
3. Run Axolotl with an FSDP FULL_SHARD config for **three steps** on the
   smallest model in the catalog. Not a training run — a mechanism check.
4. Write a checkpoint, stop, resume from it, take one more step. If sharded
   resume does not work, that is the finding.
5. Record peak VRAM per device, so the predictor's sharding arithmetic has one
   real anchor instead of zero.
6. Destroy in `finally`; confirm across consecutive samples.

### Kill criterion

**If FSDP does not run in the pinned image, the 8B full-FT capstone does not go
on the calendar** and multi-GPU stays exactly what the docs already call it:
designed and unexercised. Do not attempt to fix the image on the 27th at
₹510/hr — that is the mistake this spike exists to prevent.

### What it writes

`spike/findings-spike6.json` — device count inside the container, FSDP launch
outcome, steps completed, checkpoint format written, sharded resume outcome,
per-device peak VRAM, and the teardown trace.

### What consumes it

The 8B full-FT-on-2×-H100 capstone, scheduled for 2026-08-27. The predictor's
multi-GPU branch. The honesty of every sentence in the README about eight cards
on one host.

---

## Spike 7 — Axolotl config schema introspection

### The question

Advanced mode's design now rests on a sentence: *the exposed surface is
generated from Axolotl's own config schema, so it is exactly as wide as the
trainer and no wider.* That reconciles "expose every dial" with "unknown keys
are refused loudly" — the refusal is inherited from the trainer rather than
invented by the platform, and it cannot drift when the pinned digest moves,
because it is derived from the pinned digest.

**Whether that is an afternoon or a week depends entirely on a number nobody
has looked up: how many fields does that schema have?** Forty is a UI. Four
hundred is a different product.

### Method

No GPU, no VM. Run the pinned image locally, import Axolotl's pydantic config
model, and enumerate it:

1. Total field count, and the count after excluding fields that are
   infrastructure rather than training (paths, output dirs, logging sinks).
2. How many carry a default, how many are required, how many are typed as
   free-form.
3. Which fields are mutually exclusive or conditionally required, and whether
   that structure is expressed in the schema or only in Axolotl's runtime
   validation. **If the constraints live in runtime validation, a
   schema-derived form will happily accept combinations that fail four minutes
   into a paid job** — which is the exact class of failure the predictor
   exists to prevent.
4. Which of the correctness settings named in AGENTS.md appear, and under what
   names.

Then classify a sample into the three tiers the spec will use — exposed with a
calculated default; exposed with a named failure mode; known but unsupported,
with the reason — and record how long classifying twenty fields took, so the
full pass can be estimated honestly rather than guessed.

### Kill criterion

If the schema has no usable structure — constraints only in runtime, types
mostly free-form — then generation is not available and Advanced mode falls
back to a **hand-enumerated list with the derivation stated as rejected and
why.** That is still a defensible design; it is just a smaller one, and the
spec needs to know which it is writing.

### What it writes

`spike/findings-spike7.json` — field counts by category, constraint
expressiveness, the sample classification, and the measured minutes-per-field
for the full pass.

### What consumes it

The Advanced-mode spec, its ticket estimate, and the data file that holds the
tier classification — which is deliberately data rather than code, so the
classification is reviewable in a diff.

---

## Spike 8 — JarvisLabs SDK surface enumeration

### The question

The product's positioning includes reflecting what the provider offers. Nothing
in this repo enumerates what the SDK actually exposes: the
orchestrator uses create, destroy, list, and currency, and those were the four
that Phase A needed. **Four calls is what was required, not what exists**, and
the difference has never been looked at.

**Pause is the one to look for specifically.** If a VM can be paused more
cheaply than it can be destroyed and recreated, that changes three flows at
once: cancellation stops being purely destructive, OOM retry stops paying a
2–4 minute cold start per attempt, and resume-from-checkpoint on a fresh
machine becomes resume-on-the-same-machine. Each of those is currently
specified around a cost that pausing might remove.

### Method

No GPU. Read the installed SDK and the published docs together, and record
where they disagree — Phase A already found one silent divergence (startup
scripts accepted and ignored on VMs), so disagreement is expected rather than
surprising.

Enumerate: instance lifecycle beyond create/destroy (pause, resume, restart,
resize), storage operations (attach, resize, persistent volumes), image and
template controls, SSH key management, availability and pricing queries, any
event or webhook surface, and anything billing-adjacent that reads rather than
writes.

For each: is it documented, is it in the SDK, and has it been tested here. Mark
all three separately — **"documented" and "works on VMs" have already been
shown to be different claims on this platform.**

### Kill criterion

None. This spike cannot fail; it can only return a smaller list than hoped. But
if pause exists and is cheaper, **it reopens the cancellation ADR** (ADR-0003,
cancellation is destructive) and that reopening is the point.

### What it writes

`spike/findings-spike8.json` — the capability table with documented / in-SDK /
tested marked separately, and an explicit note on pause pricing if pause
exists.

### What consumes it

Whether ADR-0003 is superseded. The OOM-retry ticket's cost model. The
"reflects the provider's features" claim, which becomes either specific or
struck.

---

## Spike 9 — Streaming validation throughput

### The question

The upload ceiling is 1 GB, and the README is explicit that this is **a limit
of the in-memory validation path, which holds about 4.8× the file size — not a
product rule.** Streaming validation removes it.

That ceiling was fine when the largest model was 8B. It is not fine alongside a
claim about 70B and real-world use cases: a 70B fine-tune on a 200 MB dataset
is its own kind of toy, and the ceiling now contradicts the scale claim
directly.

Two numbers are needed before the ticket is written:

- **Throughput.** How many MB/s does validation sustain when it does not hold
  the file? If a 5 GB dataset takes six minutes to validate, that is a UX
  problem — the user is staring at a page — and it wants finding today rather
  than in front of a reviewer.
- **Token counting cost.** The quote is priced per training token, so token
  counting is a hard prerequisite of the quote, and the quote is the highest-
  value flow in Lane B. Tokenising a multi-GB dataset is the expensive half of
  validation and has never been timed.

### Method

No GPU, no VM. Generate synthetic JSONL at 1, 5 and 20 GB with a realistic row
shape, then measure a streaming pass: MB/s and peak RSS for parse-and-validate
alone, and again with tokenisation included. Confirm peak memory stays flat as
file size grows — **that flatness is the whole claim**, and a streaming path
that quietly accumulates is worse than an honest ceiling.

Record where the line-number guarantee costs anything: every error names its
line, and holding that promise while streaming is the part most likely to leak
memory.

### Kill criterion

If tokenisation dominates to the point that a 5 GB dataset cannot be validated
in a time a user will wait, then validation and token counting **split into two
phases** — validate on upload, count tokens asynchronously before the quote —
and the quote gains a "counting" state. Better to design that now than to
discover it behind a spinner.

### What it writes

`spike/findings-spike9.json` — MB/s and peak RSS at each size, with and without
tokenisation, and the derived answer for what the upload ceiling becomes.

### What consumes it

The streaming-validation ticket, the token-counting ticket, the upload ceiling
in `config.py`, and the quote's dependency graph.

---

## Rules that apply to all five

- **Every path that creates a VM destroys it in a `finally` block and then
  confirms by listing instances across consecutive samples.** A destroy call's
  return value is not evidence, and neither is one absent listing (C17).
- **Anything that SSHes runs through PowerShell, not Git Bash.** Git Bash
  cannot see the Windows `ssh-agent` service, and the failure presents as a
  dead VM. That misdiagnosis has already cost an evening and produced a wrongly
  filed platform bug.
- **`python -u`** for anything backgrounded.
- **Findings are written as JSON in the same shape as spikes 1–4**, because the
  spike directory is the evidence base and a finding that only exists in a
  terminal scrollback is not evidence.
- **Mark every number as measured or derived.** The spike directory's value is
  that its numbers are the first kind.

## What this spec produces

- Five findings files, and the amendments they force to Specs 005+.
- **ADR-0009**, superseding ADR-0004: the machine holds no long-lived
  credential and cannot enumerate storage, but may write its own artifact to a
  pre-signed URL scoped to one key. Written after Spike 5 confirms the sizes
  that force it.
- A go/no-go on each capstone run: 8B full FT on 2× H100 (2026-08-27) and 70B
  QLoRA on 1× RTX-PRO6000 (2026-08-28).

## Outcomes — run 2026-08-23

All five ran. Findings in `spike/findings-spike5.json` … `findings-spike9.json`,
written up in [spike/README.md](../../spike/README.md#phase-b-spikes--2026-08-23).
Total GPU spend across every attempt, including three runs of spike 6: **~₹25**.

| # | Kill criterion | Fired? | Answer |
| --- | --- | --- | --- |
| 5 | disk cannot exceed ~200 GB | **no** | **7200 GB**, named by the API's own refusal. 70B is not disk-bound. Download 364 MB/s |
| 6 | FSDP does not run in the pinned image | **partly — see below** | it runs, shards, checkpoints and resumes. **The loss is `nan`** |
| 7 | schema has no usable structure | **yes, in part** | 388 fields; only ~12% of constraints are in the schema. Derive the form, hand-write the rules |
| 8 | (cannot fail) | — | **pause exists.** ADR-0003 flagged for reopening |
| 9 | tokenisation dominates beyond a tolerable wait | **yes** | needs to be 24.8× faster to fit 60s. Counting splits into its own phase |

### Go / no-go on the capstones

**8B full fine-tune on 2× H100, 2026-08-27 — NO-GO as things stand.**

Not for the reason the spike expected. The mechanism is proven: two devices are
visible inside the pinned image, FSDP FULL_SHARD shards and steps, a
`torch.distributed.checkpoint` `.distcp` checkpoint is written, and **sharded
resume works** — which was the step this spike existed to doubt.

**The loss collapses to zero with a `nan` grad_norm on every step, in both
model cases.** Taking steps is not training. A capstone that shards perfectly
and learns nothing is a *more* expensive failure than one that will not launch,
because it produces an artifact that looks like a success and nothing
downstream would catch it. The `nan` gets explained first — tracked as
[#81](https://github.com/thp728/temper/issues/81).

**70B QLoRA on 1× RTX-PRO6000, 2026-08-28 — the disk question is closed.**
Spike 5 removes the constraint that would have blocked it: 7200 GB of disk and
2.7s per GB of download mean a 141 GB model lands in about six minutes. Whether
it goes ahead now depends on VRAM and on the same `nan`, not on storage.

### Amendments these force

- **ADR-0009** written, superseding ADR-0004's property 2, on the measured
  sizes — a 13.6 GB sharded checkpoint at 4B is the number that forces it.
- **ADR-0005 corrected**: the 4.8× memory multiplier was extrapolated from small
  files; measured at a real 1 GB file it is **5.93×**.
- **ADR-0003 flagged for reopening**: the provider can pause, which is a second
  cheap outcome for a stopped run.
- **Spec 005 (predictor and quote)** gains two measured anchors it did not have:
  peak VRAM per device of 4,678 MiB at 0.6B and 21,162 MiB at 4B under FSDP,
  and a `preparing` ETA derived from 2.7s per GB rather than extrapolated.
- **Spec 006 (streaming transport)** gains a `counting` state on the quote, and
  the upload ceiling becomes a property of a streaming path that holds +4 MB
  flat from 1 GB to 20 GB.
- **Spec 009 (advanced mode)** is re-scoped: the field list is generated from
  the pinned digest, the cross-field rules are hand-enumerated, and the estimate
  uses ~120 design-relevant fields rather than 344.

## Related

- [spike/README.md](../../spike/README.md) — spikes 1–4, what each proved and cost
- [ADR-0004](../adr/0004-the-machine-is-a-pure-compute-node.md) — the transport rules these spikes test the limits of
- [ADR-0007](../adr/0007-the-feasibility-warning-is-an-estimate-and-warns-rather-than-blocks.md) — the estimate-versus-block pattern the predictor inherits
