# ADR-0021 — The first assembled run: it works, and three defects that reading caught before the GPU did

- **Status:** accepted
- **Date:** 2026-08-19

> **A note on the number and the date.** This decision was made on
> 2026-08-19 and filed in this directory on 2026-08-26
> ([#26](https://github.com/thp728/temper/issues/26)). Its file number reflects
> when the record landed here, not when the decision was made. The wording is
> the original: file paths are given as the record wrote them — where a file
> has since moved or changed (for example `thinking.py` now lives in
> `packages/core`, and the trainer image copies both modules), the record
> below describes the state that produced the decision.

## Context

**The checkpoint question is answered: a user can go dataset → adapter through the product.**
Measured, on an L4, 2026-08-19: job `job_973e2b46ba97`, **336s wall clock**, ending in a
132 MB LoRA the user downloaded from the product. Stage timings, all measured: provision 10s,
SSH ready 47s, image build 87s, `axolotl train` 161s, adapter fetch and verify 20s. Teardown
confirmed by listing instances from a separate process, not by trusting the destroy call.

The orchestrator had been stubbed in all 26 tests and had never provisioned a VM; every piece
under it was proven individually. Assembling proven parts is not proof of the assembly. Three
defects were found by reading, each of which would have cost a full provision-build-fail cycle to
discover on hardware, and two of which would have failed *quietly enough to ship*.

### 1. `thinking.py` was never added to the trainer image

`apps/trainer/Dockerfile` copies `entrypoint.py` only. `entrypoint.py` grew a **top-level**
`from thinking import ...` in commit `babad11`, which landed after the Dockerfile was written
and after spike 4 last built the image — so nothing had rebuilt it since. The container would
have died on import, **before the `try/finally` that guarantees `result.json` is always
written**, and the orchestrator would have reported
`training_failed: "Trainer produced no result.json"` — an error pointing at training that says
nothing about a missing file in the image.

**The interesting part is not the missing COPY, it is the hole in the guarantee.** "result.json
is always written" holds only from the moment `main()` is entered. An import-time failure
escapes it entirely, and that is the exact failure a Dockerfile drift produces. The guarantee
needs its boundary stated in `apps/trainer/README.md` rather than left implied.

### 2. The control plane never loaded the API key

Every spike calls `load_dotenv(spike/.env)` itself. Nothing in the control plane did. The SDK resolves its
token as explicit argument → `JL_API_KEY` → `~/.config/jl/config.toml`, and there is **no
config file on this machine** — all three platform paths checked and absent. `Client()` inside
the orchestrator would have raised `AuthError` on the first job, caught by the generic handler
and reported as `internal_error`, the least useful code in the set.

Fixed with a config module (`temper_control_plane.config`), loading the same file the spikes do,
environment winning over file. Two things deliberately added beyond the load: a **startup
warning** when no credential resolves, and a distinct **`provider_unauthenticated`** code
checked *before* `Client()` is constructed. Discovering a missing credential four seconds into a
run is survivable; reporting it as `internal_error` is not, because it sends the reader to the
wrong place.

### 3. The adapter was not loadable — and would have been the wrong adapter

Two faults in what is literally the deliverable.

*The download shipped a bare `.safetensors`.* PEFT cannot load LoRA weights without
`adapter_config.json` beside them: rank, alpha and target modules live there. The user got a
file that looks like the deliverable and is not one. The config was **already inside
`result.json`**, so shipping it costs no extra transfer. The download is now a zip of the
adapter directory.

*And artifact selection picked the wrong file.* A lexicographic sort over `run.rglob(...)[-1]`
ranked any `run/checkpoint-N/...` above `run/adapter_model.safetensors`.
**The run makes this concrete rather than theoretical:** it produced `checkpoint-8`,
`checkpoint-16`, `checkpoint-23`. Lexicographic order puts `checkpoint-8` last — so the old
code would have shipped the adapter from **step 8 of 23**, a third of the way through training,
labelled as the finished model. Silently, with a valid hash, on every run. Selection is now
explicit — final adapter first, then the numerically highest checkpoint — and `result.json`
records which via `adapter_source` (`"final"` on this run), so *"which weights did I
download?"* is answered in the artifact rather than inferred.

**Why all three are logged together:** they are one class of error — *a component was proven,
then its context changed underneath it*. The image was proven before `thinking.py` existed;
credential loading was proven inside a script that loads credentials explicitly; the adapter
selector was proven against a run that produced fewer than ten checkpoints. **"Proven" carries
a date and a set of assumptions**, and spike findings files should record both.

## Decision

Read the assembled path end to end *before* paying for it.

## Alternatives considered

*Run it and let the GPU find the bugs* — rejected, and not because of the ₹4.
Each cycle is ~6 minutes of wall clock against a Friday tripwire, and a run that dies at the
image build tells you nothing about the seven steps after it. Reading first turned one paid run
into a test of the whole path rather than a test of its first failure. *Fix only the two
certain blockers and defer the adapter work* — rejected, because "dataset → adapter" is the
checkpoint question and an adapter that does not load does not answer it.

## Consequences

The fixes were written unverified-on-hardware and verified by the run
afterwards. That is the right order, but at the moment they were written they were reasoning,
not measurement, and the commit says so.

### ⚠️ Correction 2026-08-19 — the cost figure was not measured

This entry first said **₹3.8**, and four other documents said **₹5.02**. **Neither derives from
the run record, and nothing in the code computes a cost at all.** What the events actually
support: the VM was created at provisioning and destroyed 363s later, so at ₹41.31/hr with
per-minute billing that is 7 billed minutes, **≈₹4.8**. The 336s figure is the *job* wall clock
to `complete`, which ends 28s before teardown — a smaller window than the one being paid for,
and using it undercounts.

**Worth logging rather than silently correcting**, because the whole discipline of this project
is separating measured from derived, and a number was carried into five documents wearing the
word *measured* while being neither read off an invoice nor computed by anything. **Job cost
belongs in the run record**, derived from provision-to-teardown against the price the
orchestrator already stores per job — it is also the input the Phase B quote gets calibrated
against.

## What the run exposed that is not yet fixed

Recorded here because these were the next decisions, not incidental notes.

**Training was 251 seconds of total silence.** The orchestrator issued one blocking call over
SSH for the whole build-and-train phase and only converted stderr into events *after it
returned*. Measured: all 13 log events carried the same timestamp, 318s. Worse, `axolotl`'s own
output never left the VM at all — `entrypoint.py` redirected it to `/out/train.log` inside the
container, which is destroyed with the machine. **There was no loss number anywhere retrievable,
during or after a run.** So the loss-curve task was not "parse metrics out of the stream";
there was no stream, and building one was the prerequisite.

**The adapter is float32, not bf16.** Measured from the safetensors header: 504 tensors
(36 layers × 7 target modules × 2), every one `F32`, which is why it is 132 MB rather than
~66 MB. The trainer config sets `bf16: true` and design notes said "adapters in bf16". **That
claim was measurably false as written.** The behaviour is probably correct — keeping trainable
parameters in fp32 under 4-bit quantisation is standard and is what
`prepare_model_for_kbit_training` does — but the documented claim described the compute dtype,
not the saved adapter dtype. Recorded as a decision in its own right:
[ADR-0008](0008-adapters-ship-as-fp32.md). Verified correct in the same header: r=16, α=32,
`use_rslora: false` (right, r < 32), and all seven linear projections targeted, confirming
all-linear rather than attention-only.

**`MAX_GPU_MINUTES = 90` was defined and never read.** The GPU-minute cap was a constant, not a
control. Cancellation likewise did not exist — `cancelled` was in the state machine with no
path that reached it. Both were on the Phase A list already; the point is that the safety
control currently *looked* implemented at a glance.

**Teardown events were invisible to a client that stops at the terminal state.** The
teardown messages landed *after* the `complete` transition, so a watcher that stops polling on
`complete` — as any reasonable UI would — never sees the teardown it most wants confirmed. The
UI needs to keep reading events briefly past the terminal state, or teardown needs to precede
the terminal transition.

**A note on the stray check, because it is the money control.** It ran and reported clean, and
the account was independently confirmed clean. But the provider's instance list is
*eventually consistent* after a destroy: sampled from a separate process, the machine was
absent immediately, then visible as `Destroying`, then absent for good. The check happens to
sample at the moment it reads absent. **A confirmation that passes because it queried during a
gap is not a confirmation**, and this one has not been tested against a destroy that actually
failed. It should poll until the instance is absent across consecutive samples, or explicitly
treat `Destroying` as not-yet-confirmed.

## Rollback

All three fixes are additive and independently revertible — one new module, one COPY line, one function in the entrypoint plus one route body.
