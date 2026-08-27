# Trainer Image

The pinned training container. Job spec in, adapter out.

This directory holds the Dockerfile and the entrypoint. `thinking.py` is not here: it lives in `packages/core` because the control plane validates thinking mode with the same module this image runs, and a second copy beside the entrypoint would be the hand-mirrored definition [ADR-0010](../../docs/adr/0010-the-repository-is-laid-out-as-apps-and-packages.md) forbids. The build context is therefore assembled rather than pointed at — `TRAINER_SOURCES` in the orchestrator is the list, `just image` builds from it locally, and a real job ships the same files as one tar.

## Why Axolotl is the base

Spike 3 measured the cost of owning the version matrix. Installing *latest* gave `transformers 5.15.0` + `trl 1.10.0` on `torch 2.5.1`, and three things chained:

1. TRL 1.x's `SFTConfig` **no longer inherits `TrainingArguments`** → `warmup_ratio` raised `TypeError`
2. `save_safetensors` was rejected → checkpoints written as torch `.bin`
3. transformers 5.x **refused to load them** — requires `torch >= 2.6`

**Result: checkpoint/resume was impossible.** Not a design flaw — four packages moving independently.

Axolotl's maintainers resolve and test that matrix on every build, and **Axolotl's own config surface still exposes `warmup_ratio`**, so the defaults the research settled on stay expressible regardless of what TRL does to its API. Delegating the matrix to people who test it beats pinning six packages by hand and re-resolving whenever one moves.

The base carries **torch 2.12** — far past the 2.6 floor that blocked resume — with CUDA 13.0 and Python 3.12.

## Pin discipline

**The digest is the contract. The tag is a comment.**

```
axolotlai/axolotl:main-20260817-py3.12-cu130-2.12.0
@sha256:29327e75d7ae0348e001809df5c11151887fdeec63fe4bc5c4330917777c701e
```

Changing either is a deliberate act that re-runs the GPU smoke test. Our layer adds **two files** — `entrypoint.py` from this directory and `thinking.py` from `packages/core` — and installs nothing; a `pip install` here would reintroduce the exact resolution problem this base exists to avoid.

⚠️ **`thinking.py` was missing from the COPY until 2026-08-19.** It was added to `entrypoint.py` as a top-level import after this image was last built, so nothing caught it until the first assembled run was read line by line. **Anything the entrypoint imports has to be listed here**, and the failure mode is the one below.

## Contract

| Path | Direction | Contents |
| --- | --- | --- |
| `/job/job.json` | in | job spec — see [job.example.json](job.example.json) |
| `/job/dataset.jsonl` | in | one JSON object per line |
| `/out/config.yaml` | out | the rendered Axolotl config, for auditability |
| `/out/run/` | out | checkpoints and the adapter |
| `/out/result.json` | out | **always written, including on failure** — see the boundary below |
| `/out/train.log` | out | full training output — also relayed to the container's stdout as it is produced |

## The machine may write its own artifact (ADR-0009)

When the job spec carries an `artifact_upload` block — a write URL minted by
the control plane for exactly this job's artifact key, expiring with the job —
the trainer **PUTs the adapter to that URL itself** after training, instead of
the control plane pulling it over SSH. The machine holds no credential: the
URL *is* the authorisation, and it grants one write to one key and nothing
else. The outcome of the upload is recorded in `result.json` under
`artifact_upload` — the trainer never takes the machine's word that the bytes
arrived, but it also never lets a failed upload crash the job before it is
reported. The control plane verifies what landed against the SHA-256 this
trainer computes, and only then may the job report success.

A standalone run (no grant in the spec) simply leaves the artifact on `/out`;
`artifact_upload` records that nothing was uploaded, as a fact rather than a
failure of training. The upload streams a file body with a declared
Content-Length from the standard library only — nothing new is installed in
the image, per the pin discipline above.

⚠️ **The boundary of "always written".** The guarantee comes from a `try/finally` inside `main()`, so it holds only from the moment `main()` is entered. **An import-time failure escapes it entirely** — the container exits with no `result.json`, and the orchestrator reports `training_failed: "Trainer produced no result.json"`, an error that points at training and says nothing about the image. That is exactly what a missing COPY produces. A guarantee whose boundary is undocumented is one you will over-trust.

```bash
docker run --rm --gpus all \
  -v /path/to/job:/job:ro \
  -v /path/to/out:/out \
  ghcr.io/<owner>/finetune-trainer@sha256:<digest>
```

## The job specification carries every value

**The trainer resolves nothing.** The control plane resolves every
hyperparameter before launch (issue #83) and writes the full resolved set into
`hyperparameters` in `job.json` — defaults, α recomputed from a moved rank,
rsLoRA inferred at rank ≥ 32. There is one resolver, it runs before any money
is spent, and its output is visible in the job's record; the trainer applies
exactly what arrives.

Three consequences, each enforced by tests in `tests/`:

- **A missing required key fails loudly.** `spec_incomplete`, naming the keys.
  No fallback: training on a number nobody chose is worse than not training.
- **Unknown keys are refused loudly and echoed back**, at the top level and
  inside `hyperparameters`, under `rejected_overrides` in `result.json`.
  Until #33 generates the surface from the pinned image's own schema, "known"
  means read by this entrypoint when rendering `config.yaml`
  (`REQUIRED_HYPERPARAMETERS` / `KNOWN_HYPERPARAMETERS`).
- **The trainer derives nothing.** Whatever α–r pairing or rsLoRA flag the
  spec carries is what trains, even when visibly strange — second-guessing
  the spec is the second resolver back through the wall.

### Running standalone

Outside the product you supply the complete spec yourself: copy
[job.example.json](job.example.json), which carries every required key with
the research defaults already resolved, mount it at `/job/job.json`, and run.
The required keys are exactly what `temper_core.hyperparams.effective({})`
produces — that equality is pinned by `test_agreement_with_the_domain.py`.

## What is settable, and what is not

The defaults and the overridable keys are **data in one place**:
[`packages/contracts/trainer-defaults.json`](../../packages/contracts/trainer-defaults.json),
read by `temper_core.hyperparams` to resolve the spec before launch (#82).
Everything under `hyperparameters` is then applied as given; the resolver
decides what may appear there (`ALLOWED_OVERRIDES` in that file).
**Default-locked, not hidden** — chat-template resolution, `train_on_inputs: false`,
EOS handling, NF4 double-quant, `lora_target_linear`, bf16, seed — are
constants of this entrypoint, not resolved values. They are the
highest-frequency silent-failure surface: they pass every obvious health check
and only surface as garbage generations. Advanced mode (#33) exposes each one
with its failure mode named inline; the export-time template probe is what
makes exposure safe.

Two behaviours worth knowing:

- **Unknown keys come back in `result.json` as `rejected_overrides`** — an
  override the caller believes is in effect but isn't is worse than a refusal.
- **α tracks r and rsLoRA follows rank at resolution time**, before launch,
  in `temper_core.hyperparams` — no longer here.

## Verified 2026-08-18 (spike 4)

Built and run on an L4. **The image works, and resume works.**

| | |
| --- | --- |
| Build from pinned digest | 183s, 8.5 GB |
| Job through `/job` → `/out` | 120.9s, 64 rows, exit 0 |
| `axolotl train` accepts `warmup_ratio` | ✅ — the argument TRL 1.x removed |
| **Resume from `checkpoint-2` → step 6** | ✅ 54s |
| Adapter | **safetensors**, 132.2 MB, `r=16 alpha=32 rslora=False` |
| `target_modules` | all seven — `q,k,v,o,gate,up,down` |

**C14 closed.** The version chain that made resume impossible on an unpinned stack is gone.

Spike 4 also caught a real bug here: unknown keys at the **top level** of the job spec were not refused, only unknown keys inside `hyperparameters`. A caller misspelling `max_steps` as `maxSteps` would have got a full-length run with no warning. Both levels are now validated.

## Verified 2026-08-19 (first run through the product)

Same image, driven by `api/orchestrator.py` rather than a spike script. **L4, 336s job wall clock; the VM lived 363s, ≈₹4.8 derived from provision-to-teardown at ₹41.31/hr.**

| | |
| --- | --- |
| Build from pinned digest | **87s** — pull throughput varies; 183s on 08-18 |
| `axolotl train`, 64 rows × 3 epochs | 161.4s, 23 steps, exit 0 |
| Checkpoints written | `checkpoint-8`, `checkpoint-16`, `checkpoint-23` |
| Adapter shipped | `adapter_source: "final"`, 132.2 MB, verified by SHA-256 end to end |
| Thinking mode | detected `false` from the data, not configured |

⚠️ **`collect_artifacts()` picked the wrong adapter until this run.** `sorted(rglob(...))[-1]` sorts lexicographically, so `checkpoint-8` sorted last of those three — and every `run/checkpoint-N/…` sorts after `run/adapter_model.safetensors` anyway. It would have shipped **step 8 of 23** as the finished model, silently, with a valid hash. Selection is explicit now and `result.json` records which via `adapter_source`. **A hash check proves the bytes arrived intact; it says nothing about whether they were the right bytes.**

## Still open

- **fp32 adapters.** 132.2 MB is 33M × 4 bytes; bf16 would halve it. Undecided — it doubles artifact size and transfer cost. Re-confirmed 2026-08-19 from the safetensors header: **504 tensors, every one `F32`**. Note this is a size policy, not a defect — `bf16` is the *compute* dtype, and fp32 trainable parameters under 4-bit quantisation are standard.
- ~~**Qwen3 thinking mode.** Unhandled.~~ ✅ **Closed 2026-08-18.** Detected from the dataset in `thinking.py`, applied identically at training and serving; mixed datasets block with a line-numbered error. Confirmed on the 08-19 product run: `0/64 assistant turns have <think>` → `enable_thinking: false`.
- **`chat_template: tokenizer_default` ran without error, but nothing asserts it resolved to the *right* template.** The export-time probe from report A §4.2 — re-tokenise a fixed conversation through both the training template and the artifact's, assert identical ids — is the check, and it is not written yet.
- **`sample_packing`** stays off pending a per-model varlen-attention check.
- **Not pushed to GHCR.** The image builds per-VM today. Pushing needs a token and belongs in CI. At 87–183s per run it is also the largest remaining slice of cold start.
- **No assertion that the shipped adapter matches the run's final step.** `adapter_source` records the choice but nothing cross-checks it against the trainer's last step — that is the check that would have caught the selection bug above, and it is not written.
- ~~**Training output never leaves the container.**~~ ✅ **Closed 2026-08-21.** The entrypoint now relays `axolotl`'s output to the container's stdout line by line as it is produced, and the machine no longer parks it in a file — so it reaches the control plane during the run. `/out/train.log` is still written as the copy that survives on the machine if the stream breaks. Splitting on `\r` as well as `\n` is load-bearing: tqdm redraws a progress bar without ever sending a newline, and a reader that waits for one sees nothing for the whole bar. See [ADR-0001](../docs/adr/0001-event-channel-over-ssh-stdout.md).
- **The loss is in the stream but not yet structured.** Lines carrying step, loss and epoch arrive as ordinary log events; promoting them to metric events is issue #5, and the loss curve is Phase B.
