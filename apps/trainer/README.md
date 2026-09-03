# Trainer image

The pinned training container. Job spec in, artifact out.

This directory holds the Dockerfile and the entrypoint. `thinking.py` is not here: it lives in `packages/core` because the control plane validates thinking mode with the same module this image runs, and a second copy beside the entrypoint would be the hand-mirrored definition [ADR-0010](../../docs/adr/0010-the-repository-is-laid-out-as-apps-and-packages.md) forbids. The build context is therefore assembled rather than pointed at. `TRAINER_SOURCES` in the orchestrator is the list, `just image` builds from it locally, and a real job ships the same files as one tar.

## Why Axolotl is the base

Spike 3 measured the cost of owning the version matrix. Installing *latest* gave `transformers 5.15.0` and `trl 1.10.0` on `torch 2.5.1`, and three things chained:

1. TRL 1.x's `SFTConfig` no longer inherits `TrainingArguments`, so `warmup_ratio` raised `TypeError`
2. `save_safetensors` was rejected, so checkpoints were written as torch `.bin`
3. transformers 5.x then refused to load them, because it requires `torch >= 2.6`

Checkpoint and resume were impossible. That is not a design flaw, it is four packages moving independently.

Axolotl's maintainers resolve and test that matrix on every build, and Axolotl's own configuration still exposes `warmup_ratio`, so the defaults the research settled on stay expressible regardless of what TRL does to its API. Delegating the matrix to people who test it beats pinning six packages by hand and re-resolving whenever one moves.

The base carries torch 2.12, well past the 2.6 floor that blocked resume, with CUDA 13.0 and Python 3.12.

## Pin discipline

The digest is the contract. The tag is a comment.

```
axolotlai/axolotl:main-20260817-py3.12-cu130-2.12.0
@sha256:29327e75d7ae0348e001809df5c11151887fdeec63fe4bc5c4330917777c701e
```

Changing either is a deliberate act that re-runs the GPU smoke test. Our layer adds two files, `entrypoint.py` from this directory and `thinking.py` from `packages/core`, and installs nothing. A `pip install` here would reintroduce the exact resolution problem this base exists to avoid.

Worth knowing: `thinking.py` was missing from the COPY until 2026-08-19. It was added to `entrypoint.py` as a top-level import after this image was last built, so nothing caught it until the first assembled run was read line by line. Anything the entrypoint imports has to be listed here, and the failure mode is the one described below.

## Contract

| Path | Direction | Contents |
| --- | --- | --- |
| `/job/job.json` | in | job spec, see [job.example.json](job.example.json) |
| `/job/dataset.jsonl` | in | one JSON object per line |
| `/out/config.yaml` | out | the rendered Axolotl config, for auditability |
| `/out/run/` | out | checkpoints, and the final adapter or trained model |
| `/out/model.tar.gz` | out | a full fine-tune's whole-model artifact, one streamed archive (issue #66) |
| `/out/result.json` | out | always written, including on failure. See the boundary below |
| `/out/train.log` | out | full training output, also relayed to the container's stdout as it is produced |

## Two methods (issue #66)

The job spec's `method` selects the training shape: `qlora` (the adapter the
platform shipped with, NF4 double-quant) or `full` (a whole-model fine-tune).
The control plane chooses the method at provisioning and writes it into the
spec beside the method-resolved hyperparameters. A full job's spec carries
its own lower learning rate, resolved from the one defaults contract
(`packages/contracts/trainer-defaults.json`, its `by_method` table) rather
than recomputed here.

- `qlora` renders an adapter config (`adapter`, `load_in_4bit`, the LoRA
  keys) and collects the adapter pair as the artifact.
- `full` renders no adapter config and trains every weight in bf16. Its
  artifact is the trained model directory, tarred on the machine into
  `model.tar.gz` (streamed, never held whole) and uploaded through the same
  scoped grant an adapter uses. The control plane verifies the archive's
  checksum exactly as it does an adapter's, and the download serves it with a
  manifest declaring the `full_model` kind.

A spec with no `method` reads as `qlora`, the only method that ever ran
before the key existed.

## The machine may write its own artifact (ADR-0009)

When the job spec carries an `artifact_upload` block, meaning a write URL
minted by the control plane for exactly this job's artifact key and expiring
with the job, the trainer PUTs the adapter to that URL itself after training
instead of the control plane pulling it over SSH. The machine holds no
credential. The URL *is* the authorisation, and it grants one write to one key
and nothing else. The outcome of the upload is recorded in `result.json` under
`artifact_upload`. The trainer never takes the machine's word that the bytes
arrived, but it also never lets a failed upload crash the job before it is
reported. The control plane verifies what landed against the SHA-256 this
trainer computes, and only then may the job report success.

A standalone run, with no grant in the spec, simply leaves the artifact on
`/out`. `artifact_upload` then records that nothing was uploaded, as a fact
rather than a failure of training. The upload streams a file body with a
declared Content-Length from the standard library only, so nothing new is
installed in the image, per the pin discipline above.

## Checkpoints leave the machine as they are produced (issue #37)

When the job spec carries a `checkpoint_grants` block, meaning one scoped write
URL per retention slot, minted by the control plane and expiring with the job,
the trainer ships each completed checkpoint to its slot during training,
through the same ADR-0009 machinery the artifact uses. A background thread
watches `/out/run/` and uploads each `checkpoint-N` once it is complete: its
own `trainer_state.json` must exist and agree on the step, which is how a save
cut off mid-way is kept out of storage. Each checkpoint is tarred and PUT as
one streamed object, nothing held whole, and the i-th successful upload
overwrites slot `i mod N`, so storage never holds more than `N` checkpoints
per job and retention is bounded by construction.

Training never waits for an upload. The thread uploads one checkpoint at a
time in the background, and the entrypoint's `finally` runs one last sweep so
the final checkpoint, the one a resumption would most want, ships even if
the poll never saw it. Each upload's step, held-out loss (where the step was
evaluated), slot, checksum and outcome are reported in `result.json` under
`checkpoints`. The control plane streams each stored object back through the
checksum and records it as complete only when they match. A standalone run
carries no grants and leaves its checkpoints on `/out`, reported as a fact
rather than a failure of training.

### The boundary of "always written"

The guarantee comes from a `try/finally` inside `main()`, so it holds only from
the moment `main()` is entered. An import-time failure escapes it entirely. The
container exits with no `result.json`, and the orchestrator reports
`training_failed: "Trainer produced no result.json"`, an error that points at
training and says nothing about the image. That is exactly what a missing COPY
produces. A guarantee whose boundary is undocumented is one you will over-trust.

```bash
docker run --rm --gpus all \
  -v /path/to/job:/job:ro \
  -v /path/to/out:/out \
  ghcr.io/<owner>/finetune-trainer@sha256:<digest>
```

## The side-by-side comparison runs on the warm machine (issue #69)

After training, before result.json is written, the entrypoint runs the same
held-out prompts through the base model and through the checkpoint the run's
own selection rule would choose. That is the same `select_best_checkpoint` the
control plane records the choice with, and `checkpoint.py` ships flat into the
image like `split.py`. Generation uses fixed decoding settings defined once in
`comparison.py` and recorded on the result, so a reader can tell whether two
outputs are comparable, and the interface displays them from the record
(ADR-0010). The whole step, including loading the models, can never fail the
run. A failure is recorded under `comparison` with its reason and the artifact
is still delivered. See [ADR-0061](../../docs/adr/0061-the-side-by-side-comparison-runs-on-the-warm-machine.md).

## The general-capability slice runs on the warm machine (issue #73)

After the comparison, before result.json is written, the entrypoint answers a
fixed, versioned slice of general-knowledge questions through the base model
and through the chosen checkpoint, and records the difference as a delta with
its sample size and uncertainty stated. It is a smoke test for catastrophic
forgetting rather than a benchmark. The slice is small by design (each question
costs warm-machine time), fixed (the same questions every job), general
(never the user's task), and versioned (`CAPABILITY_SLICE_VERSION`), and a
result is reported beside the number of questions it rests on and the standard
error of the change. A tuned model that loses at least
`LARGE_REGRESSION_QUESTIONS` questions to the base is flagged
(`large_regression`, with the threshold recorded) so the interface can show
it prominently from the record.

Both eval steps, the comparison and the capability slice, load the two
models exactly once between them. The entrypoint selects the checkpoint and
loads the base and tuned generators, then hands the same pair to both, so the
paid machine is not asked to load twice for two steps that take moments where
the weights already are. Either step can fail without failing the run. A
failure is recorded under its own key, `comparison` or `capability`, with its
reason, and the artifact is still delivered. See
[ADR-0067](../../docs/adr/0067-the-general-capability-check-is-a-small-slice-whose-limits-are-stated.md).

## The job specification carries every value

The trainer resolves nothing. The control plane resolves every
hyperparameter before launch (issue #83) and writes the full resolved set into
`hyperparameters` in `job.json`: defaults, α recomputed from a moved rank,
rsLoRA inferred at rank ≥ 32. There is one resolver, it runs before any money
is spent, and its output is visible in the job's record. The trainer applies
exactly what arrives.

Three consequences, each enforced by tests in `tests/`:

- **A missing required key fails loudly.** `spec_incomplete`, naming the keys.
  There is no fallback, because training on a number nobody chose is worse than
  not training.
- **Unknown keys are refused loudly and echoed back**, at the top level and
  inside `hyperparameters`, under `rejected_overrides` in `result.json`.
  Until #33 generates the advanced surface from the pinned image's own schema,
  "known" means read by this entrypoint when rendering `config.yaml`
  (`REQUIRED_HYPERPARAMETERS` and `KNOWN_HYPERPARAMETERS`).
- **The trainer derives nothing.** Whatever α–r pairing or rsLoRA flag the
  spec carries is what trains, even when visibly strange. Second-guessing
  the spec is the second resolver back through the wall.

### Running standalone

Outside the product you supply the complete spec yourself. Copy
[job.example.json](job.example.json), which carries every required key with
the research defaults already resolved, mount it at `/job/job.json`, and run.
The required keys are exactly what `temper_core.hyperparams.effective({})`
produces, and that equality is pinned by `test_agreement_with_the_domain.py`.

## What is settable, and what is not

The defaults and the overridable keys are data in one place,
[`packages/contracts/trainer-defaults.json`](../../packages/contracts/trainer-defaults.json),
read by `temper_core.hyperparams` to resolve the spec before launch (#82).
Everything under `hyperparameters` is then applied as given, and the resolver
decides what may appear there (`ALLOWED_OVERRIDES` in that file).

Some settings are default-locked rather than hidden: chat-template resolution,
`train_on_inputs: false`, EOS handling, NF4 double-quant, `lora_target_linear`,
bf16 and the seed are constants of this entrypoint, not resolved values. They
are the highest-frequency silent failures in this category, passing every
obvious health check and showing up only as garbage generations. Advanced mode
(#33) exposes each one with its failure mode named inline, and the export-time
template probe is what makes exposure safe.

Two behaviours worth knowing:

- **Unknown keys come back in `result.json` as `rejected_overrides`.** An
  override the caller believes is in effect but isn't is worse than a refusal.
- **α tracks r and rsLoRA follows rank at resolution time**, before launch,
  in `temper_core.hyperparams`. Neither happens here any more.

## Verified 2026-08-18 (spike 4)

Built and run on an L4. The image works, and resume works.

| | |
| --- | --- |
| Build from pinned digest | 183s, 8.5 GB |
| Job through `/job` → `/out` | 120.9s, 64 rows, exit 0 |
| `axolotl train` accepts `warmup_ratio` | yes, the argument TRL 1.x removed |
| Resume from `checkpoint-2` to step 6 | yes, 54s |
| Adapter | safetensors, 132.2 MB, `r=16 alpha=32 rslora=False` |
| `target_modules` | all seven: `q,k,v,o,gate,up,down` |

C14 closed. The version chain that made resume impossible on an unpinned stack is gone.

Spike 4 also caught a real bug here: unknown keys at the top level of the job spec were not refused, only unknown keys inside `hyperparameters`. A caller misspelling `max_steps` as `maxSteps` would have got a full-length run with no warning. Both levels are now validated.

## Verified 2026-08-19 (first run through the product)

Same image, driven by `api/orchestrator.py` rather than a spike script. L4, 336s job wall clock. The VM lived 363s, roughly ₹4.8 derived from provision-to-teardown at ₹41.31/hr.

| | |
| --- | --- |
| Build from pinned digest | 87s. Pull throughput varies, and it was 183s on 08-18 |
| `axolotl train`, 64 rows × 3 epochs | 161.4s, 23 steps, exit 0 |
| Checkpoints written | `checkpoint-8`, `checkpoint-16`, `checkpoint-23` |
| Adapter shipped | `artifact_source: "final"`, 132.2 MB, verified by SHA-256 end to end |
| Thinking mode | detected `false` from the data, not configured |

This run caught a bad one. `collect_artifacts()` picked the wrong adapter until it ran: `sorted(rglob(...))[-1]` sorts lexicographically, so `checkpoint-8` sorted last of those three, and every `run/checkpoint-N/…` sorts after `run/adapter_model.safetensors` anyway. It would have shipped step 8 of 23 as the finished model, silently, with a valid hash. Selection is explicit now, and `result.json` records which one via `artifact_source` (named `adapter_source` before issue #66). A hash check proves the bytes arrived intact. It says nothing about whether they were the right bytes.

## Still open

- **fp32 adapters.** 132.2 MB is 33M × 4 bytes, and bf16 would halve it. Undecided, because it doubles artifact size and transfer cost. Re-confirmed 2026-08-19 from the safetensors header: 504 tensors, every one `F32`. This is a size policy rather than a defect. `bf16` is the *compute* dtype, and fp32 trainable parameters under 4-bit quantisation are standard.
- ~~**Qwen3 thinking mode.** Unhandled.~~ Closed 2026-08-18. Detected from the dataset in `thinking.py`, applied identically at training and serving, and mixed datasets block with a line-numbered error. Confirmed on the 08-19 product run: 0 of 64 assistant turns have `<think>`, giving `enable_thinking: false`.
- **`chat_template: tokenizer_default` ran without error, but nothing asserts it resolved to the *right* template.** The export-time probe from report A §4.2, which re-tokenises a fixed conversation through both the training template and the artifact's and asserts identical ids, is the check. It is not written yet.
- **`sample_packing`** stays off pending a per-model varlen-attention check.
- **Not pushed to GHCR.** The image builds per-VM today. Pushing needs a token and belongs in CI. At 87–183s per run it is also the largest remaining slice of cold start.
- **No assertion that the shipped adapter matches the run's final step.** `artifact_source` records the choice, but nothing cross-checks it against the trainer's last step. That is the check that would have caught the selection bug above, and it is not written.
- ~~**Training output never leaves the container.**~~ Closed 2026-08-21. The entrypoint now relays `axolotl`'s output to the container's stdout line by line as it is produced, and the machine no longer parks it in a file, so it reaches the control plane during the run. `/out/train.log` is still written as the copy that survives on the machine if the stream breaks. Splitting on `\r` as well as `\n` is load-bearing: tqdm redraws a progress bar without ever sending a newline, and a reader that waits for one sees nothing for the whole bar. See [ADR-0001](../../docs/adr/0001-event-channel-over-ssh-stdout.md).
- **The loss is in the stream but not yet structured.** Lines carrying step, loss and epoch arrive as ordinary log events. Promoting them to metric events is issue #5, and the loss curve is Phase B.
