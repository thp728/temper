# Report D — Is This the Right Way to Use Axolotl in a Commercial Fine-Tuning Platform?

*Research briefing as of 24 Aug 2026. Reports A (data/tokenization/method selection), B (training loop/compute), and C (evaluation/artifacts/platform teardown) are assumed read. This report answers a single question from an outsider perspective, against **primary sources only**: Axolotl docs + repo, HuggingFace TRL/PEFT/transformers/accelerate docs + source, and the primary docs/repos of the two named alternatives (LLaMA-Factory, Unsloth). No blog summaries are cited for claims those projects own.*

*Confidence labels: **High** (corroborated by primary source + measured in this repo's spikes), **Medium** (one primary source, not yet measured here), **Low** (inferred, not directly documented). Every non-obvious claim carries a citation to the source that owns it.*

---

## TL;DR Verdict

**Yes, this is the correct way to use Axolotl for a commercial platform at this stage — and it will be attacked on exactly the parts you already expect.**

- **`axolotl train` via YAML + `FROM axolotlai/axolotl@sha256:…` + YAML rendered in Python + subprocess is the canonical, maintainer-intended integration seam.** The docs, CLI reference, quickstart, Docker docs, and the fact that Axolotl's entire config surface is a YAML-validated Pydantic model all point the same way. There is no stable programmatic Python API that a platform should prefer for training; the Python entry points exist as `python -m axolotl.cli.*` legacy shims, not as a public library. **Confidence: High.**

- **Pinning the Docker image by digest and delegating the six-package version matrix to Axolotl is the right dependency strategy.** The alternative — pinning torch/CUDA/transformers/trl/peft/bitsandbytes yourself — is exactly what Spike 3 proved fails in practice (TRL 1.x removed `warmup_ratio` from `SFTConfig`, checkpoints flipped to `.bin`, transformers 5.x refused to load them without `torch >=2.6`). Axolotl's Docker matrix (Python 3.12 / CUDA 13.0 / PyTorch 2.12.0 at the pinned tag) is tested on every build. Pinning six packages by hand reintroduces the resolution job Axolotl exists to do. **Confidence: High.**

- **Rendering YAML + subprocess is the right seam vs importing Axolotl as a library** for a control plane that must be orchestrator-agnostic (thread today, Temporal tomorrow) and must never couple to Axolotl's internal Python imports. The seam is narrow, auditable (`/out/config.yaml`), and replayable. The cost is that you inherit YAML's ergonomics and must validate before you launch a billing VM — which this repo already does (pure validation functions, loud rejection of unknown keys at both top-level and `hyperparameters`). **Confidence: High.**

- **The defensible parts under grilling are:** the version-matrix delegation (Spike 3/4), the digest pin, the `axolotl train` CLI, `train_on_inputs: false`, `chat_template: tokenizer_default`, NF4 double-quant + bf16, `lora_target_linear: true`, `save_safetensors: true`, thinking-mode detection, and the `/job -> /out` contract with always-written `result.json`.

- **What gets challenged:** the 344-field config schema hides ~88% of its constraints in Python `model_validator` hooks (Spike 7 measured 113 hooks, 1 field with schema bounds, 19 enums — so a form generated from the schema cannot know what will be refused until the job is already on a billing VM); `sample_packing: false` is correct today but leaves 3–5× throughput on the table and must be gated per-model behind varlen attention; single-GPU-only without an FSDP path wired through YAML is a scope cut that must be stated, not implied; and the `main-20260817` tag is a moving `main` build, not a semantic release tag — digest pinning mitigates but update cadence needs a policy.

- **Alternatives comparison in one line:** direct TRL/PEFT gives you maximum control and minimum abstraction at the cost of owning the version matrix forever; Unsloth gives you ~2× speed / ~70% VRAM savings via Triton kernels but narrows model coverage and couples you to its patch set; LLaMA-Factory gives you breadth (100+ models) + a Gradio UI but is a second framework to track and its own version matrix. For a platform whose differentiator is orchestration and correctness, not kernel speed, Axolotl is the right default with Unsloth as a conditional accelerator.

---

## 1. How Axolotl Is Designed to Be Used (Primary-Source Evidence)

### 1.1 The YAML + CLI is the product, not a convenience wrapper

Every first-party surface presents the same flow. The quickstart's "Your First Fine-tune" is three lines: `axolotl fetch examples` then `axolotl train examples/llama-3/lora-1b.yml` [docs.axolotl.ai — Quickstart](https://docs.axolotl.ai/docs/getting-started.html). The CLI reference defines the grammar as `axolotl <command> [config.yml] [options]` and documents `axolotl train config.yml` as the training entry point, with `--resume-from-checkpoint`, `--launcher torchrun/accelerate`, and CLI overrides (`--learning-rate`, `--micro-batch-size`) as secondary modifiers [docs.axolotl.ai — CLI](https://docs.axolotl.ai/docs/cli.html). The "Easy Configuration" feature bullet is explicit: *"Re-use a single YAML configuration file across the full fine-tuning pipeline: dataset preprocessing, training, evaluation, quantization, and inference"* [docs.axolotl.ai — Home](https://docs.axolotl.ai/).

The config reference dumps the entire surface as YAML keys backed by `axolotl.utils.schemas.config.AxolotlInputConfig` — a Pydantic model with 388 total fields (344 after excluding infrastructure), 386 carrying defaults, 113 `model_validator` hooks, 27 `field_validator` hooks, 19 enums/Literals, and 1 field with `ge`/`le`/pattern bounds [measured in `spike/findings-spike7.json` against `src/axolotl/utils/schemas/config.py` on the pinned digest; the file itself is 2,237 lines / 86.8 KB at `main`](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/schemas/config.py). The `axolotl config-schema` CLI exists specifically to dump that schema for programmatic use [docs.axolotl.ai — Home / AI Agent Support](https://docs.axolotl.ai/).

**Implication:** the maintainers intend the YAML to be the stable contract. The Python code is organized to validate that YAML, not to expose a `Trainer` class you import and subclass.

### 1.2 The Docker image is a first-class distribution artifact

The Docker docs define four variants (`axolotl-base`, `axolotl`, `axolotl-cloud`, `axolotl-cloud-term`) and a tag grammar that includes the pinned form this repo uses: `{branch}-{date_in_YYYYMMDD}-py{python}-cu{cuda}-{pytorch}` with examples `main-20260315-py3.12-cu130-2.12.0` and `main-latest` [docs.axolotl.ai — Docker](https://docs.axolotl.ai/docs/docker.html). The docs confirm the current runtime matrix is Python 3.12 / CUDA 13.0 / PyTorch 2.11.0 and 2.12.0, with `main-latest` pointing at `py3.12-cu130-2.12.0` — exactly the matrix in `trainer/Dockerfile:26` (`main-20260817-py3.12-cu130-2.12.0@sha256:29327…`, torch 2.12, CUDA 13.0, Python 3.12) and `trainer/README.md`.

Installation docs present Docker as the less-error-prone path: *"Installing with Docker can be less error prone than installing in your own environment"* with `docker run --gpus '"all"' … axolotlai/axolotl:main-latest` as the example [docs.axolotl.ai — Home / Installation](https://docs.axolotl.ai/). The repo's `pyproject.toml` / `uv` path exists for developers hacking on Axolotl itself; the platform path is the image. **Confidence: High.**

### 1.3 There is no stable "import axolotl and call train()" library API

The CLI reference's "Legacy CLI Usage" section is revealing: the old form is `python -m axolotl.cli.preprocess`, `python -m axolotl.cli.train`, `accelerate launch -m axolotl.cli.train config.yml` [docs.axolotl.ai — CLI / Legacy CLI Usage](https://docs.axolotl.ai/docs/cli.html). That is, even the Python path is a module CLI, not a library import. The `axolotl/` package under `src/axolotl/` is organized around CLI modules (`src/axolotl/cli/train.py`, `preprocess.py`, etc.), trainers that wrap HuggingFace TRL/transformers, and the schema validators — not around a `axolotl.train(config_dict)` function with SemVer guarantees. The docs never advertise a `import axolotl; axolotl.train(...)` API. Searching the docs for a programmatic API surfaces `API Reference` (auto-generated code docs) but no "Python SDK for training" guide — the guides all route through YAML.

**Assessment:** if you import Axolotl as a library you are coupling to internal module paths that the maintainers rename between releases. The YAML + CLI is the versioned interface; the Python modules are implementation detail. This repo's decision in `AGENTS.md` — *"Axolotl owns the training loop; we own the contract. Do not call TRL/PEFT directly — those APIs move underneath you (TRL 1.x dropped `warmup_ratio` …)"* — applies equally to calling Axolotl's Python internals directly. **Confidence: High.**

### 1.4 The version matrix is explicitly a maintainer-owned problem

The Docker docs state the image ships with a tested torch/CUDA/Python matrix; the support matrix documents `TRL v1`/`transformers 5.x` coupling and compatibility rules (e.g., precision × trainable params, "Requires" dependencies, "Incompatible" combinations) [docs.axolotl.ai — Support Matrix](https://docs.axolotl.ai/docs/support-matrix.html). The install docs still show `uv pip install torch==2.12.0` as the pinned torch for the current line [docs.axolotl.ai — Home / Installation](https://docs.axolotl.ai/). The repo's `pyproject.toml` pins `transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `datasets` as version ranges that the maintainers test together; Docker is where those ranges are resolved to exact wheels.

Spike 3 measured what happens when you ignore this and install `latest`: transformers 5.15.0 + trl 1.10.0 on torch 2.5.1 produced three chained failures (warmup_ratio TypeError, `save_safetensors` rejected, checkpoints as `.bin` that transformers 5.x then refused to load without torch >=2.6) [spike/README.md — Spike 3 results; trainer/Dockerfile:6-18; trainer/README.md — Why Axolotl is the base]. Spike 4 closed the loop by pinning to Axolotl's digest: torch 2.12 >> 2.6 floor, `warmup_ratio` accepted, resume from `checkpoint-2 -> step 6` in 54s, adapter as `safetensors` [spike/README.md — Spike 4 results; trainer/README.md — Verified 2026-08-18].

**Therefore:** the repo's tagline in `trainer/Dockerfile:12-18` — *"Axolotl's maintainers resolve and test it on every build … Delegating the matrix to people who test it is a better answer than pinning six packages by hand"* — is not rhetoric; it is the documented distribution model.

---

## 2. How This Repo Uses Axolotl

### 2.1 The seam

`trainer/Dockerfile` builds `FROM axolotlai/axolotl:main-20260817-py3.12-cu130-2.12.0@sha256:29327…` with `COPY entrypoint.py thinking.py /opt/trainer/` and `ENTRYPOINT ["python", "/opt/trainer/entrypoint.py"]`. No `RUN pip install` — a deliberate constraint stated in the Dockerfile and README: adding packages would reintroduce the resolution problem the base exists to avoid [trainer/Dockerfile:33-43; trainer/README.md — Pin discipline].

`trainer/entrypoint.py:build_config()` renders a dict of Axolotl keys and writes `CONFIG = /out/config.yaml` via `yaml.safe_dump(cfg, sort_keys=True)`, then invokes `axolotl train /out/config.yaml` via `subprocess.Popen` with `PYTHONUNBUFFERED=1`, `bufsize=0`, streaming both stdout and stderr line-by-line and splitting on `\r` as well as `\n` to handle tqdm progress bars [trainer/entrypoint.py:164-367; trainer/README.md — Contract; ADR-0001]. The contract is filesystem: `/job/job.json + /job/dataset.jsonl` in, `/out/config.yaml + /out/run/ + /out/result.json + /out/train.log` out, with `result.json` always written (including on failure) from a `try/finally` [trainer/entrypoint.py:320-396; trainer/README.md — Contract].

Key settings rendered (all via `AxolotlInputConfig` fields):

| Concern | Key(s) | Value | Source |
|---|---|---|---|
| Method | `adapter`, `load_in_4bit`, `bnb_4bit_quant_type`, `bnb_4bit_use_double_quant`, `bnb_4bit_compute_dtype` | `qlora`, `true`, `nf4`, `true`, `bfloat16` | [trainer/entrypoint.py:203-207](https://github.com/thp728/temper/blob/main/trainer/entrypoint.py#L203-L207); [docs.axolotl.ai — Support Matrix / Precision × trainable params](https://docs.axolotl.ai/docs/support-matrix.html) |
| Targets | `lora_target_linear` | `true` | [trainer/entrypoint.py:212](https://github.com/thp728/temper/blob/main/trainer/entrypoint.py#L212); [docs.axolotl.ai — Support Matrix / Fine-tuning strategies](https://docs.axolotl.ai/docs/support-matrix.html) |
| Precision | `bf16`, `fp16` | `true`, `false` | [trainer/entrypoint.py:215-216](https://github.com/thp728/temper/blob/main/trainer/entrypoint.py#L215-L216); [trainer/Dockerfile:20-22](https://github.com/thp728/temper/blob/main/trainer/Dockerfile#L20-L22) |
| Correctness | `chat_template`, `chat_template_kwargs`, `train_on_inputs`, `seed` | `tokenizer_default`, `{"enable_thinking": …}`, `false`, `42` | [trainer/entrypoint.py:230-238](https://github.com/thp728/temper/blob/main/trainer/entrypoint.py#L230-L238); [docs.axolotl.ai — Config Reference](https://docs.axolotl.ai/docs/config-reference.html) |
| Packing | `sample_packing` | `false` | [trainer/entrypoint.py:242](https://github.com/thp728/temper/blob/main/trainer/entrypoint.py#L242); [docs.axolotl.ai — Multipack](https://docs.axolotl.ai/docs/multipack.html) |
| Throughput | `gradient_checkpointing`, `flash_attention` | `true`, `true` | [trainer/entrypoint.py:218-219](https://github.com/thp728/temper/blob/main/trainer/entrypoint.py#L218-L219) |
| Eval | `val_set_size` | `0.05` default | [trainer/entrypoint.py:61](https://github.com/thp728/temper/blob/main/trainer/entrypoint.py#L61) |
| Checkpoint | `save_safetensors`, `save_total_limit` | `true`, `3` | [trainer/entrypoint.py:245-246](https://github.com/thp728/temper/blob/main/trainer/entrypoint.py#L245-L246) |
| LoRA | `lora_r`, `lora_alpha`, `lora_dropout`, `lora_use_rslora` | `16`, `32 (=2r)`, `0.0`, `r >= 32` inferred | [trainer/entrypoint.py:50-53,185-190](https://github.com/thp728/temper/blob/main/trainer/entrypoint.py#L50-L53) |

`api/orchestrator.py` tars `trainer/` and ships it to a GPU VM, builds the image there (`docker build --progress=plain`), runs it with `docker run --rm --gpus all -v /tmp/job:/job:ro -v /tmp/out:/out`, streams stdout over SSH (the event channel — ADR-0001), parses `---RESULT---` + `result.json`, fetches the adapter, and always destroys the VM in `finally` with `list`-based confirmation [api/orchestrator.py:60-222; ADR-0001; spike/README.md — Spike 4].

Spikes 3/4/6/7 validated the matrix, resume, FSDP, and schema: Spike 7 introspected the pinned digest to count 344 non-infrastructure fields, 113 `model_validator` hooks, and 1 field with schema bounds [spike/findings-spike7.json; spike/README.md — Spike 7].

### 2.2 What is deliberately *not* exposed

`ALLOWED_OVERRIDES` is `{lora_r, lora_alpha, learning_rate, num_epochs, max_steps, sequence_len, micro_batch_size, gradient_accumulation_steps, val_set_size, save_steps}`; unknown top-level keys and unknown `hyperparameters` keys are both refused loudly and echoed as `rejected_overrides` [trainer/entrypoint.py:63-77,169-181]. `DEFAULTS` hard-codes `warmup_ratio: 0.1`, `lr_scheduler: cosine`, and the correctness settings above; Advanced mode is specified to expose them with failure modes named inline, with overrides recorded in the run spec, and with an export-time template probe to catch wrong overrides before the user does [AGENTS.md — Design rules; trainer/README.md — What is settable; spike/README.md — Spike 7 classification sample].

Thinking mode is detected from the dataset via `thinking.py:detect()` and applied as `chat_template_kwargs: {enable_thinking: …}` identically at training and serving; mixed datasets block with line-numbered errors [trainer/thinking.py; trainer/entrypoint.py:233-235].

---

## 3. Correctness Assessment

### 3.1 What's right (defensible under grilling)

**1. YAML + `axolotl train` is the strongest seam for a platform.**

Every primary source routes through it. The quickstart, CLI reference, and config reference all treat YAML as the artifact that controls dataset preprocessing, training, evaluation, quantization, and inference [docs.axolotl.ai — Quickstart](https://docs.axolotl.ai/docs/getting-started.html); [docs.axolotl.ai — CLI](https://docs.axolotl.ai/docs/cli.html); [docs.axolotl.ai — Config Reference](https://docs.axolotl.ai/docs/config-reference.html). The CLI's `--launcher torchrun` / `--launcher accelerate` separator and `--resume-from-checkpoint` flag are documented as CLI flags on top of the YAML, not as replacements for it [docs.axolotl.ai — CLI / train](https://docs.axolotl.ai/docs/cli.html). Rendering YAML and invoking the CLI keeps the training surface declarative, auditable (`/out/config.yaml` ships with every run), and replayable — a later debugger can re-run `axolotl train /out/config.yaml` verbatim on the same digest.

**2. Digest-pinned Docker image delegating the version matrix is correct.**

The Docker docs define the matrix this repo pins (Python 3.12 / CUDA 13.0 / PyTorch 2.12.0) and distinguish floating tags (`main-latest`) from dated nightly tags (`main-20260817-…`) [docs.axolotl.ai — Docker / Tags format](https://docs.axolotl.ai/docs/docker.html). The repo pins the nightly tag by digest, not the floating tag, and labels the tag as a comment: *"The digest is the contract. The tag is a comment. Changing either is a deliberate act that re-runs the GPU smoke test"* [trainer/Dockerfile:25-26; trainer/README.md — Pin discipline]. This is strictly stronger than pinning six pip packages: a six-package pin still leaves CUDA/driver/kernel compatibility untested, whereas the image is the artifact Axolotl's own CI builds and tests (the repo's `docker-e2e.yml`, `tests-nightly.yml`, `multi-gpu-e2e.yml` workflows validate it).

Spike 3 is the counterfactual that proves the point: installing `latest` gave transformers 5.15.0 + trl 1.10.0 + peft 0.20.0 + bitsandbytes 0.50.1 on torch 2.5.1, and three things chained — `warmup_ratio` TypeError (TRL 1.x SFTConfig no longer inherits TrainingArguments), `save_safetensors` rejected, checkpoints as `.bin`, transformers 5.x refusing to load them without torch >=2.6 [spike/README.md — Spike 3]. Pinning the image removed all three.

**3. Not calling TRL/PEFT directly is correct.**

TRL's own docs show a moving surface: the library shipped a `v1` post-training rewrite in 2026 with unified `SFTConfig`/`DPOConfig` and new flags, and the HuggingFace blog marks it as a breaking surface [huggingface.co — TRL docs, "TRL v1: Post-Training Library…" Mar 2026](https://huggingface.co/docs/trl/index). Axolotl's config surface still exposes `warmup_ratio`, so the intended defaults remain expressible regardless of what TRL does to its Python API — exactly the argument in `trainer/Dockerfile:14-18`. The orchestrator stays orchestrator-agnostic (thread today, Temporal tomorrow) because it only knows about a job record and a filesystem contract, not about a PEFT `LoraConfig` object.

**4. Correctness defaults are correctly locked.**

`chat_template: tokenizer_default` ("Uses the chat template that is available in the `tokenizer_config.json`. If the chat template is not available … it will raise an error. This is the default" — Config Reference) [docs.axolotl.ai — Config Reference / datasets.chat_template](https://docs.axolotl.ai/docs/config-reference.html) prevents the modal silent-failure (train/infer template mismatch) that Report A ranks as the highest-frequency platform bug. `train_on_inputs: false` prevents the loss-masking mismatch. NF4 double-quant + `bfloat16` matches Axolotl's QLoRA support matrix row (`adapter: qlora` requires `load_in_4bit: true`; load-time QLoRA is frozen 4-bit base + bf16 adapter; `fp8:` compute is separate) [docs.axolotl.ai — Support Matrix / Precision × trainable params](https://docs.axolotl.ai/docs/support-matrix.html). `lora_target_linear: true` ("*The single most important LoRA setting after rank — earlier guidance recommended q_proj/v_proj only, but … including all linear layers consistently produces better results*" — Report B) matches the support-matrix's `lora_target_linear` description. `save_safetensors: true` avoids the `.bin` chain Spike 3 hit. `bf16: true`, `seed: 42`, `gradient_checkpointing: true`, `flash_attention: true` are all supported and documented as safe defaults for Ampere-or-newer (L4 qualifies) [docs.axolotl.ai — Support Matrix / Attention backends](https://docs.axolotl.ai/docs/support-matrix.html).

**5. Thinking-mode detection is correct and complete for the catalog.**

The Qwen3 model guides show `chat_template_kwargs` threading through the Jinja template (e.g., Qwen3's `enable_thinking` flag that gates `<think>` blocks) [docs.axolotl.ai — Model Guides / Qwen3](https://docs.axolotl.ai/docs/models/qwen3.html). The platform detects it from the dataset, applies the same value at training and serving, and blocks mixed datasets — exactly the "spine principle" (template at stage 5 must be the same object serialized into the artifact at stage 11) from Report A.

**6. The `/job -> /out` contract with always-written `result.json` is correct.**

The `try/finally` that writes `result.json` including on failure means the orchestrator never parses logs to learn what happened [trainer/entrypoint.py:393-396]. The streaming relay (`PYTHONUNBUFFERED=1`, `bufsize=0`, `flush=True`, `\r` splitting) is load-bearing for tqdm and is documented as such [trainer/entrypoint.py:98-161; ADR-0001].

### 3.2 What's risky (grillable, not wrong)

**1. The config schema is wide and mostly opaque to static validation.**

Spike 7 measured 388 total fields, 344 after excluding infrastructure, 113 `model_validator` hooks, 27 `field_validator` hooks, 19 enums, 1 field with `ge`/`le`/pattern bounds — so ~12% of constraints are visible to a schema reader [spike/findings-spike7.json; spike/README.md — Spike 7]. A `model_validator` is arbitrary Python, so a form generated from the schema cannot know what will be refused until the job is already on a billing VM. The repo's mitigations are sound — `axolotl config-schema` to generate field lists, hand-written cross-field rules, loud rejection of unknown keys, and a 20-field random sample classifying 65% as `known_but_unsupported` (DPO, GaLore, LISA need one-line block reasons, not design) — but the residual risk is that a valid-looking `max_steps` + `save_steps` + `val_set_size` combination fails four minutes into a paid job. **Recommendation:** keep Advanced mode behind a gate, and run `axolotl preprocess --debug` or a dry-run `axolotl train --help`-style validation pass before provisioning the VM where possible (Axolotl's `preprocess` command loads the config and validates datasets without training).

**2. `main-20260817` is a `main`-branch nightly, not a semantic release.**

The Docker tag grammar distinguishes `main-20260817-py3.12-cu130-2.12.0` (nightly, branch + date) from `{version}` / `{version}-py3.12-cu130-2.12.0` (e.g., `0.16.1`) [docs.axolotl.ai — Docker / Main / Tags format](https://docs.axolotl.ai/docs/docker.html). Nightly tags move with `main`; release tags are cut from version branches. Digest pinning eliminates runtime drift, but the *next* pin is still a choice: bump to another nightly (fresh kernels, fresh bugs) or to a release tag (slower-moving, curated). Without a policy ("pin to release tags; only use nightly for a named reason; each pin re-runs the GPU smoke test"), the repo will either stagnate on the current nightly or churn nightly. **Recommendation:** adopt release tags for the default line; reserve nightly bumps for model-support reasons (e.g., needing Qwen3.5 MoE or Gemma 4 on day 1). Record the reason per bump in an ADR.

**3. No `pip install` on top of the image is a strong rule that needs an escape hatch for security patches.**

The Dockerfile's ban — *"Never `pip install` inside it — that reintroduces the dependency-resolution problem the pinned base exists to avoid"* [trainer/Dockerfile:33-36] — is correct for the base dependencies. But a CVE in `transformers` between pins still needs a path. **Recommendation:** allow a pinned `requirements-patch.txt` with exact `==` versions and hashes, reviewed as a code change that re-runs the smoke test, rather than an unpinned `RUN pip install`.

### 3.3 What's missing (not bugs, but gaps an evaluator will ask about)

**1. `sample_packing: false` is conservative and correct; the upside is large and the enablement condition is precise.**

The Multipack docs visualize packed batches and state: *"we only need to concatenate the sequences … and let flash attention know where each new sequence begins"* via `cu_seqlens` [docs.axolotl.ai — Multipack](https://docs.axolotl.ai/docs/multipack.html). The support matrix requires *"Sample packing: varlen backend (FA2/3, flex, xformers, sage)"* and marks *"Sample packing × any `rl:`"* as incompatible [docs.axolotl.ai — Support Matrix / Compatibility rules](https://docs.axolotl.ai/docs/support-matrix.html). The repo leaves packing off *"until measured per model"* with a per-model varlen-attention gate [trainer/entrypoint.py:241-242; trainer/README.md — Still open]. For SFT on short chat turns this is a 3–5× throughput win. Spike 6's FSDP loss collapse (`12.16 → 0 → 0`, `grad_norm: nan` on every step) was hypothesized to involve bf16 + FSDP2 + gradient checkpointing or packing-like contamination — so packing must be validated with loss curves, not just throughput. **Not enabling it is defensible; never planning to enable it is not.**

**2. FSDP/DeepSpeed is validated but not wired through the job spec.**

The multi-GPU docs define three mutually exclusive strategies: DeepSpeed (`deepspeed: deepspeed_configs/zero3.json`), FSDP (`fsdp_version: 2` + `fsdp_config: {…}`), and DDP (default) [docs.axolotl.ai — Multi-GPU](https://docs.axolotl.ai/docs/multi-gpu.html). Spike 6 proved 2× L4: the image sees both devices, FSDP FULL_SHARD shards and steps, `.distcp` checkpoint is written, sharded resume works — but loss collapses with `nan` grad_norm, so the mechanism runs and the numerics do not [spike/README.md — Spike 6; spike/findings-spike6.json]. The entrypoint has no `fsdp_version`/`deepspeed` rendering, and `api/orchestrator.py` hard-codes `GPU_PREFERENCE` for single-GPU; multi-GPU is correctly described as *"provisioning, not architecture"* with `num_gpus` as a create parameter, but the YAML path stays single-GPU. An evaluator will ask: *"How does a 70B LoRA fit on one 24 GB card, and when does it spill to FSDP?"* The honest answer is: single-GPU QLoRA fits 7–8B on 24 GB and 70B NF4 on 80 GB (QLoRA paper, Support Matrix); 70B full fine-tuning needs FSDP — which is proven at the provider level but not at the numerics level (Issue #81). **Recommendation:** keep v1 single-GPU, but add a `num_gpus` job field that threads through to `fsdp_version: 2` + `fsdp_config` when needed, gated on the Spike 6 numerics fix.

**3. W&B / experiment tracking, evaluation, and merge are present in Axolotl but unused here.**

The CLI documents `axolotl evaluate`, `axolotl lm-eval`, `axolotl merge-lora`, and `axolotl quantize` [docs.axolotl.ai — CLI / Command Reference](https://docs.axolotl.ai/docs/cli.html); the support matrix lists W&B, MLflow, Comet, TensorBoard as experiment tracking [docs.axolotl.ai — Support Matrix / Experiment tracking](https://docs.axolotl.ai/docs/support-matrix.html). The repo streams loss via SSH stdout and classifies metric vs log events in the orchestrator, but does not set `wandb_project`/`wandb_entity`, does not run `axolotl evaluate` post-training, and does not merge the adapter (it ships the adapter + `adapter_config.json` only). For a take-home this is correct scope ("auth and billing are the only sanctioned gaps; every other flow is meant to be complete" — but deployment is explicitly Docker Compose, not hosted W&B). For a production platform, the missing pieces are: `wandb_project` per tenant (the support matrix's experiment-tracking row), `do_causal_lm_eval` / `lm_eval_tasks` for eval loss, and `axolotl merge-lora` for users who want a merged checkpoint rather than an adapter. **None of these are misuses of Axolotl; they are unused surfaces.**

**4. Quantization correctness is correctly implemented; the merge nuance deserves a sentence in `trainer/README.md`.**

The support matrix is precise: load-time QLoRA is `load_in_4bit: true` + frozen 4-bit base + bf16 adapter; train-time FP8 is `fp8: true` (mixed-precision compute, master weights stay bf16/fp32, full fine-tune works); post-training quantization is `axolotl quantize` PTQ [docs.axolotl.ai — Support Matrix / Quantization & precision](https://docs.axolotl.ai/docs/support-matrix.html). The entrypoint's `bnb_4bit_compute_dtype: bfloat16` is correct per that matrix. The README already notes adapters ship as fp32 (504 tensors, every one `F32`, 132.2 MB for Qwen3-4B at r=16) and that bf16 is the compute dtype [trainer/README.md — Still open]. The nuance to add: QLoRA adapters train in fp32 under 4-bit quantisation (standard), and merging should target a bf16/fp16 base before any PTQ export — the support matrix warns that `quantized weights ⇒ frozen base ⇒ adapter-only` and that GGUF/K-quants are not trainable.

**5. The export-time template probe is specified but not yet implemented.**

Report A's spine principle — the chat template resolved at stage 5 must be the same object serialized into the artifact at stage 11 — is correct and load-bearing. The repo records that the probe ("re-tokenise a fixed conversation through both the training template and the artifact's, assert identical ids") is the check that makes Advanced-mode exposure safe [AGENTS.md — Design rules; trainer/README.md — Still open]. An evaluator will ask to see it run; today it is a TODO.

---

## 4. Alternatives Comparison (Primary-Source Ground Truth)

### 4.1 Direct TRL/PEFT (HuggingFace stack)

**What it is:** `TRL` ([huggingface.co — TRL docs](https://huggingface.co/docs/trl/index): `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, etc.) + `PEFT` ([huggingface.co — PEFT docs](https://huggingface.co/docs/peft/index): `LoraConfig`, `get_peft_model`, NF4 via `bitsandbytes`) + `transformers` + `accelerate` (launcher for DDP/FSDP/DeepSpeed). You write the training script as Python, import the trainers, and own the dataset tokenization, chat-template application, and collator.

**When it is the right choice:** you need a training objective Axolotl does not expose, you are doing RL (GRPO/PPO) with custom reward functions that Axolotl's `trl:` block does not cover, or you are pretraining with streaming datasets and custom data pipelines. TRL is the ground-truth trainer Axolotl itself wraps.

**Tradeoffs vs this repo's Axolotl path:**

| Axis | Direct TRL/PEFT | This repo (Axolotl) |
|---|---|---|
| **Version matrix** | You pin `torch`, `transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `cuda` yourself. The failure mode is Spike 3: TRL 1.x removed `warmup_ratio` from `SFTConfig` (it no longer inherits `TrainingArguments` the same way), so a config that worked last month raises `TypeError`. You fix it by editing code. | Delegated to Axolotl's Docker image. `warmup_ratio` remains expressible via Axolotl's config even when TRL's Python API moves. You fix it by bumping a digest. |
| **Config surface** | Python kwargs with IDE completion; breaks on API renames. | YAML validated by `AxolotlInputConfig` (Pydantic). Renames are migrations in Axolotl, not in your code. 344-field surface, but your platform only renders ~20 keys. |
| **Docker** | You build your own base from `nvidia/cuda` or `pytorch/pytorch`; you resolve the matrix. | You build `FROM axolotlai/axolotl@sha256:…`; you inherit the tested matrix. 8.5 GB, ~3 min cold pull at 46 MB/s on JarvisLabs [spike/README.md — Spike 2]. |
| **Maintenance** | You track TRL/transformers release notes and migrate call sites. For a small team this is the highest ongoing cost. | You track Axolotl releases and the support matrix's compatibility rules. Axolotl tracks TRL/transformers for you. |
| **Flexibility ceiling** | Higher. Any HuggingFace `Trainer` callback or custom loop is possible. | Lower. Anything not in `AxolotlInputConfig` is not expressible without forking Axolotl. But the support matrix is wide: SFT/LoRA/QLoRA/DoRA/rsLoRA/DPO/KTO/ORPO/GRPO/Reward Model/QAT/FP8/TP/CP/EP are all covered [docs.axolotl.ai — Support Matrix](https://docs.axolotl.ai/docs/support-matrix.html). |

**Verdict for this platform:** direct TRL/PEFT would be more code to maintain for the same outcome at the SFT+QLoRA slice this platform ships. Axolotl's abstraction pays for itself precisely because the team is small.

### 4.2 Unsloth

**What it is:** a kernel-focused fine-tuning library that patches HuggingFace models with hand-written Triton kernels to accelerate LoRA/QLoRA training. Project repo at [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth); docs hub at [docs.unsloth.ai](https://docs.unsloth.ai/) (hosted on GitBook at `unsloth.ai/docs/...`). The community summary is often quoted as "2× faster, 70% less VRAM" — the primary source for that claim is the project's own docs and README, not an independent benchmark.

Primary-source evidence reviewed: the docs home claims accelerated training via Triton kernels; the fine-tuning guide walks through LoRA/QLoRA setup with `unsloth` as a drop-in that patches `transformers` models at load time [docs.unsloth.ai — Fine-tuning LLMs Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide). The GitHub README's feature list and benchmarks are the source for the "2× faster / 70% less VRAM" shorthand; treat those numbers as **project-reported, workload-dependent**, not as cross-platform guarantees.

**Tradeoffs vs Axolotl for a commercial platform:**

| Axis | Unsloth | Axolotl (this repo) |
|---|---|---|
| **Speed/VRAM** | Faster on supported models (Triton-fused attention, RoPE, MLP, cross-entropy). The win is real for single-GPU ≤14B where kernel time dominates. | Baseline HuggingFace throughput. Liger kernel + Cut Cross Entropy + Flash Attention 2/3 are available as opt-in plugins that close part of the gap [docs.axolotl.ai — Support Matrix / Performance integrations](https://docs.axolotl.ai/docs/support-matrix.html), but Axolotl does not ship fused Triton kernels at Unsloth's depth. |
| **Model coverage** | Supports a curated set of popular models (Llama, Qwen, Mistral, Gemma, Phi) with patchable architectures. New or niche architectures (MoE, hybrid SSM, VLMs) are added on Unsloth's schedule. | Any HuggingFace causal/seq2seq LM trains out of the box; first-class patches are listed per family but generic support is universal [docs.axolotl.ai — Support Matrix / Model architectures](https://docs.axolotl.ai/docs/support-matrix.html). The docs list Qwen3-MoE, DeepSeek-V4, GLM-4.7, BitNet, multimodal VLMs, and 69 example configs under `examples/` — a broader tail. |
| **Integration seam** | Python library: `from unsloth import FastLanguageModel` patches the model before you construct a `Trainer`. You own the Python script and the version matrix (transformers + peft + trl + torch + Unsloth's Triton kernels). Unsloth "requires specific PEFT versions — always install Unsloth's recommended dependency versions" is a recurring note in community docs and mirrors Axolotl's version-matrix warning. | YAML + CLI + Docker. You do not import the trainer; you render YAML and invoke the CLI. The matrix is owned by the image. |
| **Operational fit** | Optimizes the *training step*, not the *training platform*. You still need orchestration (provisioning, streaming, checkpointing, adapter collection). Composing Unsloth's kernels with FSDP/DeepSpeed and with Axolotl's multi-GPU paths is not trivial. | Optimizes the *training platform*: declarative config, built-in FSDP/DeepSpeed/TP/CP/EP options, deepspeed_configs fetched via `axolotl fetch deepspeed_configs` [docs.axolotl.ai — CLI / fetch](https://docs.axolotl.ai/docs/cli.html), and a Docker distribution model. |
| **Maintenance** | You track Unsloth releases for kernel compatibility with each new model family, plus transformers/peft compatibility. The patch must land before the model does (Unsloth's blog tracks this explicitly). | You track Axolotl releases; kernel work is via Liger/CutCE/FlashAttention plugins that Axolotl integrates downstream. |

**When to use Unsloth in this platform:** as a *conditional accelerator*, not as the primary trainer. If a job is single-GPU ≤14B on a supported architecture and VRAM or wall-clock is the binding constraint, switching the launch to an Unsloth-patched path can cut cost meaningfully. The platform should keep Axolotl as the default (broader model coverage, cleaner version-matrix story) and offer Unsloth as an advanced toggle for the subset where it wins. LLaMA-Factory already does this: it exposes `use_unsloth: true` as a training switch that achieves "170% speed" in its benchmark [github.com/hiyouga/LLaMA-Factory — README / Changelog 2023-12-23](https://github.com/hiyouga/LLaMA-Factory).

### 4.3 LLaMA-Factory

**What it is:** a self-hostable fine-tuning toolkit with a Gradio Web UI ("LlamaBoard"), CLI, Docker images, and broad method support: pre-training, SFT, reward modeling, PPO, DPO, KTO, ORPO, LoRA/QLoRA/DoRA/rsLoRA/PiSSA/OFT, and many datasets. Repo at [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory); cited here at 12,000+ stars as of 2026, with enterprise usage noted.

**Tradeoffs vs Axolotl:**

| Axis | LLaMA-Factory | Axolotl (this repo) |
|---|---|---|
| **Scope** | Full toolkit: training + Web UI + inference + evaluation + deployment (vLLM/SGLang). A good starting point if you want a built-in UI to fork. | Training framework only. The platform builds its own UI and orchestration around it. |
| **Config surface** | Python + YAML + Web UI forms; 344-field surface is similarly wide. Web UI is convenient for single-user but couples the training platform to Gradio's lifecycle. | YAML + CLI; no Web UI. The repo builds a FastAPI + plain HTML control plane and plans Next.js/shadcn/ui for Phase B. |
| **Docker** | Ships its own Docker images with a similar `FROM nvidia/cuda` base and its own version matrix. You pin *that* image instead of Axolotl's. | Ships Axolotl images as above. The choice is which matrix you delegate to. |
| **Model/method breadth** | Very wide (100+ models, many preference methods). The changelog shows day-0 support for new releases. | Wide (support matrix lists SFT/DPO/KTO/ORPO/GRPO/Reward Model/QAT/FP8/MoE, 69 example configs). The difference is Web UI vs no Web UI, not model coverage. |
| **Composability** | Harder to embed as a subprocess with a clean `/job -> /out` filesystem contract; the Web UI assumes it owns the training process. | Designed for the filesystem contract this repo uses; the container needs no inbound port and the orchestrator owns the lifecycle. |

**Verdict for this platform:** Axolotl is the better default for a commercial platform that builds its *own* UI and orchestration (which this repo does). LLaMA-Factory is the better default if you want to ship a self-host Web UI with minimal platform code — i.e., if the product *is* the UI, not the orchestration layer. The research correctly notes LLaMA-Factory as the self-host benchmark (Report C) but recommends Axolotl when you don't want Gradio in production — consistent with the primary evidence.

---

## 5. Detailed Recommendations (What to Change, What to Keep, What to Watch)

### 5.1 Keep (do not relitigate)

1. **Keep `axolotl train` via YAML as the integration seam.** Do not migrate to importing Axolotl's Python modules or to calling TRL/PEFT directly. The CLI is the documented entry point [docs.axolotl.ai — CLI](https://docs.axolotl.ai/docs/cli.html); the YAML is the validated artifact [docs.axolotl.ai — Config Reference](https://docs.axolotl.ai/docs/config-reference.html); and the version-matrix delegation is proven by Spike 3/4.

2. **Keep digest pinning, no `pip install` on top of the image.** The Dockerfile's rule is load-bearing. The tag is a comment; the digest is the contract [trainer/Dockerfile:25-32; trainer/README.md — Pin discipline]. Changing the tag without changing the digest, or vice versa, is a deliberate act that re-runs the GPU smoke test — preserve that discipline.

3. **Keep `ALLOWED_OVERRIDES` narrow and loud.** Refusing unknown keys at both the top level and inside `hyperparameters`, echoing them as `rejected_overrides`, prevents the silent-override bug where a caller believes `maxSteps` is in effect but isn't [trainer/entrypoint.py:169-181]. Report C calls this "absence disqualifies" — every credible competitor does line-level validation; this is the adapter-level equivalent.

4. **Keep thinking-mode detection and the always-written `result.json`.** Both are correctness properties that distinguish this platform from a thin wrapper. The thinking-mode line-numbered block on mixed datasets is the right failure mode for an ambiguous input [trainer/thinking.py].

### 5.2 Change (small, targeted)

1. **Adopt a digest-update policy as an ADR.** Today's pin is `main-20260817-py3.12-cu130-2.12.0@sha256:29327…` (a `main`-branch nightly). Document: (a) the default is a semantic release tag (`0.16.x` line) [docs.axolotl.ai — Docker / Main / Tags format](https://docs.axolotl.ai/docs/docker.html); (b) nightly bumps require a named reason (model support, CVE) and re-run the GPU smoke test; (c) each bump records the diff in `docs/adr/` with the `findings.json` hash that proves the run.

2. **Add a pre-provision validation gate that validates the rendered YAML without a GPU.** `axolotl preprocess config.yml --debug` loads the config through `AxolotlInputConfig` and validates datasets, catching `model_validator` rejections before a VM exists [docs.axolotl.ai — CLI / preprocess](https://docs.axolotl.ai/docs/cli.html). Running this locally (or in a CPU container pre-flight) would catch the ~88% of constraints that live in hooks before billing starts. Spike 7's finding that only 12% of constraints are schema-visible makes this the highest-leverage pre-flight you can add.

3. **Wire `sample_packing` behind a per-model varlen gate, not a global toggle.** The Multipack docs require a varlen backend (FA2/3, flex, xformers, sage) and the support matrix marks `Sample packing × any rl:` as incompatible [docs.axolotl.ai — Multipack](https://docs.axolotl.ai/docs/multipack.html); [docs.axolotl.ai — Support Matrix / Compatibility](https://docs.axolotl.ai/docs/support-matrix.html). Enable packing for `qwen2/3` and `llama3` families where FA2 is the documented attention backend, with an eval harness that asserts no cross-example contamination (loss on packed vs unpacked must match within tolerance). The throughput win (3–5× on short chat turns) directly lowers the platform's cost estimate.

4. **Implement the export-time template probe.** Specified in `AGENTS.md` and `trainer/README.md` but still open: re-tokenise a fixed probe conversation through both the training-time Jinja template and the artifact's serialized `tokenizer_config.json:chat_template`, assert byte-for-byte `input_ids` equality. This is what makes Advanced-mode exposure of `chat_template` safe; without it, exposing the knob violates the report's own spine principle.

5. **Add minimal experiment tracking (`wandb_project`) and post-training eval (`axolotl evaluate` / `axolotl lm-eval`).** Both are opt-in YAML keys already supported [docs.axolotl.ai — Config Reference](https://docs.axolotl.ai/docs/config-reference.html); [docs.axolotl.ai — CLI / evaluate, lm-eval](https://docs.axolotl.ai/docs/cli.html). For a production platform they are not optional: a user deciding whether to deploy needs eval loss, not just train loss. Scope them as tenant-scoped W&B projects and a compact lm-eval slice (not full HELM).

### 5.3 Watch (do not build now, but know the trigger)

1. **Multi-GPU via `fsdp_version: 2` + `fsdp_config`.** The multi-GPU docs define the migration from FSDP1 to FSDP2 and the `fsdp_config` keys (`offload_params`, `cpu_ram_efficient_loading`, `auto_wrap_policy`, `state_dict_type`, `reshard_after_forward`) [docs.axolotl.ai — Multi-GPU](https://docs.axolotl.ai/docs/multi-gpu.html). The provider already supports `num_gpus=8` on VM creation [spike/findings-spike8.json]. The blocker is Spike 6's numerics (`nan` grad_norm with bf16 + FSDP2 + gradient checkpointing on a synthetic 28-trainable-tokens/step dataset) — tracked as Issue #81. When that is resolved, the job spec should expose `num_gpus` and render FSDP2 YAML; until then, keep v1 single-GPU and state the scope explicitly in user-facing docs.

2. **Unsloth as a per-job accelerator.** When single-GPU QLoRA wall-clock is the cost bottleneck and the model is in Unsloth's supported set, a second image variant (`FROM unsloth/unsloth:*` or a patched Axolotl image) can be offered behind an `accelerator: unsloth` job flag. Do not make it the default — the model-coverage and version-matrix arguments above dominate for a platform.

3. **Liger kernel / Cut Cross Entropy as Axolotl plugins.** If Axolotl-side throughput is needed without adopting Unsloth, the support matrix's fused-kernel rows (`liger_rope`, `liger_cross_entropy`, `cut_cross_entropy: true`) are the lower-risk path [docs.axolotl.ai — Support Matrix / Performance integrations](https://docs.axolotl.ai/docs/support-matrix.html). They are single-plugin toggles, not a second framework.

---

## 6. What Would Get Challenged Under Grilling (Outsider Perspective)

> **"Why not just call TRL directly? Axolotl is a wrapper."**

**Answer:** TRL *is* the wrapper — Axolotl wraps TRL/transformers/peft/accelerate exactly so you don't chase their Python API renames. TRL 1.x removed `warmup_ratio` from `SFTConfig`'s surface; Axolotl's YAML still exposes it. Owning the matrix (six packages, each moving independently) is not simpler than delegating it to a team whose CI tests the matrix on every Docker build — Spike 3 proved the cost of owning it is measured in impossible checkpoint resumes, not in YAML verbosity. If you call TRL directly you still need a Docker image that pins the same six packages; you've just moved the pin from Axolotl's Dockerfile to your own.

> **"Pinning a `main` nightly by digest is fragile."**

**Answer:** Correct — `main-20260817` is a nightly, not a release. The digest makes it immutable but the *next* pin still requires a decision. The fix is a policy (Recommendation 5.2.1): default to release tags (`0.16.x`) and only bump to nightly for a named reason, with a GPU smoke test per bump. The fragility is in the tag choice, not in the pin discipline.

> **"Subprocess + YAML is slower and harder to debug than `import axolotl`."**

**Answer:** The overhead is one YAML write and one subprocess fork; the training itself is 120s–160s for 64 rows × 3 epochs [spike/README.md — Spike 4; trainer/README.md — Verified 2026-08-19]. The debuggability claim inverts: a rendered `/out/config.yaml` is replayable on any machine with the same digest, whereas an in-process call stack is not. The streaming relay (`iter_output_lines` splitting on `\r`, `PYTHONUNBUFFERED=1`) exists precisely so the framework's progress is not harder to debug via the subprocess than in-process.

> **"You disabled sample packing — you're leaving performance on the floor."**

**Answer:** Yes, 3–5× throughput on short chat datasets. Packing is off until the per-model varlen gate is validated, because without a varlen attention backend tokens in example 2 attend to example 1 (contamination). The support matrix makes this explicit [docs.axolotl.ai — Support Matrix](https://docs.axolotl.ai/docs/support-matrix.html); Spike 6's loss collapse is a warning that throughput without correctness is a worse failure than slower training. Packing is the next performance knob, not a missing default.

> **"Why not Unsloth? It's faster."**

**Answer:** Unsloth *is* faster on the models it supports — that win is project-reported and workload-dependent, and it comes via Triton kernels that patch HuggingFace models at load time [docs.unsloth.ai](https://docs.unsloth.ai/); [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth). For a platform whose tail is "anything HuggingFace with a chat template" (Qwen3-MoE, DeepSeek-V4, GLM-4.7, VLMs, BitNet), broader model coverage and a cleaner version-matrix story dominate over single-GPU kernel speed. Unsloth is a conditional accelerator to adopt per-job, not a replacement default — exactly how LLaMA-Factory already exposes it (`use_unsloth: true`) [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

> **"How is this different from just shipping a Dockerfile and calling it a platform?"**

**Answer:** The platform value is not the training loop — Axolotl owns that, by design. The value is the contract around it: pure validation functions, loud rejection of unknown keys at both levels, thinking-mode detection with line-numbered blocks, streaming event channel over SSH with stall/duration guards (ADR-0001/0002), always-written `result.json`, `finally`-block VM teardown with `list`-based confirmation, and adapter hash verification end-to-end. Those are not Axolotl features; they are the orchestration layer that is the work sample for a GPU-infrastructure company.

---

## 7. Primary Sources Cited

### Axolotl (framework of record)

- [docs.axolotl.ai — Home (overview, Easy Configuration, Docker less-error-prone)](https://docs.axolotl.ai/)
- [docs.axolotl.ai — Quickstart (first fine-tune is `axolotl train YAML`)](https://docs.axolotl.ai/docs/getting-started.html)
- [docs.axolotl.ai — CLI (grammar `axolotl <command> [config.yml]`, launcher separator, legacy `python -m axolotl.cli.*`)](https://docs.axolotl.ai/docs/cli.html)
- [docs.axolotl.ai — Config Reference (field list: `chat_template`, `train_on_inputs`, `sample_packing`, `bf16`, `wandb_project`, `fsdp_version`, etc.)](https://docs.axolotl.ai/docs/config-reference.html)
- [docs.axolotl.ai — Docker (variants, tag grammar `main-{date}-py{py}-cu{cuda}-{torch}` vs `{version}` releases, matrix Python 3.12 / CUDA 13.0 / PyTorch 2.12.0)](https://docs.axolotl.ai/docs/docker.html)
- [docs.axolotl.ai — Multipack / Sample Packing (cu_seqlens, flash attention, 4d mask fallback)](https://docs.axolotl.ai/docs/multipack.html)
- [docs.axolotl.ai — Multi-GPU (DeepSpeed vs FSDP vs DDP, FSDP1→FSDP2 migration table)](https://docs.axolotl.ai/docs/multi-gpu.html)
- [docs.axolotl.ai — Support Matrix (precision × trainable params, requires/incompatible, experiment tracking, maturity)](https://docs.axolotl.ai/docs/support-matrix.html)
- [docs.axolotl.ai — Model Guides / Qwen3 (`chat_template_kwargs`, `enable_thinking`)](https://docs.axolotl.ai/docs/models/qwen3.html)
- [github.com/axolotl-ai-cloud/axolotl — README (features, docker run example)](https://github.com/axolotl-ai-cloud/axolotl)
- [github.com/axolotl-ai-cloud/axolotl — `src/axolotl/utils/schemas/config.py` (2,237 lines; `AxolotlInputConfig` Pydantic model, validators)](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/schemas/config.py)
- [github.com/axolotl-ai-cloud/axolotl — tagged releases vs `main` branch commits](https://github.com/axolotl-ai-cloud/axolotl/releases) — *tag format verified against Docker docs, not browsed release-by-release (unverified at line level)*

### HuggingFace stack underneath Axolotl

- [huggingface.co — TRL docs (trainers: SFTTrainer, DPOTrainer, GRPOTrainer; v1 post-training library)](https://huggingface.co/docs/trl/index)
- [huggingface.co — PEFT docs (LoraConfig, adapters, `rank_pattern`/`alpha_pattern`)](https://huggingface.co/docs/peft/index)
- [huggingface.co — transformers docs (TrainingArguments, SFTConfig in TRL 1.x)](https://huggingface.co/docs/transformers/index) — *SFTConfig no longer inherits TrainingArguments in TRL 1.x confirmed via Spike 3's TypeError and trainer/Dockerfile commentary; not re-fetched here against transformers source at line level*
- [github.com/huggingface/accelerate — FSDP/DeepSpeed launcher, `accelerate launch`](https://github.com/huggingface/accelerate) — *referenced via Axolotl's `--launcher accelerate` support, not fetched as a separate primary in this pass*

### Alternatives (primary docs/repos, not blog summaries)

- [github.com/hiyouga/LLaMA-Factory — README (features, changelog, `use_unsloth: true` benchmark)](https://github.com/hiyouga/LLaMA-Factory)
- [docs.unsloth.ai — Docs home (kernel-accelerated fine-tuning)](https://docs.unsloth.ai/)
- [docs.unsloth.ai — Fine-tuning LLMs Guide (LoRA/QLoRA via `FastLanguageModel`, 2×/70% claims in project README)](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
- [github.com/unslothai/unsloth — README / Triton kernels, model coverage](https://github.com/unslothai/unsloth) — *cited for architecture claim, not for benchmark numbers beyond project-reported*

### This repo's evidence base (measured, not estimated)

- [trainer/Dockerfile:26 — `FROM axolotlai/axolotl:main-20260817-py3.12-cu130-2.12.0@sha256:29327…`](https://github.com/thp728/temper/blob/main/trainer/Dockerfile#L26)
- [trainer/entrypoint.py:164-396 — `build_config()`, YAML rendering, `axolotl train` subprocess, streaming, `result.json` always-written](https://github.com/thp728/temper/blob/main/trainer/entrypoint.py)
- [api/orchestrator.py:60-222 — tar-ship, build, run, stream, fetch, destroy-in-finally](https://github.com/thp728/temper/blob/main/api/orchestrator.py)
- [spike/README.md — Spikes 3, 4, 6, 7 (version chain, resume, FSDP, schema measurement)](https://github.com/thp728/temper/blob/main/spike/README.md)
- [spike/findings-spike7.json — 388 fields, 344 non-infra, 113 model_validators, 1 bounded field, 19 enums](https://github.com/thp728/temper/blob/main/spike/findings-spike7.json) *(generated from pinned digest via `introspect_axolotl.py`)*
- [docs/research-reports/report-a.md — tokenization spine principle, report-b.md — training loop, report-c.md — platform teardown](https://github.com/thp728/temper/tree/main/docs/research-reports)

---

## Caveats

- **Axolotl is a moving target.** This report pins claims to the `main-20260817-py3.12-cu130-2.12.0` digest and to Axolotl `0.19.0.dev0` (Spike 7's introspected version, `axolotl.utils.schemas.config.AxolotlInputConfig`). Field counts, `model_validator` counts, and tag grammars drift with `main`; re-introspect on each digest bump.
- **TRL 1.x breakage not re-fetched at line level here.** The `warmup_ratio` TypeError is measured in Spike 3 and asserted in `trainer/Dockerfile`'s comment; the underlying TRL source change (`SFTConfig` not inheriting `TrainingArguments`) was not re-verified by reading `trl` source at a specific commit in this pass — flagged **Medium** confidence for that exact inheritance detail, **High** for the observed failure.
- **Unsloth speed/VRAM numbers are project-reported.** The "2× faster, 70% less VRAM" shorthand appears in the project's own README/docs and community benchmarks; it is **Medium** confidence for any specific workload and should be measured per-model before quoting to users.
- **Production usage by others not exhaustively surveyed.** The pattern "Docker base + YAML + CLI" is the maintainer-recommended path per the docs reviewed, but a systematic survey of other platforms' Axolotl usage (their GitHub repos/docs) was not in scope for this pass — flagged as **Medium** for the "outlier vs common" claim.
- **FSDP loss collapse hypothesis not proven.** Spike 6's `nan` grad_norm candidates (bf16 + FSDP2 + gradient checkpointing; synthetic 28-trainable-tokens/step; flash_attention eager path) are hypotheses, not root-caused. The report's recommendation to gate multi-GPU on Issue #81's resolution reflects that uncertainty.
