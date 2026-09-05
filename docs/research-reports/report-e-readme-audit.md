# Report E — README-vs-Code Consistency Audit

*Scope: every verifiable claim in the root `README.md` (255 lines, as read 2026-09-05) checked against the primary source that owns it — `compose.yaml`, `justfile`, `apps/control-plane/src`, `apps/worker/src`, `apps/web`, `apps/trainer`, `packages/core/src`, `packages/contracts`, `LICENSE`. Secondary write-ups (ADRs, specs, spike notes) are cited only for provenance; where an ADR and the code disagree, the code wins and the disagreement is noted. No code was edited, nothing was provisioned, no GPU/network calls were made.*

*Convention: verdicts are **CONFIRMED** (code matches), **CONFLICT** (README contradicts code — both sides quoted), or **UNVERIFIABLE** (no code path found; search scope stated). Confidence labels (High/Medium/Low) mark non-obvious claims only.*

---

## TL;DR

- **54 claims checked: 34 CONFIRMED, 15 CONFLICT, 5 UNVERIFIABLE** (several claims are split — e.g. "warning yes / quote no" — and counted once under their dominant verdict; §7 lists each atomic verdict).
- The README is accurate on ports/URLs, health, seeding, banner, volumes, the no-Docker path, the credential failure path, the catalog, the trainer pin, the duration-warning mechanism, the destroy-then-confirm loop, the event/stream endpoints, `result.json`, sha256 verification, and the license.
- The conflicts cluster in **five staleness themes**: (1) the **worker migration shipped** but the README still says "one FastAPI process + a thread per job" and "worker empty on purpose"; (2) the **dataset-limit paragraph is two ADRs behind** (limit is now 5 GB, validation streams, streaming validation is built); (3) **features the README says don't exist do exist** — cost quote (`/v1/quotes`), inference endpoints (issue #78), HF dataset import (issue #45), loss-curve chart (issue #53), derived job-cost recording (issue #77); (4) **small-number drift** — cold-start ranges, "two containers", "exactly one value", "78.3%/11%", artifact path; (5) **over-strict absolutes** — "no I/O" in `packages/core`, "nothing hand-written" in `packages/contracts`.
- Nothing found suggests the README misleads about cost, safety defaults, or the zero-cost tier's blindness to transport defects. The fixes are documentation-only and prioritized in §8.

---

## 1. Running it — the one command (README L118–189)

### 1.1 `docker compose up` builds "two containers"

- **Verdict: CONFLICT (High).**
- README L131–132 says "That builds two containers, the control plane (the API) and the web shell".
- Code: `compose.yaml:52,57,72,109,151` declares **four** services — `postgres`, `control-plane`, `worker`, `web`. Header comment `compose.yaml:5–9`: "Four services: PostgreSQL … the control plane … the worker … and the web shell."
- `justfile:102–103` (`up: docker compose up`) is CONFIRMED — only the container count is wrong.

### 1.2 Ports and URLs table (5780 / 5781 / /health / /docs)

- **Verdict: CONFIRMED (High).**
- `compose.yaml:92` `"${TEMPER_BACKEND_PORT:-5781}:8000"`; `compose.yaml:169` `"${TEMPER_WEB_PORT:-5780}:3000"`.
- `apps/control-plane/src/temper_control_plane/main.py:1767–1768` `@app.get("/health") / def health()`.
- `/docs`: CONFIRMED by absence — `main.py:220–225` constructs `FastAPI(title=…)` with no `docs_url=None` override, so the default `/docs` serves. Internal corroboration: `apps/web/src/lib/backend.ts:5–6` (`BACKEND_PORT = 8000`), `apps/web/Dockerfile:53–61` (`next start -p 3000`).

### 1.3 `TEMPER_FAKE_PROVIDER=1`, `x-zero-cost-mode` anchor, "exactly one value", "every service", bare-process default

- **Verdict: CONFIRMED on mechanism, CONFLICT on wording (High).**
- Anchor exists: `compose.yaml:50` `x-zero-cost-mode: &zero-cost-mode "1"`, consumed at `compose.yaml:77` (control-plane), `:122` (worker), `:164` (web).
- "Exactly one value beyond defaults" is false as written: the same file also sets `TEMPER_DATABASE_URL` (`:82`, `:123`), `TEMPER_BACKEND_URL` (`:158`), `POSTGRES_*` (`:60–62`).
- "Every service reads that one value" is false for `postgres` (a stock DB image); true for the three app services.
- Bare-process default CONFIRMED: `config.py:270` `DEFAULT_FAKE_PROVIDER = False`, `config.py:273` `FAKE_PROVIDER = _flag(…)`; faults off by default `config.py:297–301` `DEFAULT_FAULT_SURFACE = False`, refusal `config.py:304–328` (`"fault_surface_refused"`; `config.py:295–296` "A default-configured start lands in the safe mode: real compute, faults refused").

### 1.4 Seeded sample dataset + completed run on first boot (`samples/sample-chat.jsonl`)

- **Verdict: CONFIRMED (High).**
- `apps/control-plane/src/temper_control_plane/seed_demo.py:47–48` `SAMPLE_DATASET_PATH = config.REPO_ROOT / "samples" / "sample-chat.jsonl"`; gate `seed_demo.py:149–160` `maybe_seed()` ("Does nothing unless the zero-cost tier is in force and the database is empty"); wired at boot `main.py:166–171`; the seeded run is driven for real `seed_demo.py:139` `orchestrator.run_job(job_id)`.

### 1.5 Demonstration banner on every page, same setting, cannot be turned off

- **Verdict: CONFIRMED (High).**
- Root layout `apps/web/src/app/layout.tsx:29,37` (`const zeroCost = isZeroCostMode(); {zeroCost ? <DemoBanner /> : null}`, `force-dynamic` at `:16` = every page); same-setting reader `apps/web/src/lib/demo-mode.ts:11–17`; no-dismiss marking `apps/web/src/components/DemoBanner.tsx:1–6`, pinned by `demo-marking.spec.ts:39–42`.

### 1.6 Data survives a restart (named volume; `down`/`up` preserves)

- **Verdict: CONFIRMED (High; singular/plural nuance).**
- `compose.yaml:64` `temper-postgres:/var/lib/postgresql/data`; `compose.yaml:94,133` `temper-objects:/data`; declared `compose.yaml:181–183`. README says "a named volume"; there are two. Header comment `compose.yaml:21–24` ("`docker compose down && docker compose up` keeps everything"); `justfile:105–108` `down: docker compose down`.

### 1.7 `/health` reports tier + database and object store separately

- **Verdict: CONFIRMED (High).**
- `main.py:1767–1787` `def health()` → `:1778` `"provider": "fake" if config.FAKE_PROVIDER else "real"`, `:1779` `"dependencies": dependencies`; split probes `main.py:1751–1764` (`"database", db.ping` + `"storage", storage.STORE.ensure_ready`); broken dep → 503 (`:1781–1787`).

### 1.8 No-Docker path (`just setup/check/dev/dev-web`, port 8000)

- **Verdict: CONFIRMED (High).**
- `justfile:20–21` `setup: uv sync`; `:28` `check: fmt-check lint types …`; `:79–80` `dev: uv run uvicorn temper_control_plane.main:app --reload` (no `--port`; 8000 is uvicorn's default, corroborated by `backend.ts:5` and `compose.yaml:92,43`); `:169–170` `dev-web: corepack pnpm --dir apps/web dev`.
- Prereq versions (uv/just/Node 22 + corepack): code corroborates Node 22 + corepack only (`Dockerfile:17,37` `FROM node:22-slim`, `justfile` `corepack pnpm` invocations); no `engines`/`.nvmrc`/uv-version pin found — **UNVERIFIABLE from code alone (Low significance)**.

### 1.9 "Without credentials … datasets validate fine. Jobs fail at provisioning with the reason named"

- **Verdict: CONFIRMED (High).**
- Boot warning `main.py:177–181` ("missing provider credentials, jobs will fail at provisioning", `reason="JL_API_KEY unset and no jl config file; datasets validate fine"`); failure at provider build `provider.py:285–291` (`"provider_unauthenticated"`, `"JL_API_KEY is not set…"`); surfaced as a job-record failure `orchestrator.py:2341–2359` ("job failed before provisioning", `error_code=e.code`).

### 1.10 Real jobs: `JL_API_KEY`, SSH key, `ssh-add -l`

- **Verdict: CONFIRMED (High).**
- `config.py:78–92` `provider_credentials_present()` (`JL_API_KEY` or jl config file); boot log `main.py:177–181`; job-record error `provider.py:285–291`; SSH-agent remedy in the error record `provider.py:364–370` (`"ssh_auth_failed"`, `"check \`ssh-add -l\` lists the JarvisLabs key"`).

### 1.11 Windows PowerShell-vs-Git-Bash trap

- **Verdict: CONFIRMED as documented; UNVERIFIABLE in product code, as the README itself predicts (Medium).**
- Zero hits in `apps/control-plane/src` / `apps/worker/src`; only `AGENTS.md:13–14`, `spike/AGENTS.md:19`, `spike/README.md:166`, `docs/specs/004-phase-b-spikes.md:348–349`. No boot check asserts the shell — consistent with README L216–218 ("The one thing nothing can check for you").

---

## 2. What it does + deliberately absent (README L10–34)

### 2.1 Browser journey (upload → validate → plan → watch → download; destroyed + confirmed gone)

- **Verdict: CONFIRMED (High).**
- Upload/validate: `main.py:521–545` (`upload_dataset` → `datasets.ingest`), `datasets.py:292–313` (`ingest` → `_validate_in_background`); model choice `main.py:447–464` (`list_models` → `catalog.CATALOG.values()`); plan + warning `main.py:726–767` (`get_job_spec_preview`, `"warning": warn`); stream `main.py:1171–1198` (`text/event-stream`), history-first-then-follow `main.py:1073–1168`; artifact `main.py:1286–1466` (streaming `_zip_chunks`, `attachment` filename); teardown `orchestrator.py:1479–1494` (`_teardown` → `confirmed_destroy`), `orchestrator.py:1434–1454` (`provider.list_machines()` loop, 3-consecutive-absent per `:179`, `destroying` ≠ absent `:1455–1461`).

### 2.2 "The shell's client is generated from the API's published contract, and it is the only interface"

- **Verdict: CONFIRMED with scope nuance (High).**
- `apps/web/orval.config.ts:1–22` (`input: "../../packages/contracts/openapi.json"`); `apps/web/AGENTS.md:8–13` ("generated, never hand-written"); `apps/control-plane/AGENTS.md:3–7` (server-rendered pages deleted, `/v1` contract only; no jinja/templates/`text/html` in `apps/control-plane/src`).
- "Only" means only *browser* surface — the README itself (L21) says "The same journey is available over the API", and the `/v1` routes exist (`main.py`: datasets L521–638, jobs L673–948, events L951–989, stream L1171, artifact L1286).

### 2.3 Cancellation + runaway-job limits, "exercised suite-wide … and on real runs"

- **Verdict: CONFIRMED on existence; UNVERIFIABLE on "exercised … on real runs" from code alone (Medium).**
- `db.py:628–676` (`request_cancel` / `cancel_requested`); `orchestrator.py:1521–1535` (`_cancellation_check`); `orchestrator.py:1826–1833` (`guard(provider.stream(…), limits, … check=cancelled)`); `limits.py:280–368` (`guard`: stall + duration + spend); `limits.py:114–121` (`RunLimits.from_config`; stall 15 min, duration 24 h per `config.py:151,159`).
- Suite-vs-stub: `tests/conftest.py:143–155` (autouse `no_real_provider`), `test_limits.py`, `test_orchestrator.py`, `test_teardown_confirmation.py` exist. "On real runs during the build" leaves no code artifact; not decidable read-only.

### 2.4 No auth/billing

- **Verdict: CONFIRMED with nuance (High).**
- `main.py:1–10` ("Deliberately not here: auth, billing"). Nuance: per-endpoint keys exist (`serving.py:15` "Keys are stored hashed"; `main.py:1708` `endpoint_unauthorized`) — endpoint-scoped keys, not product auth.

### 2.5 "A creation-time warning exists … An actual price quote does not, and no code path reads an invoice"

- **Verdict: SPLIT — warning CONFIRMED, "no quote" CONFLICT, "no invoice" CONFIRMED (High).**
- Warning: `jobs.py:242–245` (`feasibility.warning(…)`), frozen `jobs.py:279–296`.
- Quote EXISTS: `packages/core/src/temper_core/quote.py:1–27` (estimate engine); `main.py:770–842` (`GET/POST /v1/quotes`); `jobs.py:77–80,285` (quote frozen on job row); `QuoteView.tsx` exists. README L30 is stale (spec 005 shipped).
- Invoice: grep for `invoice` finds hits only in `README.md:30,112,248`, `docs/`, `spike/` — zero under `apps/control-plane/src` and `packages/core/src`. CONFIRMED absent.

### 2.6 "An inference endpoint [does not exist]"

- **Verdict: CONFLICT (High).**
- README L31 vs `serving.py:1–41` ("Temporary authenticated endpoints (issue #78)… answers prompts"); `main.py:1580` + `1682–1725` (endpoint create/infer routes with key auth); `apps/web/src/components/EndpointSection.tsx` exists.

### 2.7 "Imports from Hugging Face [do not exist]. Models come from a curated, pinned catalog of two, and datasets are uploaded files"

- **Verdict: SPLIT — catalog-of-two CONFIRMED; "no imports / upload-only" CONFLICT (High).**
- Catalog: `catalog.py:63–91` (two pinned entries; see §3.3).
- Imports EXIST: `remote_datasets.py:1–39` (HF parquet import seam, issue #45); `main.py:548–583` (`POST /v1/datasets/import`); `ImportForm.tsx` exists. Out-of-catalog models also admittable: `main.py:472–498` (`probe_model`).

### 2.8 "Streaming validation … not built, so uploads stay capped"

- **Verdict: SPLIT — capped CONFIRMED; "not built" CONFLICT (High).**
- Cap: `datasets.py:271–289` (`total > cfg.MAX_DATASET_BYTES` → 413); `config.py:220–226` (`DEFAULT_MAX_DATASET_MB = 5120` — see §3.5 for the limit-value conflict).
- Streaming EXISTS: `datasets.py:132–167` (`validate_chunks(storage.STORE.get_stream…)`), `validation.py:78–87` (`CHUNK_BYTES`, "keep memory flat"), `validation.py:298+` (`validate_chunks`). Whether it achieves flat-+4 MB is UNVERIFIABLE from code alone (needs a memory measurement, not a reading).

### 2.9 "Temporal, Redis and MinIO are specified and not built"

- **Verdict: SPLIT (Medium).**
- Temporal/Redis CONFIRMED absent (no hits in `*.py`; `compose.yaml:52–183` services are only postgres / control-plane / worker / web).
- MinIO: no MinIO *service* ships in compose (volumes are `temper-postgres`, `temper-objects`), but an S3-compatible seam exists — `storage.py:12–13` ("`S3Storage` — any S3-compatible store… MinIO included"), `config.py:400` (`TEMPER_S3_ENDPOINT_URL`), `storage.py:31–35` (moto-backed S3 tests; "does not prove MinIO itself").

### 2.10 "Persistence moved to PostgreSQL … with real versioned migrations … Everything else still runs as one FastAPI process and a thread per job"

- **Verdict: SPLIT — Postgres CONFIRMED; "one process + thread per job" CONFLICT (High).**
- Postgres: `db.py:1–27` ("Function names and return shapes stay; only their bodies change"); `migrations.py:1–39` (0001 baseline, 0002 Phase-B tables), `migrations.py:59–80` (`CREATE TABLE datasets… jobs…`); `db.py:528–559` (transition + event in one transaction).
- Worker SHIPPED (issue #51, ADR-0066): `apps/worker/src/temper_worker/worker.py:1–42,113–195` (`run_once` claims via `db.claim_next_job` → `orchestrator.run_job`); `db.py:1279–1403` (`claim_next_job`, `FOR UPDATE SKIP LOCKED`); `compose.yaml:109–149` (worker service, `python -m temper_worker`); `main.py:182–194` ("orchestration now lives in the worker… no thread in this process"); `jobs.py:297–304` ("The request path no longer starts threads"); `apps/control-plane/AGENTS.md:9–17`. README L34 describes the pre-#51 world.

### 2.11 "The domain logic already extracted into `packages/core`"

- **Verdict: CONFIRMED (High).** `packages/core/src/temper_core/__init__.py:1`; grep of `packages/core/src` for `fastapi|sqlalchemy|psycopg|requests|httpx|urllib|socket` finds only `validation.py:231` (`path.open("rb")` inside a stdlib file-helper; imports at `validation.py:53–63` are stdlib + `temper_core` only). See §6.2 for the "no I/O" absolute.

---

## 3. What it runs — method, models, compute, limits (README L36–48)

### 3.1 SFT method "chosen by the predictor and overridable"

- **Verdict: CONFIRMED (High).**
- `packages/core/src/temper_core/selection.py:14–19` ("ranges over the whole (method, gpu_type, device_count) space … cheapest"); `packages/core/src/temper_core/overrides.py:1–9` ("Pin one decision, recompute the rest"); `jobs.py:225–237` (pins refused with `not_executable`, never silently ignored).

### 3.2 QLoRA definition (NF4 double-quant, bf16 compute, rank 16, α=32, all linear)

- **Verdict: CONFIRMED (High).**
- `apps/trainer/entrypoint.py:864–875` (`"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16", "lora_target_linear": True`); `packages/contracts/trainer-defaults.json:3–4` (`"lora_r": 16, "lora_alpha": 32`).

### 3.3 Full fine-tuning (issue #66: every weight in bf16, own lower LR, whole model as one archive)

- **Verdict: CONFIRMED as built, CONFLICT on framing (Medium).**
- `entrypoint.py:850–855` (full branch `"load_in_4bit": False`, `"bf16": True` at `:814`); `trainer-defaults.json:16–20` (`by_method.full.learning_rate: 1e-5` vs default `2e-4`); `entrypoint.py:918–925` (`FULL_MODEL_ARCHIVE = "model.tar.gz"`, "shipped as one streamed archive"). README cites "#66" as if pending; `apps/trainer/README.md:46–66` documents both methods as shipped.

### 3.4 "The QLoRA adapter ships fp32 … 132 MB rather than around 66 MB"

- **Verdict: CONFIRMED (High).**
- `docs/adr/0008-adapters-ship-as-fp32.md:17–20` ("all 504 adapter tensors are F32, which is why a Qwen3-4B LoRA at r=16 is a 132 MB download rather than the ~66 MB a half-precision cast would produce"); `entrypoint.py:857–863` (same measurement inline); `apps/trainer/README.md:227` ("Adapter | safetensors, 132.2 MB, `r=16 alpha=32 rslora=False`").

### 3.5 Dataset limit "1 GB per upload (`TEMPER_MAX_DATASET_MB`)" + "holds about 5.9×" + "streaming … not built yet" + refusal naming both sizes

- **Verdict: CONFLICT on limit value and "not built"; CONFIRMED on refusal mechanics (High).**
- Limit value: `config.py:220` `DEFAULT_MAX_DATASET_MB = 5120` (5 GiB; comment `:203–212` cites wireframe 5 GB + measured 21.6 MB/s). README's 1 GB matches neither the current default (5 GB) nor ADR-0036's interim 1.3 GB (`docs/adr/0036…:44–51`).
- 5.9×: CONFIRMED as history, CONFLICT as present tense — `validation.py:9–11` ("measured at 5.93x its size, spike 9"; ADR-0005 correction 4.8×→5.93×), but validation **now streams** (`validation.py:7–8` "reads the file in chunks… peak memory stays flat"; `datasets.py:144–148` calls `validate_chunks`; ADR-0036 accepted 2026-08-27 supersedes ADR-0005).
- Streaming "not built": CONFLICT — `docs/adr/0036…:18–20` ("peak RSS flat at +4 MB from 1 GB to 20 GB"), `:33–42` ("Validation streams… one row at a time"); implementation in `validation.py` + `datasets.py` above.
- Refusal naming both sizes: CONFIRMED — `datasets.py:51–67` (`too_large` 413 names `actual` and `limit`); `:70–82` (`refuse_before_read`); ADR-0036 `:53–56` (mid-stream refusal covers lying Content-Length).

### 3.6 Duration warning (24-hour ceiling, ~1.19 row-passes/s, launches anyway)

- **Verdict: CONFIRMED (High).**
- `config.py:159` `DEFAULT_MAX_JOB_DURATION_S = 24*60*60`, `:170–172` (`TEMPER_MAX_JOB_DURATION_S`); `feasibility.py:22–29` ("64 rows x 3 epochs = 192 row-passes in 161.4s… ~1.19 row-passes/s"), `:40` (`ROWS_PER_SECOND = 192/161.4`); `jobs.py:239–245` (warning computed at creation; job proceeds to `db.create_job` at `:279+`); `feasibility.py:14–18` ("A warning, never a refusal… The job launches regardless").

### 3.7 Predictor ("cheapest configuration that fits", "prefers the more capable method at a tied price", "showing its reasoning")

- **Verdict: CONFIRMED with mechanism nuance (Medium).**
- `selection.py:39–50` (`METHODS_BEST_FIRST = ("full","lora","qlora")`, "the search takes the first executable method that fits at each price point, so the more capable method wins a tied price"); `:421–428` (break after first fitting method, best-to-worst, per price point); `decisions.py:1–11` ("every choice the predictor made, with why and the cost of the alternatives"). Nuance: the tie preference is implemented via best-first search order + per-price-point break, not a score comparison in `_cheaper` (tie-breaks only on device count, `:177–193`).

### 3.8 Models "curated and pinned: `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`"

- **Verdict: CONFIRMED (High).**
- `catalog.py:63–91`: `qwen3-4b` repo `Qwen/Qwen3-4B` rev `1cfa9a72…df60c`, `qwen3-8b` repo `Qwen/Qwen3-8B` rev `b968826d…91218`, both `license="Apache-2.0"`; `:93–101` (fail fast unless 40-char SHA).

### 3.9 Compute "JarvisLabs VMs, provisioned and destroyed per job" + "Trainer: Axolotl in a digest-pinned container"

- **Verdict: CONFIRMED (High).**
- `provider.py:279–333` (`JarvisLabsProvider`: `create` → `Machine`, `await_ready`, `destroy`); `orchestrator.py:2738–2745` (per-attempt `finally:` destroy-before-next-provision); `:1374–1469` (`confirmed_destroy` requires consecutive absent listings).
- `apps/trainer/Dockerfile:26` (`FROM axolotlai/axolotl:main-20260817-py3.12-cu130-2.12.0@sha256:29327e75…c701e`; `:24–25` "PIN DISCIPLINE: the digest is the contract").

---

## 4. Architecture diagram (README L50–86)

Each mermaid arrow was checked; the sequence is right, three labels are stale.

### 4.1 Validate (`packages/core`, pure, off the event loop) → line-numbered report, stable-code refusal

- **Verdict: CONFIRMED (High).**
- `validation.py:236–630` (pure `validate`/`validate_bytes`/`validate_chunks`); `datasets.py:169–187` (background thread, "Runs off the request thread"); `main.py:527–545` (`def` worker-pool handler citing ADR-0006); stream reads via `asyncio.to_thread` (`main.py:1108–1150`).
- `validation.py:91–97` (`Issue{line, code, message}`); `:202–226` (`line_no` iteration); codes e.g. `invalid_json` `:404`, `missing_messages` `:516`, `mixed_thinking` `:592`, `too_few_rows` `:614`; `jobs.py:45–55` (invalid dataset refused, `dataset_invalid`).

### 4.2 "Orchestrator thread: provision → bootstrap → train → collect → destroy"

- **Verdict: SPLIT — sequence CONFIRMED, "thread" CONFLICT (High).**
- Sequence: `orchestrator.py:1` ("provision, bootstrap, train, collect, destroy"); stages `:1700–1726` (provision/create + `preparing`), `:1728–1775` (await_ready + `push_stream`), `:1777–1834` (`training` + `guard(provider.stream…)`), `:725–840` (verify/collect), `:1479–1494` (teardown; docstring `:14–18`: `finally`, confirmed *before* terminal transition). Driver is the worker *process*, not a thread: `worker.py:160–162`, `main.py:182–194` (see §2.10).

### 4.3 "Digest-pinned Axolotl container, no published ports"

- **Verdict: CONFIRMED (High).** `Dockerfile:26` (digest pin); no `EXPOSE` in Dockerfile; run command `orchestrator.py:521–525` (`--gpus all`, `-v…`, no `-p`); `orchestrator.py:482–483` + trainer AGENTS.md no-ports rule.

### 4.4 "provision (L4 / RTX-PRO6000 / H100)"

- **Verdict: CONFIRMED with omission note (Medium).**
- `packages/core/src/temper_core/gpus.py:17–21` (`L4 24, RTX-PRO6000 96, H100 80, H200 141`); `selection.py` ranges over `(method, gpu_type, device_count)` (`:15`, `:331+`); `orchestrator.py:1700–1719` (`provider.create(plan.gpu_type, plan.device_count, …)`). The mermaid omits H200, which the code supports.

### 4.5 "push dataset + trainer sources over SSH"

- **Verdict: SPLIT — dataset CONFIRMED, "trainer sources" CONFLICT (High).**
- Dataset: `provider.py:76–95` (wire `cat >` push), `:191`/`:379` (`push_stream` impl), `orchestrator.py:1731–1740` (dataset tar push), `:1761–1765` (checkpoint push). Trainer sources are NOT pushed — the machine pulls the published digest: `orchestrator.py:1734–1735`, `:265–292` (`_trainer_reference`), remote script `:495–507` (`docker pull {reference}`); sources baked at build (`Dockerfile:104` COPY).

### 4.6 "stdout event stream over SSH, guarded by stall + duration limits"

- **Verdict: CONFIRMED (High).** `provider.py:208`/`510` (`stream`); `orchestrator.py:1826–1833` (guard wraps transport); `limits.py:280–368` (stall + duration + spend ceiling on the same clock tick, `:342–347`).

### 4.7 "state transition + event appended in one transaction"

- **Verdict: CONFIRMED (High).** `db.py:528–559` + header `db.py:18–22`; `test_transactions.py` exists.

### 4.8 Poll `/v1/jobs/{id}/events`, stream `/v1/jobs/{id}/stream`

- **Verdict: CONFIRMED (High).**
- `main.py:951–989` (`after`/`limit≤500`, `total`, progress+output); `main.py:1171–1198` (SSE, `Last-Event-ID` cursor via `_stream_cursor` `:1003–1020`).

### 4.9 "`result.json` — always written, pass or fail"

- **Verdict: CONFIRMED (High).** `entrypoint.py:1920–1932` (`finally: … write_result(result)`); missing-result path named not parsed: `orchestrator.py:153–167` (`NO_RESULT`), `:666–681`.

### 4.10 "verify machine-written artifact bytes, sha256-checked against result" → `data/artifacts/<job>/`

- **Verdict: SPLIT — sha256-verify CONFIRMED, path CONFLICT (Medium).**
- Verify: `orchestrator.py:725–840` (checksum required `:754–764`, streamed hash `:769–774`, mismatch → `artifact_corrupt` + object deleted `:789–801`). Destination is object key `artifacts/{job}/{name}` under the storage seam (`storage.py:79–81`), STORE root = `/data` volume (`compose.yaml:94,132`) — not literal `data/artifacts/<job>/`. Also the machine writes directly via scoped grant (ADR-0009); the control plane verifies, not collects (`orchestrator.py:1778–1789`).

### 4.11 "destroy, then confirm by listing machines"

- **Verdict: CONFIRMED (High).** `orchestrator.py:1406–1476` (`provider.destroy` `:1410`, retry ×3 `:1408–1421`, `list_machines` `:1436`, 3-consecutive-absent `:1448`, `destroying` ≠ absent `:1455–1461`, STRAY escalation `:1471–1476`); `provider.py:14–15`.

---

## 5. Why these choices + Measured, not estimated (README L88–112)

### 5.1 All-linear 78.3% / attention-only ~11%

- **Verdict: UNVERIFIABLE on precision (Medium).**
- Exact "78.3%" occurs only in `README.md:92`. All code/contracts say "~78%": `entrypoint.py:872` ("The MLP is ~78%…"), `axolotl-field-tiers.json:995`, `advanced-surface.json:601`. "11%" likewise only in the README. Direction CONFIRMED, decimals sourced nowhere in the repo.

### 5.2 Digest-vs-six-packages story (transformers 5.15 / trl 1.10 / torch 2.5.1)

- **Verdict: CONFIRMED (Medium).** `apps/trainer/Dockerfile:6–10` + `apps/trainer/README.md:9–13` (TRL 1.x drops `warmup_ratio` → `save_safetensors` rejected → `.bin` → transformers 5.x requires `torch >= 2.6`); base carries torch 2.12 (`Dockerfile:20–21`, README `:19`).

### 5.3 Correctness locked by default; Advanced exposes each + failure mode + recorded override + export-time probe

- **Verdict: CONFIRMED (High).**
- `surface.py:19–28` (tiers incl. "exposed with a named failure mode — … the specific thing that goes wrong written beside it"); `entrypoint.py:1134–1171` (`probe_export`; "Identical token ids are required; the probe runs on every export"), invoked `:1821`; overrides frozen into the job spec (`jobs.py:279+`; `overrides.py:185–203` `ResolvedOverrides.overridden`).

### 5.4 Measured table — VRAM 5.31 GB / 33,030,144 params / resume / 336s / ₹4.8

- **Verdict: CONFIRMED except cold-start ranges and the cost-computation footnote (High).**
- VRAM: `memory.py:33–37` ("the run's own peak… was 5.31 GB (2026-08-19, apps/trainer/README.md)"); `tests/test_memory.py:59–60` (`MEASURED_PEAK_GB = 5.31`), `:86–98` (prediction-within-tolerance test).
- Params: `memory.py:26–31` ("Qwen3-4B gives 33,030,144 at r=16… Verified against both real Qwen3 configs"); `tests/test_memory.py:66–70` (exact-match test); run-side `docs/adr/0021…:138–140` + `spike/README.md:158`.
- Resume: `apps/trainer/README.md:226` ("Resume from `checkpoint-2` to step 6 | yes, 54s").
- 336s / 363s / ₹4.8: `docs/adr/0021…:18–21` ("336s wall clock… provision 10s, SSH ready 47s, image build 87s, train 161s"), `:103–110` ("VM … destroyed 363s later, so at ₹41.31/hr… ≈₹4.8"); rate `spike/findings-spike5.json:24`; `apps/trainer/README.md:236`.
- Cost provenance: `actuals.py:34–39` ("Cost is derived, never measured: no code in this product reads a bill… measured duration times the rate the job froze at launch"); `provider.py:318–319` (`currency()` read live).

### 5.5 Cold-start breakdown (10–13s provision / 40–47s SSH / 87–183s pull)

- **Verdict: CONFLICT on two of three ranges (Medium).**
- Pull 87–183s matches `quote.py:78` (`IMAGE_PULL_S = (87.0, 183.0)`). But provision: code says `quote.py:67` `(13.0, 17.0)` (README's 10–13s is the single 08-19 run in ADR-0021 `:19–20`). SSH: code says `quote.py:72` `(40.0, 68.0)` citing spike 5's 68s — README caps at 47s (the single-run figure), dropping the measured 68s upper end.

### 5.6 "Nothing in the product computes a job cost yet"

- **Verdict: CONFLICT — stale (High).**
- `orchestrator.py:1919–1956` (`_record_actuals` → `actuals.measure` → `db.record_actuals`, frozen at terminal on every run at `:2761`); `actuals.py:145–163` (`_derived_cost_minor`); `contracts_models.py:583` (publishes `cost_minor`). Product computes derived job cost since issue #77. The "derived, never invoiced" half remains true.

---

## 6. Layout (README L220–239)

### 6.1 `apps/control-plane`, `apps/web`, `apps/trainer`, `packages/contracts`, `spike/`, ADR-0010 rule

- **Verdict: CONFIRMED (High).**
- Control plane routes (`main.py:446` models, `:883` cancel, `:951` events + `limit: int = 500`, `:1171` stream; `jobs.py:59` "Validate the request, freeze the spec and launch"); web generation (`orval.config.ts:8–10` `input: ../../packages/contracts/openapi.json → output: ./src/lib/api/generated/client.ts`; `.gitignore:54` generated dir; `control-plane/AGENTS.md:5–8` server pages deleted at last Spec 007 port); trainer `/job`→`/out` (`Dockerfile:108–110` + `VOLUME ["/job", "/out"]`; `apps/trainer/README.md:36–44`); contracts cross-boundary set (`openapi.json`, `advanced-surface.json`, `axolotl-schema.json`, `axolotl-field-tiers.json`, `trainer-defaults.json`, `trainer-image.json`, `fault-surface.json`, `delivery-formats.json`; `packages/contracts/README.md:9`); spike probes (`spike.py`…`spike9.py`, `findings-spike*.json`); rule + flat-alternative record (`docs/adr/0010…:50–51`, `:53–61`, `:113–161`).

### 6.2 `apps/worker` "reserved for orchestration (#51). Empty on purpose"

- **Verdict: CONFLICT — stale (High).**
- `apps/worker/src/temper_worker/` holds `worker.py`, `reconciler.py`, `__main__.py`; `apps/worker/README.md:3–6` ("The process that claims queued jobs and drives them (issue #51). It polls `temper_control_plane.db.claim_next_job`, which uses `SELECT ... FOR UPDATE SKIP LOCKED`"); `pyproject.toml:1–3` ("Reserved by ADR-0010… Now the process that claims queued jobs"); `compose.yaml:8–9` ("The worker was empty by design … and joins this file now that it ships"). Was true; no longer true.

### 6.3 `packages/core` ("No framework imports, no I/O")

- **Verdict: SPLIT — contents + no-framework CONFIRMED; "no I/O" CONFLICT (Medium).**
- Contents: `validation.py`, `hyperparams.py`, `feasibility.py`, `thinking.py`, `catalog.py`, `events.py` all present. No-framework: grep for `^(import|from) (fastapi|pydantic|sqlalchemy|starlette)` over `packages/core/src` = no matches; `pyproject.toml:12` (`dependencies = []`; `:4–7` "no web framework, no ORM, no cloud SDK").
- I/O: `validation.py:231` (`with path.open("rb") as fh:`). Self-corrected on record: `docs/adr/0010…:228–233` ("The Decision also says `packages/core` has 'no I/O'. It has some").

### 6.4 `packages/contracts` ("generated artifacts crossing a language boundary")

- **Verdict: SPLIT — exists + cross-boundary CONFIRMED; "nothing hand-written" CONFLICT (Low).**
- `packages/contracts/README.md:9` (`openapi.json` from the FastAPI app → generated TS client in `apps/web`) vs `:13` (`trainer-defaults.json` "hand-maintained") vs `README.md:3–5`-adjacent claim "Nothing here is written by hand". Self-corrected: `docs/adr/0010…:257–268` ("some of those are generated (`openapi.json`) and some are not (the defaults)"). The generated *client* itself is gitignored output at `apps/web/src/lib/api/generated/`, not in `contracts/`.

---

## 7. Known gaps + License (README L241–255)

### 7.1 "Loss reaches the page as a number, not a curve"

- **Verdict: CONFLICT — since built (Medium).**
- Curve exists: `JobRecordView.tsx:667` (`latestLoss(events)`), `:669–674` ("The two series the loss chart draws (issue #53)"), `:751–759` (loss-curve tab `loss-chart-heading`); `lib/jobs/loss.ts:65–68` (`lossSeries… { training; heldOut }`). Classifier provenance CONFIRMED: `events.py:15–31` (transformers log-dict repr, `'loss'`-key rule, held-out series).

### 7.2 "A finished job's page truncates its log" (cap 500; live polls past it)

- **Verdict: CONFIRMED with mechanism nuance (High).**
- `db.py:1254` / `main.py:952` (`limit: int = 500`); `main.py:971–978` (`invalid_limit` outside 1–500); `main.py:964–969` (`total` regardless of window; full record reachable by paging `after` — issue #56). Single-page window truncates; not silent now (`db.py:1238–1245` `count_events`). Live path walks past 500 over time (`main.py:1138–1140` cursor re-read; `stream.ts:23–25` `after`).

### 7.3 "The job log is mostly build noise" (several hundred events, majority build progress)

- **Verdict: CORROBORATED as code-doc; counts UNVERIFIABLE read-only (Low).**
- `events.py:39–47` ("docker's layer-pull lines and the model-download bars, hundreds per pull… promoted into a `progress` record… raw lines retained as collapsed detail"); `:114+` (pull/download shapes). "Several hundred / large majority" are run-output quantities; no fixture pins the count.

### 7.4 "Costs are derived, never invoiced" + "mainly started on Windows" + spec 012 cold-clone

- **Verdict: CONFIRMED (High).**
- `quote.py:1–9` ("every figure here is a **range**, never a point"), `:21–26` (anchored to measured 1.19 row-passes/s); `invoice` = zero hits under `apps/control-plane/src`, `packages/core/src`. Windows/restart/cold-clone: README `:210–218`, `:126–129`, `:171–177` + `compose.yaml:21–24` (volumes `:181–183`); `docs/specs/012-clone-and-run.md:16` ("never run on anything but Windows"), `:115–119` (cold clone as gate), `:155–158` (verification clause).

### 7.5 License (MIT; over Apache-2.0/GPL/source-available per ADR-0012; adapters carry Qwen3 terms; catalog surfaces license + revision)

- **Verdict: CONFIRMED (High).**
- `LICENSE:1` (`MIT License`); `docs/adr/0012…:33` (MIT as root `LICENSE`), `:45–61` (Apache-2.0 rejected — patent grant insures a nonexistent risk), `:63–66` (GPL/AGPL copyleft rejected), `:68–71` (source-available rejected), `:81–84` ("an adapter produced from Qwen3 weights carries Qwen's Apache-2.0 terms").
- Catalog: `catalog.py:71,81` (`license="Apache-2.0"` both; `:53` "Both are dense and Apache-2.0"). Surfacing: `contracts_models.py:234–251` (`CatalogEntry` carries `license`, `license_url`, `revision`); `main.py:122–157` (`_catalog_entry` spreads `**m.to_dict()` + fresh `peak_memory`); `main.py:446–464` (`GET /v1/models`). Scope note: the durable job row freezes the *revision* (`jobs.py:284` `base_revision`; `contracts_models.py:815`), with the license shown at creation from the catalog.

---

## 8. Prioritized README fixes

Ordered by mislead-potential × read-frequency (highest first). Each item names the README lines to change and the code that wins.

1. **Dataset-limit paragraph (L46) — rewrite against ADR-0036.** Limit is `config.py:220` (5 GB default), validation streams (`validation.py:7–8`, `datasets.py:144–148`), streaming validation is built (`docs/adr/0036…:18–20,33–42`). The whole "not built yet / 1 GB / 5.9× present-tense" paragraph is two ADRs behind.
2. **Worker / orchestration model (L34, L63–66 mermaid label, L229).** "One FastAPI process and a thread per job" → worker process claims via `SELECT … FOR UPDATE SKIP LOCKED` (`db.py:1279–1403`, `worker.py:113–195`, `compose.yaml:109–149`); "worker/ Empty on purpose" → ships (issue #51). Also fix the mermaid "Orchestrator thread" label and the "two containers" count (L131–132; four services).
3. **Features listed as absent that exist (L29–34 + gaps).** Cost quote exists (`quote.py`, `main.py:770–842`, `QuoteView.tsx`); inference endpoints exist (issue #78: `serving.py`, `main.py:1580,1682–1725`); HF dataset import exists (issue #45: `remote_datasets.py`, `POST /v1/datasets/import`); loss curve exists (issue #53: `JobRecordView.tsx:667–674,751–759`); derived job cost is recorded (issue #77: `orchestrator.py:1919–1956`). Either document them or move them to "since built".
4. **"Nothing computes a job cost yet" (L112) + cold-start ranges (L108).** Cost: contradicted by `_record_actuals`/`record_actuals`/`cost_minor`. Ranges: provision `(13.0, 17.0)` (`quote.py:67`) not 10–13s; SSH `(40.0, 68.0)` (`quote.py:72`) not capped at 47s; pull `(87.0, 183.0)` (`quote.py:78`) is correct.
5. **SSH-push / artifact-path mermaid labels (L76, L81).** Trainer sources are pulled as a digest (`orchestrator.py:265–292,495–507`), not pushed over SSH; artifact destination is object key `artifacts/{job}/{name}` (`storage.py:79–81`), not literal `data/artifacts/<job>/`.
6. **Absolutes to soften (L34, L232–234, L131–132, L92, L179–182).** `packages/core` "no I/O" → has `validation.py:231` (already corrected in ADR-0010 `:228–233`); `contracts` "generated artifacts" → includes hand-maintained `trainer-defaults.json` (already corrected in ADR-0010 `:257–268`); "exactly one value" / "every service" → three app services + `TEMPER_DATABASE_URL`/`TEMPER_BACKEND_URL`/`POSTGRES_*`; "78.3%"/"11%" → "~78%" (exact figures exist nowhere but the README); H200 omission from the provision label (`gpus.py:17–21`).
