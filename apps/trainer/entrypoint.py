"""Trainer entrypoint ΓÇö job spec in, adapter + result out.

Runs inside the pinned image. The orchestrator writes a job spec and a dataset
to /job, runs this container, and reads /out.

    /job/job.json       the job spec (see job.example.json)
    /job/dataset.jsonl  training data, one JSON object per line
    /out/               config.yaml, checkpoints, adapter/, result.json, train.log

Design notes worth keeping, because each is a decision:

* **Axolotl owns the training loop, we own the contract.** We render YAML and
  invoke `axolotl train`. We do not call TRL/PEFT directly ΓÇö spike 3 showed
  those APIs move underneath you (TRL 1.x dropped `warmup_ratio` from
  SFTConfig), while Axolotl's config surface stayed stable and still exposes it.
* **Correctness settings are not user-settable.** Chat template resolution,
  EOS handling and loss masking are the highest-frequency silent-failure
  surface: they pass every obvious health check and only show up as garbage
  generations. They are decided here, not exposed.
* **The trainer resolves nothing.** Hyperparameters arrive in the job spec
  already resolved by the control plane (#83); this entrypoint applies what it
  is given, refuses a spec missing required values (`spec_incomplete`), and
  echoes unknown keys back rather than dropping them. Two resolvers that could
  disagree meant the one that ran was the one nobody could see.
* **Everything is reported, including failure.** result.json is always written,
  even on a crash, so the orchestrator never has to parse logs to find out what
  happened.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from thinking import MixedThinkingDataset
from thinking import detect as detect_thinking


def _contract_path() -> Path:
    """Find the single definition of the trainer's defaults.

    Sibling first: in the image this file sits beside trainer-defaults.json at
    /opt/trainer (copied by the Dockerfile), and in the repo it sits in
    apps/trainer. Falls back to the workspace tree so an import under the test
    suite resolves without the image. The trainer never installs
    packages/core (ADR-0010), so it reads the data, not the resolver module.
    """
    sibling = Path(__file__).resolve().parent / "trainer-defaults.json"
    if sibling.is_file():
        return sibling
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "packages" / "contracts" / "trainer-defaults.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "packages/contracts/trainer-defaults.json not found beside this file "
        "or anywhere in the workspace tree"
    )


JOB_DIR = Path(os.environ.get("JOB_DIR", "/job"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/out"))
CONFIG = OUT_DIR / "config.yaml"
RESULT = OUT_DIR / "result.json"
LOG = OUT_DIR / "train.log"

# The trainer resolves nothing (#83): the control plane applies its resolver to
# the user's overrides before launch and writes the full resolved set into the
# job spec, so there is one resolver, the visible one. What remains here is the
# guard: the keys this entrypoint enforces are not declared here. They are read
# from the single definition in packages/contracts/trainer-defaults.json (#82),
# copied beside this file at build time and resolved through the same data by the
# control-plane resolver. A default added there without the trainer learning it
# would fail this guard before the GPU does any work -- training on a number
# nobody chose is worse than not training -- and the equality is pinned by
# apps/trainer/tests/test_agreement_with_the_domain.py, so the two cannot
# drift. `lora_use_rslora` is in the resolved set because the resolver infers it
# at rank >= 32 rather than carrying it in the data.
_CONTRACT = json.loads(_contract_path().read_text(encoding="utf-8"))
REQUIRED_HYPERPARAMETERS = set(_CONTRACT["defaults"]) | {"lora_use_rslora"}
KNOWN_HYPERPARAMETERS = REQUIRED_HYPERPARAMETERS | {
    # Optional wherever they appear; smoke tests use them to keep a job short.
    "max_steps",
    "save_steps",
}

# Top-level keys the job spec may carry. Spike 4 caught a real hole here: an
# unknown key at the TOP level passed silently because only `hyperparameters`
# was being validated. A caller who misspells `max_steps` as `maxSteps` would
# have got a full-length training run and no warning. Keys starting with
# "_comment" are documentation and ignored on purpose.
ALLOWED_JOB_KEYS = {
    "job_id",
    "base_model",
    "base_revision",
    "messages_field",
    "hyperparameters",
    "max_steps",
    "save_steps",
    "resume_from_checkpoint",
}


class IncompleteJobSpec(ValueError):
    """The job spec omitted values the trainer refuses to invent."""


def log(msg: str) -> None:
    print(f"[trainer] {msg}", flush=True)


def write_result(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2))


# How much of a failed job's output is carried back inside result.json. The
# lines were already streamed; this is so the orchestrator never has to
# reassemble a failure from the event log.
TAIL_LINES = 40

# A line ends at a newline OR a carriage return; see iter_output_lines.
RB_LINE_END = re.compile(rb"[\r\n]")


def iter_output_lines(
    stream: IO[bytes], chunk_size: int = 4096
) -> Iterator[str]:
    """Yield the framework's output a line at a time, as it is produced.

    Not `for line in stream`, for two reasons:

    * **Progress bars never send a newline.** tqdm ΓÇö which transformers uses
      for every epoch ΓÇö redraws with a carriage return. A reader that waits
      for `\n` sees nothing for the whole length of a bar, which on a training
      phase that *is* one bar is indistinguishable from the silence this
      channel exists to remove. So `\r` ends a line here too.
    * **Iterating a pipe in text mode buffers.** Python reads ahead, so lines
      arrive in blocks rather than as they happen.

    Blank lines are dropped: each one would otherwise become an event, and a
    framework that prints a spacer between epochs would triple the log.
    """
    buf = b""
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        buf += chunk
        parts = re.split(RB_LINE_END, buf)
        buf = parts.pop()  # whatever follows the last terminator
        for part in parts:
            text = part.decode("utf-8", "replace").strip()
            if text:
                yield text
    tail = buf.decode("utf-8", "replace").strip()
    if tail:
        yield tail


def run_streaming(cmd: list[str]) -> tuple[int, list[str]]:
    """Run `cmd`, relaying its output to stdout as it arrives.

    Returns its exit code and the tail of what it said.

    Three things here are load-bearing, and dropping any one of them
    reintroduces the silence:

    * `bufsize=0` so the pipe is read raw ΓÇö a buffered reader waits to fill a
      block before handing us anything.
    * `PYTHONUNBUFFERED` in the child's environment, because the child is
      itself Python and will otherwise buffer its own stdout when it sees a
      pipe rather than a terminal. This is the innermost of the redirections;
      the two outside it are worthless if this one holds output back.
    * `flush=True` on every relayed line, for the same reason one layer up.

    train.log is still written. It costs nothing, and it is the only copy that
    survives on the machine if the stream itself breaks.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8", errors="replace") as lf:
        for line in iter_output_lines(proc.stdout):
            print(line, flush=True)
            lf.write(line + "\n")
            lf.flush()
            tail.append(line)
    return proc.wait(), list(tail)


def spec_hyperparameters(job: dict) -> dict:
    """The hyperparameters the job spec carries, or a loud refusal.

    The control plane resolves every value before launch, so anything missing
    here is a hole in the spec, not an occasion to pick a number. Naming the
    missing keys is the difference between a fixable refusal and a guess.
    """
    hp = job.get("hyperparameters")
    if not isinstance(hp, dict) or not hp:
        raise IncompleteJobSpec(
            "job specification carries no hyperparameters; the trainer "
            "resolves nothing and was given nothing"
        )
    missing = sorted(k for k in REQUIRED_HYPERPARAMETERS if k not in hp)
    if missing:
        raise IncompleteJobSpec(
            "job specification is missing required hyperparameters "
            f"{missing}; the trainer resolves nothing and will not fall "
            "back to defaults it no longer has"
        )
    return hp


def build_config(
    job: dict, enable_thinking: bool = False
) -> tuple[dict, dict]:
    """Job spec -> Axolotl config. Returns (config, rejected).

    Every hyperparameter is read from the spec exactly as given. Derivations
    that used to live here -- alpha tracking rank, rsLoRA above rank 32 --
    happen in the control plane's resolver before launch; redoing them would
    be the second resolver again.
    """
    hp = spec_hyperparameters(job)
    rejected: dict = {}

    # Validate the top level before anything else.
    unknown_top = sorted(
        k
        for k in job
        if k not in ALLOWED_JOB_KEYS and not k.startswith("_comment")
    )
    for k in unknown_top:
        rejected[k] = job[k]
    unknown_hp = sorted(k for k in hp if k not in KNOWN_HYPERPARAMETERS)
    for k in unknown_hp:
        rejected[k] = hp[k]

    cfg = {k: v for k, v in hp.items() if k in KNOWN_HYPERPARAMETERS}

    cfg.update(
        {
            "base_model": job["base_model"],
            "output_dir": str(OUT_DIR / "run"),
            # --- method: QLoRA. NF4 + double-quant, bf16 compute -------------------
            # This said "adapters in bf16" and that was wrong: measured from the
            # first real run's safetensors header, all 504 adapter tensors are F32,
            # which is why the artifact is 132 MB rather than ~66 MB. bf16 is the
            # COMPUTE dtype; trainable parameters are kept in fp32 under 4-bit
            # quantisation, which is standard and correct. The claim was the bug,
            # not the behaviour.
            "adapter": "qlora",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            # --- targets: ALL linear, not attention-only -------------------------
            # The MLP is ~78% of every transformer block's parameters, so
            # attention-only leaves most of the model untouched at any rank.
            "lora_target_linear": True,
            # --- precision: bf16, same exponent range as fp32, no loss scaler ----
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": True,
            "flash_attention": True,
            "seed": 42,
            # --- data ------------------------------------------------------------
            "datasets": [
                {
                    "path": str(JOB_DIR / "dataset.jsonl"),
                    "type": "chat_template",
                    "field_messages": job.get("messages_field", "messages"),
                }
            ],
            # Let the model's own template decide. Hand-writing role delimiters is
            # the modal production bug in this category.
            "chat_template": "tokenizer_default",
            # Detected from the dataset, never guessed and never exposed. Qwen3
            # emits <think> blocks through its template by default; training data
            # without them under that template is a silent train/infer mismatch.
            # The SAME value must be applied at serving.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            # Train on the assistant turn only. Full-sequence loss on instruction
            # data teaches the model to recite prompts back.
            "train_on_inputs": False,
            # Packing is a throughput win but needs varlen attention to avoid
            # cross-example contamination. Off until measured per model.
            "sample_packing": False,
            "logging_steps": 1,
            "save_safetensors": True,  # never torch .bin ΓÇö see spike 3 / C14
            # save_total_limit is no longer set here: it now arrives resolved
            # in `hp`, from the same contract `temper_core.disk` reads to
            # size the machine's disk (issue #64) -- a value both must agree
            # on is defined once, in packages/contracts/trainer-defaults.json.
        }
    )

    if job.get("max_steps"):
        cfg["max_steps"] = int(job["max_steps"])
    if job.get("save_steps"):
        cfg["save_steps"] = int(job["save_steps"])
    if job.get("resume_from_checkpoint"):
        cfg["resume_from_checkpoint"] = job["resume_from_checkpoint"]

    return cfg, rejected


def _checkpoint_step(path: Path) -> int:
    """Step number from a `checkpoint-N` directory name, -1 if unparseable."""
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def collect_artifacts() -> dict:
    """Find what training produced, and fingerprint it.

    Which adapter gets shipped is not a detail: it is the entire deliverable,
    and "which weights did I actually download?" is a question the product has
    to answer exactly. Two ways to get it wrong were both present here:

    * `sorted(rglob(...))[-1]` sorts lexicographically, so once a run produces
      ten checkpoints it picks `checkpoint-9` over `checkpoint-10`.
    * It also preferred a checkpoint over the final adapter Axolotl writes at
      the top of the output directory at the end of training, because
      `run/checkpoint-N/...` sorts after `run/adapter_model.safetensors`.

    The rule is explicit instead: the end-of-training adapter wins; failing
    that, the numerically highest checkpoint. `adapter_source` records which,
    so the answer is in result.json rather than inferred.
    """
    run = OUT_DIR / "run"
    ckpt_dirs = sorted(
        (p for p in run.glob("checkpoint-*") if p.is_dir()),
        key=_checkpoint_step,
    )
    info: dict = {"checkpoints": [p.name for p in ckpt_dirs]}

    # Search order, most authoritative first: the final adapter, then
    # checkpoints from the highest step down.
    search_dirs = [run, *reversed(ckpt_dirs)]
    adapter = None
    for d in search_dirs:
        for cand in ("adapter_model.safetensors", "adapter_model.bin"):
            if (d / cand).is_file():
                adapter = d / cand
                break
        if adapter:
            break

    if adapter:
        raw = adapter.read_bytes()
        info.update(
            {
                "adapter_path": str(adapter.relative_to(OUT_DIR)),
                "adapter_bytes": len(raw),
                "adapter_sha256": hashlib.sha256(raw).hexdigest(),
                "adapter_format": adapter.suffix.lstrip("."),
                "adapter_source": (
                    "final" if adapter.parent == run else adapter.parent.name
                ),
            }
        )
        # The config that sits WITH the chosen adapter, not whichever one
        # happened to sort last. A LoRA is not loadable without it, so the
        # orchestrator ships it alongside the weights.
        cfg_path = adapter.parent / "adapter_config.json"
        if cfg_path.is_file():
            info["adapter_config"] = json.loads(cfg_path.read_text())
    return info


def main() -> int:
    started = time.time()
    result: dict = {"ok": False, "started_at": time.time()}
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        job = json.loads((JOB_DIR / "job.json").read_text())
        result["job_id"] = job.get("job_id")
        result["base_model"] = job.get("base_model")
        result["base_revision"] = job.get("base_revision")
        log(
            f"job {job.get('job_id')} ΓÇö base_model={job.get('base_model')}"
            f"@{job.get('base_revision') or 'unpinned'}"
        )

        ds = JOB_DIR / "dataset.jsonl"
        if not ds.exists():
            raise FileNotFoundError(f"dataset not found at {ds}")
        parsed = [json.loads(line) for line in ds.open() if line.strip()]
        rows = len(parsed)
        result["dataset_rows"] = rows
        # 10 is the hard floor adopted from OpenAI's enforced minimum; below it
        # a run cannot produce a meaningful adapter, so block rather than waste
        # the GPU.
        if rows < 10:
            raise ValueError(f"dataset has {rows} rows; minimum is 10")
        log(f"dataset: {rows} rows")

        # Decide thinking mode from the data before building the config.
        # A mixed dataset is ambiguous by construction, so it blocks here
        # rather than training half the rows against the wrong template.
        try:
            think = detect_thinking(
                parsed, job.get("messages_field", "messages")
            )
        except MixedThinkingDataset:
            result["error_code"] = "dataset_mixed_thinking"
            raise
        result["thinking"] = think.as_dict()
        log(
            f"thinking mode: {think.enable_thinking} "
            f"({think.with_think}/{think.assistant_turns} assistant turns have <think>)"
        )

        # An incomplete spec must fail as a named refusal here rather than as
        # a training anomaly minutes into a paid machine.
        try:
            cfg, rejected = build_config(
                job, enable_thinking=think.enable_thinking
            )
        except IncompleteJobSpec:
            result["error_code"] = "spec_incomplete"
            raise
        # Recorded filtered to keys this trainer knows: the record should name
        # only values that were trained with, and anything else is already
        # echoed verbatim under rejected_overrides.
        result["hyperparameters"] = {
            k: v
            for k, v in (job.get("hyperparameters") or {}).items()
            if k in KNOWN_HYPERPARAMETERS
        }
        if rejected:
            # Not silently dropped: a key the caller sent is something the
            # caller believes is in effect.
            result["rejected_overrides"] = rejected
            log(f"REJECTED unknown keys: {list(rejected)}")

        import yaml  # provided by the base image

        CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=True))
        result["config"] = cfg
        log(f"config written to {CONFIG}")

        cmd = ["axolotl", "train", str(CONFIG)]
        log(f"running: {' '.join(cmd)}")
        t0 = time.time()
        code, tail = run_streaming(cmd)
        result["train_seconds"] = round(time.time() - t0, 1)
        result["exit_code"] = code

        if code != 0:
            result["error"] = "axolotl train failed"
            # The tail is kept in result.json even though every one of these
            # lines was already streamed: the result document is what the
            # orchestrator reads, and it should never have to go back and
            # reassemble a failure out of the event log.
            result["log_tail"] = tail
            log(f"training FAILED (exit {code})")
            return code

        result.update(collect_artifacts())
        result["ok"] = True
        log(f"training complete in {result['train_seconds']}s")
        return 0

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        log(f"ERROR: {result['error']}")
        return 1
    finally:
        result["total_seconds"] = round(time.time() - started, 1)
        write_result(result)
        log(f"result written to {RESULT}")


if __name__ == "__main__":
    sys.exit(main())
