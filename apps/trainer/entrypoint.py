"""Trainer entrypoint — job spec in, adapter + result out.

Runs inside the pinned image. The orchestrator writes a job spec and a dataset
to /job, runs this container, and reads /out.

    /job/job.json       the job spec (see job.example.json)
    /job/dataset.jsonl  training data, one JSON object per line
    /out/               config.yaml, checkpoints, adapter/, result.json, train.log

Design notes worth keeping, because each is a decision:

* **Axolotl owns the training loop, we own the contract.** We render YAML and
  invoke `axolotl train`. We do not call TRL/PEFT directly — spike 3 showed
  those APIs move underneath you (TRL 1.x dropped `warmup_ratio` from
  SFTConfig), while Axolotl's config surface stayed stable and still exposes it.
* **Correctness settings are not user-settable.** Chat template resolution,
  EOS handling and loss masking are the highest-frequency silent-failure
  surface: they pass every obvious health check and only show up as garbage
  generations. They are decided here, not exposed.
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

JOB_DIR = Path(os.environ.get("JOB_DIR", "/job"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/out"))
CONFIG = OUT_DIR / "config.yaml"
RESULT = OUT_DIR / "result.json"
LOG = OUT_DIR / "train.log"

# Defaults from the research, and the reasoning lives in the wiki rather than
# here. Anything a user may override is in ALLOWED_OVERRIDES; anything absent
# from that set is a correctness decision and is deliberately not settable.
DEFAULTS = {
    "lora_r": 16,
    "lora_alpha": 32,  # α = 2r; recompute if r changes
    "lora_dropout": 0.0,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 8,  # effective batch 8; see wiki
    "sequence_len": 2048,
    "warmup_ratio": 0.1,
    "lr_scheduler": "cosine",
    "val_set_size": 0.05,
}
ALLOWED_OVERRIDES = {
    "lora_r",
    "lora_alpha",
    "learning_rate",
    "num_epochs",
    "max_steps",
    "sequence_len",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "val_set_size",
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
    "messages_field",
    "hyperparameters",
    "max_steps",
    "save_steps",
    "resume_from_checkpoint",
}


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

    * **Progress bars never send a newline.** tqdm — which transformers uses
      for every epoch — redraws with a carriage return. A reader that waits
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

    * `bufsize=0` so the pipe is read raw — a buffered reader waits to fill a
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


def build_config(job: dict, enable_thinking: bool = False) -> dict:
    """Job spec -> Axolotl config. Returns the config dict."""
    cfg = dict(DEFAULTS)
    applied, rejected = {}, {}

    # Validate the top level before anything else.
    unknown_top = sorted(
        k
        for k in job
        if k not in ALLOWED_JOB_KEYS and not k.startswith("_comment")
    )
    for k in unknown_top:
        rejected[k] = job[k]
    for k, v in (job.get("hyperparameters") or {}).items():
        if k in ALLOWED_OVERRIDES:
            cfg[k] = v
            applied[k] = v
        else:
            rejected[k] = v

    # α is mechanically tied to r. If the caller moved r but not α, recompute
    # rather than silently pairing a new rank with a stale scale.
    if "lora_r" in applied and "lora_alpha" not in applied:
        cfg["lora_alpha"] = 2 * int(cfg["lora_r"])

    # rsLoRA above rank 32: plain α/r scaling over-shrinks high-rank adapters
    # and training destabilises. Inferred from rank, never exposed.
    cfg["lora_use_rslora"] = int(cfg["lora_r"]) >= 32

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
            "save_safetensors": True,  # never torch .bin — see spike 3 / C14
            "save_total_limit": 3,
        }
    )

    if job.get("max_steps"):
        cfg["max_steps"] = int(job["max_steps"])
    if job.get("save_steps"):
        cfg["save_steps"] = int(job["save_steps"])
    if job.get("resume_from_checkpoint"):
        cfg["resume_from_checkpoint"] = job["resume_from_checkpoint"]

    return cfg, applied, rejected


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
        log(f"job {job.get('job_id')} — base_model={job.get('base_model')}")

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

        cfg, applied, rejected = build_config(
            job, enable_thinking=think.enable_thinking
        )
        result["applied_overrides"] = applied
        if rejected:
            # Not silently dropped: an override we refuse is something the
            # caller believes is in effect.
            result["rejected_overrides"] = rejected
            log(f"REJECTED non-overridable keys: {list(rejected)}")

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
