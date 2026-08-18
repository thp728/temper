#!/bin/bash
# Spike 3 — firewall, a real QLoRA step, checkpoint/resume, adapter integrity.
#
# Piped in over SSH like spike 2. Emits JSON on stdout, progress on stderr.
# Never exits non-zero: a failed step is a result.

set -u
say() { echo "[$(date +%H:%M:%S)] $*" >&2; }
esc() { printf '%s' "$1" | tr -d '\r' | sed 's/\\/\\\\/g; s/"/\\"/g' | tr '\n' ' '; }

IMG="${TRAIN_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
PORT="${TEST_PORT:-8000}"
OUT=/tmp/spike3
mkdir -p "$OUT"

# ============================================================================
# PHASE A — close C12. VMs ship with a public IP and ufw inactive.
# ============================================================================
# ORDER IS LOAD-BEARING: allow 22 BEFORE enabling, or you lock yourself out of
# a box you are paying for. Belt and braces on top of that -- a dead-man switch
# disables the firewall in 10 minutes if anything goes wrong, so a mistake
# costs a wait rather than the instance.
say "arming dead-man switch (ufw disable in 600s)"
sudo nohup bash -c 'sleep 600 && ufw --force disable' >/dev/null 2>&1 &

say "firewall: allow 22 first, then default-deny inbound"
sudo ufw allow 22/tcp           >/dev/null 2>&1
sudo ufw default deny incoming  >/dev/null 2>&1
sudo ufw default allow outgoing >/dev/null 2>&1
sudo ufw --force enable         >/dev/null 2>&1
ufw_after=$(sudo ufw status 2>/dev/null | head -1)
say "ufw now: $ufw_after"

# UFW ALONE IS NOT ENOUGH, and the first spike-3 attempt proved it: ufw was
# active with default-deny and the published port was STILL reachable from the
# internet.
#
# Why: ufw writes rules on the INPUT chain (traffic terminating at the host).
# Docker publishes ports by writing NAT/FORWARD rules, so packets to a
# container never traverse INPUT and ufw never sees them. This is documented
# Docker behaviour, not a bug -- which makes "enable the firewall" a mitigation
# that looks correct, reports Status: active, and protects nothing.
#
# Docker provides the DOCKER-USER chain for exactly this, evaluated before its
# own rules. Test both mitigations:
EXT_IF=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $5; exit}')
say "external interface: ${EXT_IF:-unknown}"
if [ -n "${EXT_IF:-}" ]; then
  sudo iptables -I DOCKER-USER -i "$EXT_IF" -p tcp --dport ${PORT} -j DROP 2>/dev/null \
    && docker_user_rule=true || docker_user_rule=false
else
  docker_user_rule=false
fi
say "DOCKER-USER drop rule installed: $docker_user_rule"

# Mitigation A -- published port, protected only by DOCKER-USER.
sudo docker rm -f porttest >/dev/null 2>&1
sudo docker run --rm -d --name porttest -p ${PORT}:80 \
  ghcr.io/linuxcontainers/alpine:latest \
  sh -c "while true; do printf 'HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\nSPIKE3\n' | nc -l -p 80; done" \
  >/dev/null 2>&1

# Mitigation B -- bind to loopback only. Nothing is published to the outside
# world at all, so there is no firewall rule to get wrong. This is the one to
# prefer for anything that does not genuinely need public reachability.
sudo docker rm -f porttest_local >/dev/null 2>&1
sudo docker run --rm -d --name porttest_local -p 127.0.0.1:$((PORT+1)):80 \
  ghcr.io/linuxcontainers/alpine:latest \
  sh -c "while true; do printf 'HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nLOCAL\n' | nc -l -p 80; done" \
  >/dev/null 2>&1

sleep 3
listening=$(sudo ss -lntp 2>/dev/null | grep -c ":${PORT}" || echo 0)
local_ok=$(curl -fsS -m 5 "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 && echo true || echo false)
loopback_ok=$(curl -fsS -m 5 "http://127.0.0.1:$((PORT+1))/" >/dev/null 2>&1 && echo true || echo false)
say "listener up: $listening | localhost:$PORT=$local_ok | localhost:$((PORT+1))=$loopback_ok"

# ============================================================================
# PHASE B — install the fine-tuning stack, timed.
# ============================================================================
# The base image has torch only. Timing this is the evidence for whether a
# purpose-built image is worth the build pipeline, or whether install-at-boot
# is tolerable. Run inside a NAMED container we keep, so phases C/D share it.
say "starting work container"
sudo docker rm -f trainer >/dev/null 2>&1
sudo docker run -d --name trainer --gpus all \
  -v "$OUT":/out -e HF_HOME=/out/hf \
  "$IMG" sleep infinity >/dev/null 2>&1

say "installing transformers/peft/trl/bitsandbytes/datasets/accelerate ..."
t0=$(date +%s)
sudo docker exec trainer pip install -q --no-cache-dir \
  "transformers>=4.51" peft trl bitsandbytes datasets accelerate >/dev/null 2>&1 \
  && pip_ok=true || pip_ok=false
pip_s=$(( $(date +%s) - t0 ))
say "install done in ${pip_s}s (ok=$pip_ok)"

vers=$(sudo docker exec trainer python -c "
import transformers, peft, trl, bitsandbytes, torch
print(f'transformers={transformers.__version__} peft={peft.__version__} trl={trl.__version__} bnb={bitsandbytes.__version__} torch={torch.__version__}')
" 2>/dev/null | tr -d '\r\n' || echo "")
say "versions: $vers"

# ============================================================================
# PHASE C/D — the training script (writes its own JSON result)
# ============================================================================
cat > "$OUT/train.py" <<'PYEOF'
import json, os, time, hashlib, glob
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL = os.environ["BASE_MODEL"]
OUT = "/out/run"
res = {"model": MODEL}

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
res["load_seconds"] = round(time.time() - t0)
res["vram_after_load_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)

# Tiny but real instruction dataset.
rows = [{"messages": [{"role": "user", "content": f"What is {i}+{i}?"},
                      {"role": "assistant", "content": str(i + i)}]}
        for i in range(64)]
ds = Dataset.from_list(rows)

peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")

# TRL 1.x restructured SFTConfig: it no longer inherits TrainingArguments, so
# arguments that worked for years (warmup_ratio, save_steps, bf16, seed...) now
# raise TypeError. Rather than hard-code a guess at the new surface, ask the
# installed class what it accepts and report what got dropped. The dropped list
# IS a finding -- it tells us exactly which of our intended defaults this
# version cannot express, which is what a pinned image has to solve.
import dataclasses
_valid = {f.name for f in dataclasses.fields(SFTConfig)}
_wanted = dict(
    output_dir=OUT, save_steps=2, per_device_train_batch_size=1,
    gradient_accumulation_steps=2, learning_rate=2e-4,
    lr_scheduler_type="cosine", warmup_ratio=0.1, logging_steps=1,
    bf16=True, max_length=256, report_to=[], seed=42,
    gradient_checkpointing=True, save_safetensors=True,
)
res["sftconfig_fields"] = sorted(_valid)
res["sftconfig_dropped"] = sorted(k for k in _wanted if k not in _valid)

def make(max_steps, resume):
    kw = {k: v for k, v in _wanted.items() if k in _valid}
    if "max_steps" in _valid:
        kw["max_steps"] = max_steps
    cfg = SFTConfig(**kw)
    # max_steps may live on the trainer/TrainingArguments side now; set it
    # post-hoc if the config would not take it, so the run stays short.
    if "max_steps" not in kw:
        try:
            cfg.max_steps = max_steps
        except Exception:
            pass
    return SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg)

# --- Phase C: train 4 steps, checkpointing every 2 -------------------------
t0 = time.time()
tr = make(4, False)
out = tr.train()
res["train_seconds"] = round(time.time() - t0, 1)
res["steps_completed"] = int(out.global_step)
res["train_loss"] = round(float(out.training_loss), 4)
res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

# tokens/sec -- calibrates the MFU constant, currently a 35-50% planning band
hist = [h for h in tr.state.log_history if "loss" in h]
res["logged_steps"] = len(hist)
tps = tr.state.log_history[-1].get("train_tokens_per_second")
if tps: res["tokens_per_second"] = round(float(tps), 1)

cks = sorted(glob.glob(f"{OUT}/checkpoint-*"))
res["checkpoints"] = [os.path.basename(c) for c in cks]

# --- Phase D: resume from an earlier checkpoint ----------------------------
# Section 31 requires a run to survive interruption. Resume from checkpoint-2
# and continue to 6: proves step counter and optimizer state restore.
res["resume_ok"] = False
if cks:
    try:
        ck = [c for c in cks if c.endswith("-2")] or [cks[0]]
        tr2 = make(6, True)
        out2 = tr2.train(resume_from_checkpoint=ck[0])
        res["resume_from"] = os.path.basename(ck[0])
        res["resume_final_step"] = int(out2.global_step)
        res["resume_ok"] = int(out2.global_step) == 6
    except Exception as e:
        res["resume_error"] = f"{type(e).__name__}: {e}"[:300]

# --- adapter integrity ------------------------------------------------------
tr.model.save_pretrained("/out/adapter")
files = sorted(glob.glob("/out/adapter/*"))
res["adapter_files"] = [os.path.basename(f) for f in files]
w = "/out/adapter/adapter_model.safetensors"
if os.path.exists(w):
    res["adapter_bytes"] = os.path.getsize(w)
    res["adapter_sha256"] = hashlib.sha256(open(w, "rb").read()).hexdigest()

trainable = sum(p.numel() for p in tr.model.parameters() if p.requires_grad)
res["trainable_params"] = trainable
res["trainable_pct"] = round(100 * trainable / sum(p.numel() for p in tr.model.parameters()), 4)

json.dump(res, open("/out/train-result.json", "w"), indent=2)
print("TRAIN_OK")
PYEOF

say "running QLoRA on $MODEL (downloads weights first) ..."
t0=$(date +%s)
sudo docker exec -e BASE_MODEL="$MODEL" trainer python /out/train.py >"$OUT/train.log" 2>&1 \
  && train_ok=true || train_ok=false
train_s=$(( $(date +%s) - t0 ))
say "training phase done in ${train_s}s (ok=$train_ok)"
[ "$train_ok" = false ] && say "--- last 15 log lines ---" && tail -n 15 "$OUT/train.log" >&2

result_json="{}"
[ -f "$OUT/train-result.json" ] && result_json=$(cat "$OUT/train-result.json")

# host-side SHA of the adapter, to compare against what the container computed
host_sha=""
[ -f "$OUT/adapter/adapter_model.safetensors" ] && \
  host_sha=$(sha256sum "$OUT/adapter/adapter_model.safetensors" | cut -d' ' -f1)

sudo docker rm -f trainer >/dev/null 2>&1

cat <<JSON
{
  "ufw_status": "$(esc "$ufw_after")",
  "external_interface": "$(esc "${EXT_IF:-}")",
  "docker_user_rule_installed": $docker_user_rule,
  "port_listening_locally": $listening,
  "reachable_from_localhost": $local_ok,
  "loopback_bound_port_ok": $loopback_ok,
  "test_port": $PORT,
  "pip_install_ok": $pip_ok,
  "pip_install_seconds": $pip_s,
  "versions": "$(esc "$vers")",
  "train_ok": $train_ok,
  "train_phase_seconds": $train_s,
  "host_adapter_sha256": "$(esc "$host_sha")",
  "train_result": $result_json
}
JSON
