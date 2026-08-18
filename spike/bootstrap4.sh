#!/bin/bash
# Spike 4 — build the pinned trainer image on the VM, then run a real job
# through it and prove resume works.
#
# Runs ON the VM, piped in over SSH. The trainer sources arrive first as a tar
# on stdin? No -- they are written by spike4.py before this runs, into /tmp/trainer.
#
# Emits JSON on stdout, progress on stderr. Never exits non-zero.

set -u
say() { echo "[$(date +%H:%M:%S)] $*" >&2; }
esc() { printf '%s' "$1" | tr -d '\r' | sed 's/\/\\/g; s/"/\\"/g' | tr '\n' ' '; }

SRC=/tmp/trainer
JOB=/tmp/job
OUT=/tmp/out
IMG=finetune-trainer:spike4
MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"

rm -rf "$JOB" "$OUT"; mkdir -p "$JOB" "$OUT"

# --- security first: nothing this container does needs an inbound port -------
# Spike 3 established that ufw does not filter Docker-published ports and that
# a DOCKER-USER rule matched on the published port never fires. The trainer
# needs no inbound port at all, so the mitigation is simply to publish none.
sudo ufw allow 22/tcp >/dev/null 2>&1
sudo ufw default deny incoming >/dev/null 2>&1
sudo ufw --force enable >/dev/null 2>&1
say "ufw: $(sudo ufw status 2>/dev/null | head -1) (trainer publishes no ports)"

# --- build -------------------------------------------------------------------
say "building $IMG from pinned base (8.5 GB pull, be patient) ..."
t0=$(date +%s)
if sudo docker build -t "$IMG" "$SRC" >/tmp/build.log 2>&1; then
  build_ok=true
else
  build_ok=false
  say "--- build failed, last 20 lines ---"
  tail -n 20 /tmp/build.log >&2
fi
build_s=$(( $(date +%s) - t0 ))
say "build: ok=$build_ok in ${build_s}s"

base_digest=$(sudo docker image inspect "$IMG" \
  --format '{{index .Config.Labels "com.jarvislabs-finetune.base-digest"}}' 2>/dev/null || echo "")
img_size=$(sudo docker image inspect "$IMG" --format='{{.Size}}' 2>/dev/null || echo 0)

# --- job spec + dataset ------------------------------------------------------
python3 - <<'PY'
import json, random
random.seed(42)
rows = []
for i in range(64):
    a, b = random.randint(1, 50), random.randint(1, 50)
    rows.append({"messages": [
        {"role": "user", "content": f"What is {a} plus {b}?"},
        {"role": "assistant", "content": f"{a} + {b} = {a+b}."},
    ]})
with open("/tmp/job/dataset.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
PY

cat > "$JOB/job.json" <<JOBEOF
{
  "job_id": "spike4-0001",
  "base_model": "$MODEL",
  "messages_field": "messages",
  "hyperparameters": {"lora_r": 16, "learning_rate": 2e-4, "save_steps": 2},
  "max_steps": 4,
  "save_steps": 2,
  "not_a_real_key": "should be rejected"
}
JOBEOF
say "job spec + 64-row dataset written"

# --- run 1: train ------------------------------------------------------------
run_container() {
  sudo docker run --rm --gpus all \
    -v "$JOB":/job:ro -v "$OUT":/out \
    -e HF_HOME=/out/hf \
    "$IMG" 2>&1
}

if [ "$build_ok" = true ]; then
  say "run 1: training $MODEL, max_steps=4, save_steps=2 ..."
  t0=$(date +%s)
  run_container >/tmp/run1.log 2>&1 && run1_ok=true || run1_ok=false
  run1_s=$(( $(date +%s) - t0 ))
  say "run 1: ok=$run1_ok in ${run1_s}s"
  [ "$run1_ok" = false ] && say "--- run1 tail ---" && tail -n 25 /tmp/run1.log >&2
else
  run1_ok=false; run1_s=0
fi

result1="{}"
[ -f "$OUT/result.json" ] && result1=$(sudo cat "$OUT/result.json")

# --- run 2: resume -----------------------------------------------------------
# The criterion in section 31 is that a run survives interruption. Resume from
# checkpoint-2 and drive to 6.
resume_ok=false
result2="{}"
ckpt=$(sudo ls -d "$OUT"/run/checkpoint-2 2>/dev/null || echo "")
if [ "$run1_ok" = true ] && [ -n "$ckpt" ]; then
  sudo python3 - <<'PY'
import json
p = "/tmp/job/job.json"
j = json.load(open(p))
j["max_steps"] = 6
j["resume_from_checkpoint"] = "/out/run/checkpoint-2"
json.dump(j, open(p, "w"), indent=2)
PY
  say "run 2: resuming from checkpoint-2 -> max_steps=6 ..."
  t0=$(date +%s)
  run_container >/tmp/run2.log 2>&1 && resume_ok=true || resume_ok=false
  run2_s=$(( $(date +%s) - t0 ))
  say "run 2: ok=$resume_ok in ${run2_s}s"
  [ "$resume_ok" = false ] && say "--- run2 tail ---" && tail -n 25 /tmp/run2.log >&2
  [ -f "$OUT/result.json" ] && result2=$(sudo cat "$OUT/result.json")
else
  run2_s=0
  say "run 2 skipped (no checkpoint-2)"
fi

ckpts=$(sudo ls "$OUT/run" 2>/dev/null | grep -c '^checkpoint-' || echo 0)

cat <<JSON
{
  "build_ok": $build_ok,
  "build_seconds": $build_s,
  "image_bytes": ${img_size:-0},
  "base_digest_label": "$(esc "$base_digest")",
  "run1_ok": $run1_ok,
  "run1_seconds": $run1_s,
  "resume_ok": $resume_ok,
  "run2_seconds": ${run2_s:-0},
  "checkpoint_count": $ckpts,
  "result_run1": $result1,
  "result_run2": $result2
}
JSON
