#!/usr/bin/env bash
# Runs ON the VM for spike 6. Piped in over SSH; prints JSON on stdout only.
#
# Three claims in sequence, and only the first had any evidence before today:
#   1. num_gpus=2 attaches two devices to the machine.
#   2. The PINNED image sees both of them. The image was built and digest-locked
#      before multi-GPU was in scope, so nothing asserts the toolkit passes more
#      than one device through.
#   3. Axolotl's FSDP FULL_SHARD launches under accelerate in that image, takes
#      steps, writes a checkpoint, and resumes from it.
#
# TWO MODELS, deliberately, because the first attempt could not tell two
# different failures apart. Qwen3-4B full fine-tune across 2x24 GB needs roughly
# 17 GB per device before activations, so a failure there is ambiguous: broken
# mechanism, or a model that does not fit? The small case answers the mechanism
# question with room to spare; the large case is then a CAPACITY data point and
# is read as one.
#
# sudo on EVERY docker call. The ubuntu user on a --vm instance is not in the
# docker group, so a bare `docker` returns "permission denied ... unix:///var/
# run/docker.sock". bootstrap4.sh already knew this and the knowledge did not
# carry over -- which cost a run and very nearly cost a WRONG FINDING: the first
# pass reported "FSDP does not run in the pinned image" when the probe had never
# reached the image. That is the C16 mistake, a tooling failure filed as a
# platform one.

set -uo pipefail

IMAGE="${1:?image digest required}"
SMALL_MODEL="${2:-Qwen/Qwen3-0.6B}"
LARGE_MODEL="${3:-Qwen/Qwen3-4B}"
STEPS="${4:-3}"

log() { echo "[bootstrap6] $*" >&2; }
esc() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }

WORK=/tmp/spike6
mkdir -p "$WORK/data" "$WORK/hf"

# --- 1. devices on the host -------------------------------------------------
HOST_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
HOST_GPU_LIST=$(nvidia-smi -L 2>/dev/null | tr -s ' ')
log "host reports $HOST_GPUS device(s)"

# --- 2. devices inside the pinned image -------------------------------------
log "pulling $IMAGE"
PULL_START=$(date +%s)
sudo docker pull "$IMAGE" >/dev/null 2>"$WORK/pull.err"
PULL_RC=$?
PULL_SECONDS=$(( $(date +%s) - PULL_START ))

CONTAINER_GPUS=-1
TORCH_REPORT=""
if [ "$PULL_RC" -eq 0 ]; then
  TORCH_REPORT=$(sudo docker run --rm --gpus all "$IMAGE" python -c '
import json, torch
print(json.dumps({
    "device_count": torch.cuda.device_count(),
    "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "total_vram_bytes": [torch.cuda.get_device_properties(i).total_memory
                         for i in range(torch.cuda.device_count())],
    "torch": torch.__version__,
    "nccl_available": torch.distributed.is_nccl_available(),
}))' 2>"$WORK/torch.err")
  CONTAINER_GPUS=$(echo "$TORCH_REPORT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["device_count"])' 2>/dev/null || echo -1)
fi
log "container reports $CONTAINER_GPUS device(s)"

# Stop here if the container saw no devices. `accelerate launch
# --num_processes -1` produces an error that reads like an FSDP problem, and
# reporting it as one would file a device-visibility failure under the wrong
# heading. **A probe that could not run is not evidence about what it would
# have found.**
if [ "$CONTAINER_GPUS" -lt 1 ]; then
  log "container reported $CONTAINER_GPUS device(s) -- NOT launching FSDP"
  cat <<JSON
{
  "host_gpu_count": ${HOST_GPUS:-0},
  "host_gpu_list": $(echo "$HOST_GPU_LIST" | esc),
  "image": "$IMAGE",
  "image_pull_ok": $([ "$PULL_RC" -eq 0 ] && echo true || echo false),
  "image_pull_seconds": $PULL_SECONDS,
  "container_torch_report": $(echo "$TORCH_REPORT" | esc),
  "container_torch_stderr": $(tail -c 2000 "$WORK/torch.err" 2>/dev/null | esc),
  "container_gpu_count": ${CONTAINER_GPUS:--1},
  "fsdp_attempted": false,
  "not_attempted_reason": "the container reported no devices, so there was nothing to shard across and FSDP was never launched. NOTHING HERE IS EVIDENCE ABOUT FSDP.",
  "cases": []
}
JSON
  exit 0
fi

# --- 3. a tiny dataset ------------------------------------------------------
python3 - "$WORK/data/train.jsonl" <<'PY'
import json, sys
rows = [{"messages": [
    {"role": "user", "content": f"Say hello to customer {i}."},
    {"role": "assistant", "content": f"Hello, customer {i}. How can I help today?"},
]} for i in range(64)]
with open(sys.argv[1], "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
PY

write_config() {
  # $1 = model, $2 = output dir inside the container, $3 = max_steps, $4 = wrap class
  cat > "$WORK/fsdp.yaml" <<YAML
base_model: $1
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

datasets:
  - path: /job/data/train.jsonl
    type: chat_template
chat_template: tokenizer_default

dataset_prepared_path: /job/prepared
output_dir: $2

sequence_len: 512
sample_packing: false
pad_to_sequence_len: true

micro_batch_size: 1
gradient_accumulation_steps: 1
max_steps: $3
learning_rate: 1e-5
optimizer: adamw_torch
lr_scheduler: constant

bf16: true
gradient_checkpointing: true
flash_attention: false

save_steps: 2
save_total_limit: 4
logging_steps: 1
seed: 42

fsdp_version: 2
fsdp_config:
  offload_params: false
  state_dict_type: SHARDED_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: $4
  reshard_after_forward: true
YAML
}

run_axolotl() {
  # --gpus all and nothing else. NOTHING IS PUBLISHED: ufw does not filter
  # Docker's published ports and a DOCKER-USER rule matched on the published
  # port never fires, so not publishing is the only mitigation that holds.
  #
  # --tee 3 is not cosmetic. Without it torchrun swallows the child processes'
  # stderr and reports only "ChildFailedError ... exitcode 1", which is what
  # the previous run recorded -- a failure with no cause. The cause is the
  # whole point.
  sudo docker run --rm --gpus all --shm-size=8g \
    -v "$WORK:/job" -v "$WORK/out:/out" \
    -e HF_HOME=/job/hf \
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --entrypoint bash "$IMAGE" -lc "$1" 2>&1
}

# Sample per-device VRAM while training runs. The predictor's sharding
# arithmetic has ZERO real anchors today; this gives it one. Sampling
# nvidia-smi rather than reading the training log because the log only reports
# it if the trainer chose to, and the previous run's log did not.
start_vram_sampler() {
  ( while true; do
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | tr '\n' ';' >> "$WORK/vram.samples"
      echo >> "$WORK/vram.samples"
      sleep 2
    done ) &
  VRAM_PID=$!
}
stop_vram_sampler() { kill "$VRAM_PID" 2>/dev/null; wait "$VRAM_PID" 2>/dev/null; }

peak_vram_json() {
  python3 - "$WORK/vram.samples" <<'PY'
import json, sys, collections
peak = collections.defaultdict(int)
try:
    for line in open(sys.argv[1]):
        for entry in line.strip().strip(';').split(';'):
            if not entry.strip():
                continue
            idx, mib = (x.strip() for x in entry.split(','))
            peak[int(idx)] = max(peak[int(idx)], int(mib))
except OSError:
    pass
print(json.dumps({f"gpu{k}_peak_mib": v for k, v in sorted(peak.items())}))
PY
}

# Steps come from trainer_state.json, not from grepping the log. The previous
# run reported "0 steps" while a checkpoint sat on disk, because the log format
# did not match the pattern -- and "0 steps" next to "a checkpoint was written"
# is a self-contradicting finding.
state_steps() {
  python3 - "$1" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1] + "/trainer_state.json"))["global_step"])
except Exception:
    print(0)
PY
}

CASES_JSON=""

run_case() {
  local tag="$1" model="$2" wrap="$3"
  local outdir="$WORK/out-$tag"
  rm -rf "$outdir" "$WORK/prepared" "$WORK/vram.samples"
  mkdir -p "$outdir"

  log "=== case $tag: $model ==="
  write_config "$model" "/out/run" "$STEPS" "$wrap"
  # /out is bound per case so the checkpoints do not mix.
  local saved_work_out="$WORK/out"
  rm -rf "$WORK/out"; ln -sfn "$outdir" "$WORK/out"

  start_vram_sampler
  local t0=$(date +%s)
  local train_log
  train_log=$(run_axolotl "accelerate launch --num_processes $CONTAINER_GPUS --use_fsdp --tee 3 -m axolotl.cli.train /job/fsdp.yaml")
  local train_rc=$?
  local train_seconds=$(( $(date +%s) - t0 ))
  stop_vram_sampler
  local vram
  vram=$(peak_vram_json)
  echo "$train_log" | tail -30 >&2

  local last_ckpt
  last_ckpt=$(sudo ls -d "$outdir"/run/checkpoint-* 2>/dev/null | sort -V | tail -1)
  local steps=0 ckpt_listing="" ckpt_format="none" ckpt_bytes=0
  if [ -n "$last_ckpt" ]; then
    steps=$(state_steps "$last_ckpt")
    ckpt_listing=$(sudo find "$last_ckpt" -maxdepth 2 -printf '%P\t%s\n' 2>/dev/null | head -40 | tr -s ' ')
    ckpt_bytes=$(sudo du -sb "$last_ckpt" 2>/dev/null | awk '{print $1}')
    if   echo "$ckpt_listing" | grep -q "distcp";        then ckpt_format="torch.distributed.checkpoint (.distcp shards)"
    elif echo "$ckpt_listing" | grep -q "model-00001-of"; then ckpt_format="sharded safetensors index"
    elif echo "$ckpt_listing" | grep -q "safetensors";    then ckpt_format="single safetensors"
    elif echo "$ckpt_listing" | grep -q "\.bin";          then ckpt_format="torch .bin"
    fi
  fi
  log "case $tag: rc=$train_rc steps=$steps ckpt=$ckpt_format"

  # Resume, only if there is something to resume from.
  local resume_rc="null" resume_log="" resume_steps=0 resume_target=0
  if [ -n "$last_ckpt" ]; then
    resume_target=$(( STEPS + 1 ))
    write_config "$model" "/out/run" "$resume_target" "$wrap"
    log "case $tag: resuming from $(basename "$last_ckpt")"
    resume_log=$(run_axolotl "accelerate launch --num_processes $CONTAINER_GPUS --use_fsdp --tee 3 -m axolotl.cli.train /job/fsdp.yaml --resume_from_checkpoint /out/run/$(basename "$last_ckpt")")
    resume_rc=$?
    local resumed_ckpt
    resumed_ckpt=$(sudo ls -d "$outdir"/run/checkpoint-* 2>/dev/null | sort -V | tail -1)
    [ -n "$resumed_ckpt" ] && resume_steps=$(state_steps "$resumed_ckpt")
    echo "$resume_log" | tail -20 >&2
  fi

  rm -f "$WORK/out"; mkdir -p "$saved_work_out"

  CASES_JSON="${CASES_JSON}{
    \"case\": \"$tag\",
    \"model\": \"$model\",
    \"launch_rc\": $train_rc,
    \"seconds\": $train_seconds,
    \"steps_requested\": $STEPS,
    \"steps_completed\": ${steps:-0},
    \"checkpoint_format\": \"$ckpt_format\",
    \"checkpoint_bytes\": ${ckpt_bytes:-0},
    \"checkpoint_listing\": $(echo "$ckpt_listing" | esc),
    \"peak_vram\": $vram,
    \"resume_attempted\": $([ -n "$last_ckpt" ] && echo true || echo false),
    \"resume_rc\": $resume_rc,
    \"resume_target_steps\": $resume_target,
    \"resume_steps_completed\": ${resume_steps:-0},
    \"train_log_tail\": $(echo "$train_log" | tail -c 8000 | esc),
    \"resume_log_tail\": $(echo "$resume_log" | tail -c 4000 | esc)
  },"
}

# Small first. If the mechanism is broken, the large case cannot tell you that,
# and there is no reason to spend ten minutes finding out twice.
run_case "small" "$SMALL_MODEL" "Qwen3DecoderLayer"
run_case "large" "$LARGE_MODEL" "Qwen3DecoderLayer"

VRAM_TOTALS=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | tr '\n' ';')

cat <<JSON
{
  "host_gpu_count": ${HOST_GPUS:-0},
  "host_gpu_list": $(echo "$HOST_GPU_LIST" | esc),
  "image": "$IMAGE",
  "image_pull_ok": $([ "$PULL_RC" -eq 0 ] && echo true || echo false),
  "image_pull_seconds": $PULL_SECONDS,
  "container_torch_report": $(echo "$TORCH_REPORT" | esc),
  "container_torch_stderr": $(tail -c 2000 "$WORK/torch.err" 2>/dev/null | esc),
  "container_gpu_count": ${CONTAINER_GPUS:--1},
  "fsdp_attempted": true,
  "nvidia_smi_total_memory": "$VRAM_TOTALS",
  "cases": [${CASES_JSON%,}]
}
JSON
