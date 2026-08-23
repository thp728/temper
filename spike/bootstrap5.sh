#!/usr/bin/env bash
# Runs ON the VM for spike 5. Piped in over SSH (startup scripts are accepted
# and silently ignored on VMs -- correction C11), so it must be self-contained
# and it must print JSON on stdout and nothing else.
#
# Two questions, in this order:
#   1. Is the disk the API said it created actually there, and writable?
#   2. How fast do model weights arrive onto it?
#
# Diagnostics go to stderr so stdout stays parseable.

set -uo pipefail

REPO="${1:-Qwen/Qwen3-8B}"
FILL_GB="${2:-0}"          # 0 = skip the fill test
SAMPLE_INTERVAL=5

log() { echo "[bootstrap5] $*" >&2; }

json_escape() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }

# --- 1. the disk ------------------------------------------------------------
log "inspecting disk"
ROOT_DEV=$(findmnt -no SOURCE / 2>/dev/null || echo unknown)
DF_TOTAL_KB=$(df -Pk / | awk 'NR==2 {print $2}')
DF_AVAIL_KB=$(df -Pk / | awk 'NR==2 {print $4}')
DF_HUMAN=$(df -h / | tail -1 | tr -s ' ')
LSBLK=$(lsblk -b -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null | tr -s ' ')

# Writable is a separate claim from present. A read-only or thin-provisioned
# volume reports its nominal size in df and fails on write, and "the API
# accepted 400" is not evidence of either.
log "write test"
WRITE_OK=false
WRITE_MBPS=0
WRITE_ERR=""
TESTFILE=/tmp/spike5-write-test.bin
if WRITE_OUT=$(dd if=/dev/zero of="$TESTFILE" bs=1M count=2048 oflag=direct 2>&1); then
  WRITE_OK=true
  WRITE_MBPS=$(echo "$WRITE_OUT" | tail -1 | grep -oE '[0-9.]+ [MG]B/s' | head -1 || echo "")
else
  WRITE_ERR="$WRITE_OUT"
fi
rm -f "$TESTFILE"

# --- 1b. optional fill test -------------------------------------------------
# The strongest evidence that a 400 GB disk is 400 GB is writing most of it.
# fallocate is instant on filesystems that support it, which proves the
# allocation but not the media; dd proves the media and takes an hour. This
# uses fallocate and says which it did.
FILL_OK=null
FILL_DETAIL=""
if [ "$FILL_GB" != "0" ]; then
  log "fill test: fallocate ${FILL_GB}G"
  if fallocate -l "${FILL_GB}G" /tmp/spike5-fill.bin 2>/dev/null; then
    FILL_OK=true
    FILL_DETAIL="fallocate ${FILL_GB}G succeeded (allocation proven, media not written)"
  else
    FILL_OK=false
    FILL_DETAIL="fallocate ${FILL_GB}G failed"
  fi
  rm -f /tmp/spike5-fill.bin
fi

# --- 2. the download --------------------------------------------------------
# A --vm instance has python3 but NOT pip3 (measured 2026-08-23 -- spike 1
# recorded "Python 3.10.12 and git present, no uv" and nobody checked for pip).
# Resolve it in three steps rather than assuming: the binary, the module, then
# apt. Assuming pip3 cost this spike one run.
log "resolving pip"
PIP=""
if command -v pip3 >/dev/null 2>&1; then
  PIP="pip3"
elif python3 -m pip --version >/dev/null 2>&1; then
  PIP="python3 -m pip"
else
  log "no pip -- installing python3-pip via apt"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip >/dev/null 2>&1
  if python3 -m pip --version >/dev/null 2>&1; then
    PIP="python3 -m pip"
  fi
fi
log "pip is: ${PIP:-NONE}"

PIP_OUT=""
PIP_RC=1
if [ -n "$PIP" ]; then
  log "installing huggingface_hub"
  PIP_OUT=$($PIP install --quiet --disable-pip-version-check 'huggingface_hub[cli]' 2>&1)
  PIP_RC=$?
fi
# The console script lands in ~/.local/bin, which is not on a non-login PATH.
export PATH="$HOME/.local/bin:$PATH"
HF_BIN=$(command -v hf || command -v huggingface-cli || echo "")

DL_OK=false
DL_BYTES=0
DL_SECONDS=0
SAMPLES="[]"
DL_ERR=""

if [ -z "$HF_BIN" ]; then
  DL_ERR="huggingface CLI not on PATH after pip install (pip=${PIP:-NONE}, rc=$PIP_RC): $PIP_OUT"
else
  # HF_HUB_ENABLE_HF_TRANSFER is deliberately NOT set. The measured rate is
  # therefore the rate of the plain python client, which is what the trainer
  # image uses today. Turning it on would measure a path this product does not
  # take, and would report a number nothing else could reproduce.
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  DEST=/tmp/spike5-model
  mkdir -p "$DEST"
  log "downloading $REPO"

  "$HF_BIN" download "$REPO" --local-dir "$DEST" >/dev/null 2>/tmp/spike5-dl.err &
  DL_PID=$!

  # Sample the growing directory rather than parsing progress output: the size
  # on disk is the ground truth, and it is what an ETA would be computed from.
  # Bursty versus steady is the point -- a live ETA from a 30-second window
  # oscillates badly if the rate is bursty, and that is a UI decision.
  START=$(date +%s.%N)
  SAMPLE_LIST=""
  while kill -0 $DL_PID 2>/dev/null; do
    sleep $SAMPLE_INTERVAL
    NOW=$(date +%s.%N)
    SZ=$(du -sb "$DEST" 2>/dev/null | awk '{print $1}')
    SZ=${SZ:-0}
    T=$(echo "$NOW - $START" | bc)
    SAMPLE_LIST="${SAMPLE_LIST}{\"t\":$T,\"bytes\":$SZ},"
  done
  wait $DL_PID
  DL_RC=$?
  END=$(date +%s.%N)

  DL_SECONDS=$(echo "$END - $START" | bc)
  DL_BYTES=$(du -sb "$DEST" 2>/dev/null | awk '{print $1}')
  DL_BYTES=${DL_BYTES:-0}
  SAMPLES="[${SAMPLE_LIST%,}]"
  if [ "$DL_RC" -eq 0 ]; then
    DL_OK=true
  else
    DL_ERR=$(tail -c 2000 /tmp/spike5-dl.err 2>/dev/null || echo "rc=$DL_RC")
  fi

  # Disk after the download, so the findings can say what a real model costs
  # in disk terms rather than what the repo card claims.
  DF_AFTER_KB=$(df -Pk / | awk 'NR==2 {print $4}')
  rm -rf "$DEST"
fi
DF_AFTER_KB=${DF_AFTER_KB:-$DF_AVAIL_KB}

# --- report -----------------------------------------------------------------
cat <<JSON
{
  "root_device": "$ROOT_DEV",
  "df_total_kb": ${DF_TOTAL_KB:-0},
  "df_available_kb": ${DF_AVAIL_KB:-0},
  "df_available_after_download_kb": ${DF_AFTER_KB:-0},
  "df_human": "$DF_HUMAN",
  "lsblk": $(echo "$LSBLK" | json_escape),
  "write_test_ok": $WRITE_OK,
  "write_test_rate": "$WRITE_MBPS",
  "write_test_error": $(echo "$WRITE_ERR" | json_escape),
  "fill_test_ok": $FILL_OK,
  "fill_test_detail": "$FILL_DETAIL",
  "download_repo": "$REPO",
  "download_ok": $DL_OK,
  "download_bytes": $DL_BYTES,
  "download_seconds": ${DL_SECONDS:-0},
  "download_error": $(echo "$DL_ERR" | json_escape),
  "download_samples": $SAMPLES,
  "hf_transfer_enabled": false
}
JSON
