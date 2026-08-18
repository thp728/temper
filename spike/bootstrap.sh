#!/bin/bash
# Spike 2 — the SSH-driven bootstrap, run ON the VM.
#
# Piped in over SSH (`ssh ... bash -s`) rather than stored as a JarvisLabs
# startup script, because startup scripts are silently ignored on --vm
# instances (correction C11). This IS the replacement mechanism, so the point
# of this file is to prove the replacement works end to end.
#
# Emits JSON on stdout. Progress goes to stderr so it cannot corrupt the report.
# Never exits non-zero: a failed step is a result.

set -u
say() { echo "[$(date +%H:%M:%S)] $*" >&2; }

IMG="${TRAIN_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
PORT="${TEST_PORT:-8000}"
OUT=/tmp/spike2-artifacts
mkdir -p "$OUT"

jq_escape() { printf '%s' "$1" | tr -d '\r' | sed 's/\\/\\\\/g; s/"/\\"/g' | tr '\n' ' '; }

# --- 1. Pull a realistic training image, timed -------------------------------
# The duration is not incidental: image pull happens on every cold run, so it
# belongs in the ETA the quote shows. Guessing it would make the quote wrong
# from day one.
say "pulling $IMG ..."
t0=$(date +%s)
if sudo docker pull "$IMG" >/dev/null 2>&1; then pull_ok=true; else pull_ok=false; fi
pull_s=$(( $(date +%s) - t0 ))
say "pull done in ${pull_s}s (ok=$pull_ok)"

img_size=$(sudo docker image inspect "$IMG" --format='{{.Size}}' 2>/dev/null || echo 0)

# --- 2. Digest pinning -------------------------------------------------------
# Architecture principle: the image is referenced by digest, not a mutable tag.
# Resolve the digest, then re-pull BY digest to prove the pin is usable.
digest=$(sudo docker image inspect "$IMG" --format='{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' 2>/dev/null || echo "")
if [ -n "$digest" ] && sudo docker pull "$digest" >/dev/null 2>&1; then
  digest_ok=true
else
  digest_ok=false
fi
say "digest pin: $digest_ok"

# --- 3. GPU visible INSIDE the container -------------------------------------
# Host nvidia-smi working proves nothing about the container. Ask torch, not
# the driver -- torch.cuda is what the trainer actually depends on.
gpu_name=$(sudo docker run --rm --gpus all "$IMG" \
  python -c "import torch;print(torch.cuda.get_device_name(0))" 2>/dev/null | tr -d '\r\n' || echo "")
[ -n "$gpu_name" ] && gpu_ok=true || gpu_ok=false
say "torch sees GPU: $gpu_ok ($gpu_name)"

torch_ver=$(sudo docker run --rm "$IMG" python -c "import torch;print(torch.__version__)" 2>/dev/null | tr -d '\r\n' || echo "")
bf16_ok=$(sudo docker run --rm --gpus all "$IMG" \
  python -c "import torch;print(torch.cuda.is_bf16_supported())" 2>/dev/null | tr -d '\r\n' || echo "unknown")

# --- 4. Artifact out of the container ----------------------------------------
# The whole training flow depends on a bind mount surviving container exit.
# Writes an adapter-shaped file (a real tensor via safetensors-free torch.save).
sudo docker run --rm --gpus all -v "$OUT":/out "$IMG" \
  python -c "
import torch, json, os
t = torch.randn(16, 4096)
torch.save({'lora_A': t}, '/out/adapter.pt')
json.dump({'shape': list(t.shape)}, open('/out/meta.json','w'))
" >/dev/null 2>&1
if [ -f "$OUT/adapter.pt" ]; then
  artifact_ok=true
  artifact_bytes=$(stat -c%s "$OUT/adapter.pt")
else
  artifact_ok=false
  artifact_bytes=0
fi
say "artifact out of container: $artifact_ok (${artifact_bytes} bytes)"

# --- 5. Serve a port, for the C7 test ----------------------------------------
# Deliberately does NOT touch UFW. If the port is reachable with the firewall
# untouched, C7 is settled and enabling UFW is a hardening choice rather than a
# prerequisite. Enabling UFW blind over SSH risks locking us out of the box.
ufw_state=$(sudo ufw status 2>/dev/null | head -1 || echo "ufw absent")
nohup sudo docker run --rm -d --name porttest -p ${PORT}:80 \
  ghcr.io/linuxcontainers/alpine:latest \
  sh -c "while true; do printf 'HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\nSPIKE2\n' | nc -l -p 80; done" \
  >/dev/null 2>&1
sleep 3
listening=$(sudo ss -lntp 2>/dev/null | grep -c ":${PORT}" || echo 0)
say "listener on :${PORT} -> $listening (ufw: $ufw_state)"

# --- report ------------------------------------------------------------------
cat <<JSON
{
  "image": "$(jq_escape "$IMG")",
  "pull_ok": $pull_ok,
  "pull_seconds": $pull_s,
  "image_bytes": ${img_size:-0},
  "digest": "$(jq_escape "$digest")",
  "digest_repull_ok": $digest_ok,
  "gpu_in_container_ok": $gpu_ok,
  "gpu_name": "$(jq_escape "$gpu_name")",
  "torch_version": "$(jq_escape "$torch_ver")",
  "bf16_supported": "$(jq_escape "$bf16_ok")",
  "artifact_ok": $artifact_ok,
  "artifact_bytes": $artifact_bytes,
  "test_port": $PORT,
  "port_listening_locally": $listening,
  "ufw_status": "$(jq_escape "$ufw_state")"
}
JSON
