#!/bin/bash
# Phase 0 vertical spike — on-VM probe.
#
# Uploaded to JarvisLabs as a startup script and attached at instance creation.
# Runs automatically on launch. Answers, empirically, the questions the
# reference architecture assumes the answers to.
#
# Writes /tmp/probe-report.json and mirrors everything to /tmp/probe.log.
# Never exits non-zero — a failed check is a RESULT, not an error. The whole
# point is to bring back a complete report rather than die on the first
# surprise.

set -u
exec > >(tee -a /tmp/probe.log) 2>&1

# /tmp, NOT /root. Startup scripts run as root, but JarvisLabs VMs log in as
# `ubuntu`, and /root is mode 700 -- so a report written there is unreadable by
# the very user who has to fetch it. Learned by doing exactly that:
#   cat: /root/probe-report.json: Permission denied  (x12 retries)
# /tmp is 1777 and the default umask leaves the file world-readable.
REPORT=/tmp/probe-report.json
GHCR_TEST_IMAGE="ghcr.io/linuxcontainers/alpine:latest"

echo "=== JarvisLabs Phase 0 probe — $(date -Is) ==="

# --- helpers ---------------------------------------------------------------
# Emit a JSON key/value. Values are JSON literals, so strings need quoting by
# the caller.
kv() { printf '  "%s": %s,\n' "$1" "$2" >> "$REPORT"; }
jstr() { printf '"%s"' "$(printf '%s' "$1" | tr -d '\r' | sed 's/\\/\\\\/g; s/"/\\"/g' | tr '\n' ' ')"; }
have() { command -v "$1" >/dev/null 2>&1 && echo true || echo false; }

echo "{" > "$REPORT"
kv "probe_version" '"1"'
kv "started_at" "$(jstr "$(date -Is)")"

# --- C1: the question this whole spike exists to answer ---------------------
# reference-technical-architecture.md §14 step 1 assumes Docker is present.
# The SDK cannot supply a custom image, so if Docker is not here, the whole
# immutable-image approach needs replacing.
DOCKER_PRESENT=$(have docker)
kv "docker_present" "$DOCKER_PRESENT"
echo "--- docker present: $DOCKER_PRESENT"

if [ "$DOCKER_PRESENT" = "true" ]; then
  kv "docker_version" "$(jstr "$(docker --version 2>&1)")"

  # Present on PATH is not the same as usable. The daemon has to be running
  # and reachable, which in a fresh VM it may well not be.
  if docker info >/dev/null 2>&1; then
    kv "docker_daemon_running" "true"
    echo "--- docker daemon: running"
  else
    echo "--- docker daemon: not running, attempting start"
    systemctl start docker >/dev/null 2>&1 || service docker start >/dev/null 2>&1 || true
    sleep 5
    if docker info >/dev/null 2>&1; then
      kv "docker_daemon_running" "true"
      kv "docker_daemon_needed_start" "true"
      echo "--- docker daemon: started manually"
    else
      kv "docker_daemon_running" "false"
      echo "--- docker daemon: FAILED to start"
    fi
  fi

  # Can it pull from a public registry at all? Separates "no Docker" from
  # "Docker but no egress", which are very different problems.
  if docker pull "$GHCR_TEST_IMAGE" >/dev/null 2>&1; then
    kv "ghcr_pull_ok" "true"
    echo "--- ghcr pull: ok"
  else
    kv "ghcr_pull_ok" "false"
    echo "--- ghcr pull: FAILED"
  fi

  # Does the container actually see the GPU? Host nvidia-smi working proves
  # nothing about whether --gpus all does.
  if docker run --rm --gpus all "$GHCR_TEST_IMAGE" true >/dev/null 2>&1; then
    kv "docker_gpu_passthrough_ok" "true"
    echo "--- docker --gpus all: ok"
  else
    kv "docker_gpu_passthrough_ok" "false"
    echo "--- docker --gpus all: FAILED (nvidia-container-toolkit missing?)"
  fi
else
  kv "docker_daemon_running" "false"
  kv "ghcr_pull_ok" "false"
  kv "docker_gpu_passthrough_ok" "false"
  echo "--- docker absent: the fallback is a uv-provisioned env; log the decision"
fi

# --- what else is on a bare VM ---------------------------------------------
# Determines how much the startup script has to build from scratch, which is
# startup latency on every single run.
kv "nvidia_smi_present" "$(have nvidia-smi)"
if command -v nvidia-smi >/dev/null 2>&1; then
  kv "gpu_name" "$(jstr "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)")"
  kv "driver_version" "$(jstr "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)")"
  kv "vram_total_mib" "$(jstr "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>&1 | head -1)")"
fi
kv "python3_present" "$(have python3)"
kv "python3_version" "$(jstr "$(python3 --version 2>&1)")"
kv "uv_present" "$(have uv)"
kv "git_present" "$(have git)"
kv "curl_present" "$(have curl)"
kv "os_release" "$(jstr "$(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d'"' -f2)")"
kv "disk_avail" "$(jstr "$(df -h / | awk 'NR==2 {print $4}')")"
kv "cpu_cores" "$(nproc 2>/dev/null || echo 0)"
kv "ram_gb" "$(free -g 2>/dev/null | awk 'NR==2 {print $2}' || echo 0)"

# --- outbound reachability --------------------------------------------------
# §14 says training VMs expose no public ports and talk outbound over TLS.
# If outbound HTTPS is blocked or proxied, the callback protocol in §15 and
# every R2 signed URL fail — and they fail late, mid-run.
for host in https://ghcr.io https://huggingface.co https://pypi.org; do
  key="egress_$(echo "$host" | sed 's#https://##; s#\.#_#g')"
  if curl -fsS -m 15 -o /dev/null "$host" 2>/dev/null; then
    kv "$key" "true"
  else
    kv "$key" "false"
    echo "--- egress FAILED: $host"
  fi
done

# --- script arguments -------------------------------------------------------
# §14 passes a one-use bootstrap token as a short-lived script argument.
# Confirm arguments actually arrive, WITHOUT echoing the value.
kv "script_args_count" "$#"
kv "script_args_received" "$([ $# -gt 0 ] && echo true || echo false)"

kv "finished_at" "$(jstr "$(date -Is)")"
printf '  "probe_complete": true\n}\n' >> "$REPORT"

echo "=== probe complete — report at $REPORT ==="
cat "$REPORT"
exit 0
