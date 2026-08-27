"""Process configuration, read once at import.

The provider SDK resolves its token as: explicit argument, then `JL_API_KEY`,
then `~/.config/jl/config.toml`. There is no config file on this machine and
uvicorn does not inherit a shell that exported anything, so `Client()` inside
the orchestrator raised `AuthError` on the very first real job -- the spikes
worked only because each one called `load_dotenv()` itself.

So the control plane loads the same file the spikes do. Environment wins over
the file, so an explicitly exported key still beats a stale checkout.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    """Walk up to the workspace root rather than counting directories.

    A depth-coded `parents[4]` breaks silently the next time this file moves,
    which is exactly what the Phase B reorg did to the version that lived here
    before. The marker is the workspace root's own pyproject beside `packages/`.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (
            parent / "packages"
        ).is_dir():
            return parent
    raise RuntimeError("workspace root not found above " + __file__)


REPO_ROOT = _repo_root()

# spike/.env is where the key already lives and is already git-ignored at two
# levels. A repo-root .env is honoured too, for a deployment that mounts one.
ENV_FILES = (REPO_ROOT / ".env", REPO_ROOT / "spike" / ".env")


def load_env() -> list[str]:
    """Populate os.environ from the .env files. Returns the names it set."""
    loaded: list[str] = []
    for path in ENV_FILES:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and value and not os.environ.get(key):
                os.environ[key] = value
                loaded.append(key)
    return loaded


def provider_credentials_present() -> bool:
    """True if the SDK will find a token. Checked at startup, not mid-run.

    Discovering a missing credential four seconds into a job is survivable;
    discovering it is still better done at boot, where it is one line in the
    log rather than a failed run the user has to interpret.
    """
    if os.environ.get("JL_API_KEY"):
        return True
    try:
        from jarvislabs.config import config_path

        return config_path().exists()
    except Exception:
        return False


load_env()


def _seconds(name: str, default: float) -> float:
    """A duration from the environment, or its default.

    Read at import and exposed as a module attribute rather than looked up per
    use: the value is part of the process's configuration, and a limit that can
    change halfway through a run is a limit nobody can explain afterwards.

    **A value that cannot be honoured stops the process.** Falling back to the
    default would be the failure this whole module exists to remove: an
    operator who set `TEMPER_STALL_TIMEOUT_S=15m` believes a limit is in force
    that is not, which is the same shape as the unread constant, and this time
    it would be invisible in the source as well. Refusing at import means one
    legible line at boot rather than a control that quietly is not the one
    anybody configured.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} is not a number of seconds. It configures a "
            f"safety limit, so it is refused rather than ignored."
        ) from None
    if value <= 0:
        raise ValueError(
            f"{name}={raw!r} must be greater than zero. To disable the limit, "
            f"set it to a value large enough to be unreachable — there is no "
            f"value that means 'no limit', because a limit that can be "
            f"switched off by a typo is not a limit."
        )
    return value


# --- runtime limits --------------------------------------------------------
# Both are circuit breakers against a wedged job, not a cap on what a user may
# legitimately train. Both are configuration, because the right number depends
# on the catalog and the catalog will grow.

# 15 minutes. The longest legitimately quiet stretch measured on a real run
# (2026-08-19) was a 183-second image build, so this is an order of magnitude
# above the worst observed silence -- and far below an unattended overnight,
# which is the failure it exists to prevent.
STALL_TIMEOUT_S = _seconds("TEMPER_STALL_TIMEOUT_S", 15 * 60)

# 24 hours. **Adopted convention, not a measured or cited figure** -- it is the
# ceiling commonly used for managed training jobs, taken as a starting point
# because a backstop that never fires on a legitimate run is doing its job
# either way. Nothing on the current 4B/8B catalog comes close: the measured
# training phase on 2026-08-19 was 161 seconds. The number to revisit when the
# catalog grows is this one, and it is configuration for that reason.
MAX_JOB_DURATION_S = _seconds("TEMPER_MAX_JOB_DURATION_S", 24 * 60 * 60)


def _megabytes(name: str, default: float) -> float:
    """A size in megabytes from the environment, or its default.

    Same contract as `_seconds`: read once at import, and a value that cannot
    be honoured stops the process rather than falling back silently.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} is not a number of megabytes. It configures a "
            f"safety limit, so it is refused rather than ignored."
        ) from None
    if value <= 0:
        raise ValueError(
            f"{name}={raw!r} must be greater than zero. There is no value "
            f"that means 'no limit': an unbounded upload fails as an "
            f"out-of-memory crash instead of a typed error."
        )
    return value


# --- dataset size limit -----------------------------------------------------
# **1 GB, derived rather than chosen.** Validation holds the whole dataset in
# memory, and its peak resident memory was measured across five dataset sizes,
# one fresh process each (method and table in docs/adr/0004, "The measured
# memory multiplier"): it converges to **4.8x the file size** -- 10 MB -> 67.6 MB
# up to 200 MB -> 959.4 MB, the higher ratios at small sizes being fixed
# interpreter overhead. At 4.8x, a 1 GB dataset peaks around 4.8 GB: roughly
# 15% of the development machine's memory, with headroom for concurrent work.
# One gigabyte is approximately 577,000 conversational rows.
#
# This is deliberately below the 25 GB named baseline, which is a property of a
# multi-node fleet; on a single-GPU job with a 24-hour ceiling a dataset that
# size cannot finish anyway. Advertising a limit the system cannot honour is
# worse than being visibly below it.
#
# It is a limit of the current *in-memory* validation path, not a product rule:
# streaming validation (Phase B, with the storage work) removes it.
#
# Configurable because the right number depends on the machine's memory, not on
# this code. TEMPER_MAX_DATASET_MB, in megabytes.
MAX_DATASET_BYTES = int(
    _megabytes("TEMPER_MAX_DATASET_MB", 1024) * 1024 * 1024
)


# --- journey provider -------------------------------------------------------
# **Off by default.** TEMPER_FAKE_PROVIDER swaps the in-package FakeProvider
# in for launched jobs, so the browser journeys (apps/web/e2e) can drive a
# launch to completion with no hardware, no credentials and no way to reach
# the billing account -- which is what makes them runnable on every push.
# `models.new_models()` also reads this flag: the model-choice screen calls
# `/v1/models` on every load, and a journey that resolves real facts from
# Hugging Face on every run trades that same "runs everywhere" guarantee for
# a network dependency that costs nothing to remove. Everything else stays
# real: the same API, the same database, the same orchestrator transitions.
FAKE_PROVIDER = bool(os.environ.get("TEMPER_FAKE_PROVIDER"))


# --- stored objects ----------------------------------------------------------
# Spec 006 / issue #22: every stored object sits behind one storage seam, and
# which implementation answers is configuration. The values here are parsed
# only; refusing an unusable combination is `storage.from_config`'s job, because
# what counts as unusable is knowledge about backends and belongs behind the
# seam with them.


def _text(name: str) -> str | None:
    """An optional string from the environment, or None when unset or empty."""
    raw = os.environ.get(name)
    return raw if raw else None


# "filesystem" (local runs and tests) or "s3" (any S3-compatible store,
# including MinIO). Nothing set means filesystem: a fresh clone must run with
# no configuration at all.
STORAGE_BACKEND = _text("TEMPER_STORAGE_BACKEND") or "filesystem"

# Where the filesystem backend roots its keys. Under `/data/` with everything
# else runtime-written -- test_storage_paths pins that boundary.
STORAGE_ROOT = Path(
    _text("TEMPER_STORAGE_ROOT") or REPO_ROOT / "data" / "objects"
)

S3_BUCKET = _text("TEMPER_S3_BUCKET")
S3_ENDPOINT_URL = _text("TEMPER_S3_ENDPOINT_URL")
S3_REGION = _text("TEMPER_S3_REGION")

# Signing secret for filesystem write grants. Unset means a per-process random:
# correct for local runs, whose grants live exactly one job inside one process,
# and useless across restarts by construction rather than by hope.
STORAGE_SECRET = _text("TEMPER_STORAGE_SECRET")
