"""Process configuration, read once at import.

The provider SDK resolves its token as: explicit argument, then `JL_API_KEY`,
then `~/.config/jl/config.toml`. There is no config file on this machine and
uvicorn does not inherit a shell that exported anything, so `Client()` inside
the orchestrator raised `AuthError` on the very first real job -- the spikes
worked only because each one called `load_dotenv()` itself.

So the control plane loads the same file the spikes do. Environment wins over
the file, so an explicitly exported key still beats a stale checkout.

**The settings layer.** Every deployment value is read from the environment
through a typed reader with a local default, and the whole set is bundled into
one typed `Settings` object. The module-level names below (`STALL_TIMEOUT_S`,
`DB_PATH`, ...) are the public surface the rest of the codebase reads -- it
never constructs a `Settings` -- so a value two processes or halves share has
one definition (ADR-0010). A new deployment setting is a new field here, and
nowhere else.

Where the boundary sits is the subject of
[ADR-0062](../docs/adr/0062-the-configuration-boundary-sits-at-deployment-settings.md):
a value is a *deployment* setting when the right number depends on the
deployment's hardware, catalog or tolerance; a value derived against a
measured report is a *domain* constant and stays in the module that owns the
derivation, whatever it happens to read.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from temper_core import divergence, hyperparams


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


# The .env files are read into os.environ here, once, before any setting is
# parsed: the provider SDK resolves JL_API_KEY from os.environ (not from a
# settings object), so the file's contents must land where the SDK looks. The
# typed readers below then read the resulting environment, with an explicitly
# exported variable beating a stale checkout.
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
#
# Each documented default is defined once, here, and read by the typed reader
# that parses the environment *and* by the typed Settings view below -- a value
# two components must agree on is defined once and read, never retyped.

# 15 minutes. The longest legitimately quiet stretch measured on a real run
# (2026-08-19) was a 183-second image build, so this is an order of magnitude
# above the worst observed silence -- and far below an unattended overnight,
# which is the failure it exists to prevent.
DEFAULT_STALL_TIMEOUT_S = 15 * 60

# 24 hours. **Adopted convention, not a measured or cited figure** -- it is the
# ceiling commonly used for managed training jobs, taken as a starting point
# because a backstop that never fires on a legitimate run is doing its job
# either way. Nothing on the current 4B/8B catalog comes close: the measured
# training phase on 2026-08-19 was 161 seconds. The number to revisit when the
# catalog grows is this one, and it is configuration for that reason.
DEFAULT_MAX_JOB_DURATION_S = 24 * 60 * 60

# How long a quote's prices and availability are honoured for. A quote is an
# estimate against hardware that changes; "a price I was shown yesterday is not
# silently honoured against hardware that has changed" (spec 005) is the reason
# it expires at all, and 24h is the boundary the sentence implies. Configurable
# because the right value depends on how often this provider's availability and
# pricing move, which is not something this repo has measured.
DEFAULT_QUOTE_TTL_S = 24 * 60 * 60

STALL_TIMEOUT_S = _seconds("TEMPER_STALL_TIMEOUT_S", DEFAULT_STALL_TIMEOUT_S)
MAX_JOB_DURATION_S = _seconds(
    "TEMPER_MAX_JOB_DURATION_S", DEFAULT_MAX_JOB_DURATION_S
)
QUOTE_TTL_S = _seconds("TEMPER_QUOTE_TTL_S", DEFAULT_QUOTE_TTL_S)


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
            f"that means 'no limit': an unbounded upload ties the control "
            f"plane up in validation for longer than any user should wait, "
            f"and a limit nobody can adjust gets treated as arbitrary."
        )
    return value


# --- dataset size limit -----------------------------------------------------
# **1.3 GB, derived from the measured streaming throughput, not from memory.**
# Validation now streams one row at a time
# ([ADR-0036](../docs/adr/0036-the-dataset-size-limit-is-derived-from-measured-throughput.md)),
# so the memory multiplier that produced ADR-0005's 1 GB figure is gone --
# spike 9 measured the streaming path flat from 1 GB to 20 GB (peak RSS +4 MB
# regardless of size). What the limit still protects is that validation
# finishes within a tolerable synchronous wait: the mean streaming rate
# measured on a real 1 GB file was **21.6 MB/s** (a floor -- the machine was
# contended), and 60 seconds of waiting is the point at which a synchronous
# upload stops reading as a wait and starts reading as a hang. That is
# 21.6 MB/s x 60 s = 1296 MB, rounded up to a clean **1.3 GiB (1331.2 MB)** --
# a rounding of under 3%, smaller than the uncertainty in a measured rate
# that is itself a floor. The rate is **measured**; the 60-second wait is
# **a judgment**, and both are recorded in the ADR so the number can be
# revisited against a less contended measurement or a different tolerance.
#
# One gigabyte is approximately 577,000 conversational rows; 1.3 GB is
# approximately 750,000.
#
# Deliberately below the 25 GB named baseline, which is a property of a
# multi-node fleet; on a single-GPU job with a 24-hour ceiling a dataset that
# size cannot finish anyway. Advertising a limit the system cannot honour is
# worse than being visibly below it.
#
# Configurable because the right number depends on the deployment's tolerance
# for a synchronous wait, not on this code. TEMPER_MAX_DATASET_MB, in
# megabytes. 1.3 GiB is 1331.2 MB.
DEFAULT_MAX_DATASET_MB = 1331.2

# Parsed once and read twice: the byte ceiling the limit enforces, and the
# typed view's own figure -- a value two components must agree on is defined
# once and read, never retyped.
_max_dataset_mb = _megabytes("TEMPER_MAX_DATASET_MB", DEFAULT_MAX_DATASET_MB)
MAX_DATASET_BYTES = int(_max_dataset_mb * 1024 * 1024)


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
# Whether real compute is used is this one setting (ADR-0024, ADR-0062).
DEFAULT_FAKE_PROVIDER = False
DEFAULT_FAKE_LINE_DELAY_S = 0.0

FAKE_PROVIDER = bool(
    os.environ.get("TEMPER_FAKE_PROVIDER", DEFAULT_FAKE_PROVIDER)
)

# How long the simulated machine waits between output lines. Zero (default)
# completes a canned run in milliseconds, which is what the suite wants except
# when a journey is *watching*: the live-job journeys (issue #39) need a run
# that lasts long enough to see output arrive without a refresh and to cancel
# mid-run, so the journeys' own control plane is booted with a positive value.
# Fake-only -- the real provider's cadence is the trainer's, not this knob's.
FAKE_LINE_DELAY_S = float(
    os.environ.get("TEMPER_FAKE_LINE_DELAY_S") or DEFAULT_FAKE_LINE_DELAY_S
)


# --- the fault surface (issue #24) -------------------------------------------
# **Off by default, and that is a safety property, not a convenience.** A job
# whose hyperparameters carry a fault spec is refused at creation unless the
# deployment has explicitly turned the surface on -- `TEMPER_FAKE_PROVIDER`
# (the zero-cost tier: the fake provider honours the faults) or
# `TEMPER_FAULT_SURFACE` (the deliberate operator tier: lets a fault spec
# reach a real machine so the trainer-side faults genuinely fire). Without
# one of the two, no fault spec can even be created, so a fault surface that
# can be switched on by accident in front of a user does not exist.
# **A default-configured start lands in the safe mode: real compute, faults
# refused** (ADR-0062).
DEFAULT_FAULT_SURFACE = False

FAULT_SURFACE = bool(
    os.environ.get("TEMPER_FAULT_SURFACE", DEFAULT_FAULT_SURFACE)
)


def fault_surface_refusal(name: str) -> dict | None:
    """The coded refusal for launching a job carrying fault `name` under this
    process's switches, or None when the fault may be caused.

    The surface-on policy, defined once and read by the create path and the
    orchestrator alike, so the two cannot drift. The fake provider (the
    zero-cost tier) can cause every fault; the deliberate real tier can cause
    only trainer-side faults, because provider-side faults are behaviour of
    the fake provider seam and no real provider honours them. A provider-side
    fault on the real tier is refused rather than launched under a history
    that would claim a deliberate break no machine will make.
    """
    if FAKE_PROVIDER:
        return None
    if not FAULT_SURFACE:
        return {
            "code": "fault_surface_refused",
            "message": (
                "This job carries a fault spec, but the fault surface is off "
                "by default. It can only be switched on deliberately: set "
                "TEMPER_FAKE_PROVIDER for the zero-cost tier, or "
                "TEMPER_FAULT_SURFACE for the deliberate real-hardware tier. "
                "Nothing was launched."
            ),
        }
    from temper_core import faults

    if faults.side_of(name) == "provider":
        return {
            "code": "fault_not_causable",
            "message": (
                f"'{name}' is a provider-side fault: it is behaviour of the "
                "provider seam, so it can only be caused on the zero-cost "
                "tier (TEMPER_FAKE_PROVIDER). The deliberate real-hardware "
                "tier has no provider that honours it, so it is refused "
                "rather than launched under a history that claims a "
                "deliberate break no machine will make. Nothing was "
                "provisioned."
            ),
        }
    return None


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
DEFAULT_STORAGE_BACKEND = "filesystem"

STORAGE_BACKEND = _text("TEMPER_STORAGE_BACKEND") or DEFAULT_STORAGE_BACKEND

# Where the filesystem backend roots its keys. Under `/data/` with everything
# else runtime-written -- test_storage_paths pins that boundary.
DEFAULT_STORAGE_ROOT = REPO_ROOT / "data" / "objects"

STORAGE_ROOT = Path(_text("TEMPER_STORAGE_ROOT") or DEFAULT_STORAGE_ROOT)

# Where the SQLite database lives, under `/data/` with everything else
# runtime-written (test_storage_paths pins that boundary). Overridable so the
# e2e journeys can run their control plane against a database of their own
# rather than the developer's -- the same ownership rule as the journeys'
# ports: a journey must not inherit another surface's orphans, and an
# interrupted journey run must not be able to poison the database the next
# gate run boots against.
DEFAULT_DB_PATH = REPO_ROOT / "data" / "temper.db"

DB_PATH = Path(_text("TEMPER_DB_PATH") or DEFAULT_DB_PATH)

# When set, `db.init()` recreates the database at startup rather than reusing
# it. The e2e journeys set it so their control plane boots against a clean
# database on every run -- a database is not a thing a journey should inherit,
# and a run that was interrupted mid-job must not be able to poison the next
# run's startup. The developer's own database never sets this.
DEFAULT_DB_RESET = False

DB_RESET = bool(os.environ.get("TEMPER_DB_RESET", DEFAULT_DB_RESET))

S3_BUCKET = _text("TEMPER_S3_BUCKET")
S3_ENDPOINT_URL = _text("TEMPER_S3_ENDPOINT_URL")
S3_REGION = _text("TEMPER_S3_REGION")

# Signing secret for filesystem write grants. Unset means a per-process random:
# correct for local runs, whose grants live exactly one job inside one process,
# and useless across restarts by construction rather than by hope.
STORAGE_SECRET = _text("TEMPER_STORAGE_SECRET")


def _count(name: str, default: int) -> int:
    """A non-negative integer from the environment, or its default.

    Same contract as `_seconds` and `_megabytes`: read once at import, and a
    value that cannot be honoured stops the process rather than falling back
    silently.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} is not a whole number. It configures a retention "
            f"bound, so it is refused rather than ignored."
        ) from None
    if value < 0:
        raise ValueError(
            f"{name}={raw!r} must be zero or greater. A negative retention "
            f"bound would delete nothing and keep nothing."
        )
    return value


# --- divergence detection (issue #36) -----------------------------------------
# The numbers live in `temper_core.divergence` with the report-b derivation
# beside them (the way ADR-0036 records 21.6 MB/s times 60 s). This module
# reads them as defaults and keeps the TEMPER_DIVERGENCE_* overrides.
# A value two components must agree on is defined once and read, never
# retyped (AGENTS.md line 47, ADR-0010).
def _positive_float(name: str, default: float) -> float:
    """A positive float from the environment, or its default.

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
            f"{name}={raw!r} is not a number. It configures a divergence "
            f"threshold, so it is refused rather than ignored."
        ) from None
    if value <= 0:
        raise ValueError(
            f"{name}={raw!r} must be greater than zero. A non-positive "
            f"divergence threshold would never fire or would fire always."
        )
    return value


def _positive_int(name: str, default: int) -> int:
    """A positive integer from the environment, or its default."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} is not a whole number. It configures a divergence "
            f"threshold, so it is refused rather than ignored."
        ) from None
    if value < 1:
        raise ValueError(
            f"{name}={raw!r} must be at least 1. A divergence window or count "
            f"below one is not a window."
        )
    return value


DIVERGENCE_MULTIPLIER = _positive_float(
    "TEMPER_DIVERGENCE_MULTIPLIER", divergence.DIVERGENCE_MULTIPLIER
)
DIVERGENCE_WINDOW = _positive_int(
    "TEMPER_DIVERGENCE_WINDOW", divergence.DIVERGENCE_WINDOW
)
DIVERGENCE_CONSECUTIVE = _positive_int(
    "TEMPER_DIVERGENCE_CONSECUTIVE", divergence.DIVERGENCE_CONSECUTIVE
)
WARNING_CONSECUTIVE = _positive_int(
    "TEMPER_WARNING_CONSECUTIVE", divergence.WARNING_CONSECUTIVE
)

# --- checkpoint retention ----------------------------------------------------
# How many checkpoints one job may keep in object storage at once, and hence
# how many scoped write grants the control plane mints for a job. Bounded by
# construction rather than by deletion: the machine overwrites the oldest slot
# with each new checkpoint, so storage never holds more than this many objects
# per job (issue #37).
#
# The default is read from the resolver's own table rather than retyped: it
# matches the `save_total_limit` the trainer keeps on the machine's disk, and
# "a value two components must agree on is defined once and read, never
# retyped" applies to the default as much as to the value. Set
# TEMPER_CHECKPOINT_RETENTION explicitly when a deployment wants storage to
# diverge from disk.
DEFAULT_CHECKPOINT_RETENTION = hyperparams.DEFAULTS["save_total_limit"]

CHECKPOINT_RETENTION = _count(
    "TEMPER_CHECKPOINT_RETENTION", DEFAULT_CHECKPOINT_RETENTION
)


class Settings(BaseModel):
    """The typed view of process configuration, read once at import.

    One object bundling every deployment value, each read from its `TEMPER_*`
    environment variable by the typed readers above. Nothing is required:
    every value has a local default, so the process starts with no
    configuration set, because a reviewer's first run should not require a
    decision. A value that cannot be honoured is refused by the reader that
    parses it -- at import, where it is one legible line at boot rather than
    a control that quietly is not the one anybody configured.

    Call sites read the module-level names (`config.STALL_TIMEOUT_S`, ...),
    not this object; it exists so the whole configuration set is one typed,
    documented thing a reviewer can read end to end, and so a value shared by
    two processes or halves has exactly one definition.

    A bare `Settings()` is exactly what a start with nothing configured
    resolves to: every field is its documented local default, defined once in
    the `DEFAULT_*` constants above and read here and by the environment
    readers alike.
    """

    stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S
    max_job_duration_s: float = DEFAULT_MAX_JOB_DURATION_S
    quote_ttl_s: float = DEFAULT_QUOTE_TTL_S
    max_dataset_mb: float = DEFAULT_MAX_DATASET_MB
    fake_provider: bool = DEFAULT_FAKE_PROVIDER
    fake_line_delay_s: float = DEFAULT_FAKE_LINE_DELAY_S
    fault_surface: bool = DEFAULT_FAULT_SURFACE
    storage_backend: str = DEFAULT_STORAGE_BACKEND
    storage_root: Path = DEFAULT_STORAGE_ROOT
    db_path: Path = DEFAULT_DB_PATH
    db_reset: bool = DEFAULT_DB_RESET
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    storage_secret: str | None = None
    checkpoint_retention: int = DEFAULT_CHECKPOINT_RETENTION


# Read once at import, like every other piece of process configuration. The
# typed bundle every value above is drawn from; the module-level names are
# what the rest of the codebase reads.
settings = Settings(
    stall_timeout_s=STALL_TIMEOUT_S,
    max_job_duration_s=MAX_JOB_DURATION_S,
    quote_ttl_s=QUOTE_TTL_S,
    max_dataset_mb=_max_dataset_mb,
    fake_provider=FAKE_PROVIDER,
    fake_line_delay_s=FAKE_LINE_DELAY_S,
    fault_surface=FAULT_SURFACE,
    storage_backend=STORAGE_BACKEND,
    storage_root=STORAGE_ROOT,
    db_path=DB_PATH,
    db_reset=DB_RESET,
    s3_bucket=S3_BUCKET,
    s3_endpoint_url=S3_ENDPOINT_URL,
    s3_region=S3_REGION,
    storage_secret=STORAGE_SECRET,
    checkpoint_retention=CHECKPOINT_RETENTION,
)
