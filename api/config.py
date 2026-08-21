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

REPO_ROOT = Path(__file__).parent.parent

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
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# --- runtime limits --------------------------------------------------------
# Both are circuit breakers against a wedged job, not a cap on what a user may
# legitimately train. Both are configuration, because the right number depends
# on the catalog and the catalog will grow.

# 15 minutes. The longest legitimately quiet stretch measured on a real run
# (2026-08-19) was a 183-second image build, so this is an order of magnitude
# above the worst observed silence -- and far below an unattended overnight,
# which is the failure it exists to prevent.
STALL_TIMEOUT_S = _seconds("TEMPER_STALL_TIMEOUT_S", 15 * 60)

# 24 hours, matching the industry default for managed training jobs rather than
# a figure invented here. No run on the current 4B/8B catalog comes close: the
# measured training phase was 161 seconds.
MAX_JOB_DURATION_S = _seconds("TEMPER_MAX_JOB_DURATION_S", 24 * 60 * 60)
