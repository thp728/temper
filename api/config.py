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
            f"safety limit, so it is refused rather than ignored.") from None
    if value <= 0:
        raise ValueError(
            f"{name}={raw!r} must be greater than zero. To disable the limit, "
            f"set it to a value large enough to be unreachable — there is no "
            f"value that means 'no limit', because a limit that can be "
            f"switched off by a typo is not a limit.")
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
