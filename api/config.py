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
