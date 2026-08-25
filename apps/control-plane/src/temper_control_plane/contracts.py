"""Write the API contract to packages/contracts, without starting a server.

The interface consumes a client generated from this document, so the document is
the seam between the two halves. Checking it in buys two things a live-server
generation would not: the web build needs no running control plane, in the
pipeline or on a laptop, and a change to the contract shows up in a diff where
it can be reviewed.

The cost is that the file can go stale, which has exactly one cause: someone
edits a response model and does not re-run this. `just check` fails that push.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import REPO_ROOT
from .main import app

OPENAPI_PATH = REPO_ROOT / "packages" / "contracts" / "openapi.json"


def write(destination: Path = OPENAPI_PATH) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"
    destination.write_text(document, encoding="utf-8", newline="\n")
    return destination


if __name__ == "__main__":
    print(write())
