"""The tokenizer seam: a dataset row's token count, under a real tokenizer.

Issue #42. The core counting pass takes a `dict -> int` `count_row` seam so
`temper_core` stays free of any tokenizer library; this module is the control
plane's side of that seam. It loads the default catalog model's tokenizer
(`catalog.DEFAULT_MODEL`, Qwen3-4B, pinned to the catalog's revision so the
counts are reproducible like the model facts are) and defines a row's token
count as the sum over every message's content -- every turn, not just the
assistant one, because the quote is priced per training token and the prompt
is in the context window (spike 9's own definition).

The tokenizer.json is public and unauthenticated, like the model facts
(`models.py`); it is downloaded once and cached under `/data/`, and the loaded
tokenizer is cached for the process. Tests replace the `TOKENIZER` seam
attribute with a fake, exactly as `conftest.py` replaces the quote's provider
and the model-facts resolver -- so no test reaches the network.

A tokenizer that cannot be loaded must not strand a dataset: loading raises,
and the counting phase records the failure as a coded one. The count stays
null and the quote renders "—", the absent-estimate posture ADR-0031 documents
for a count that does not exist yet.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from temper_core import catalog

from . import config

# The seam. Production leaves it None and loads the real tokenizer on first
# use; tests replace it with a fake object exposing
# `encode(text, add_special_tokens=...) -> {ids: [...]}` (see
# `apps/control-plane/tests/fake_tokenizer.py`), mirroring how the quote's
# provider and the model facts seam are replaced in conftest.
TOKENIZER: Any = None

_loaded: Any = None


def _model() -> catalog.BaseModel:
    return catalog.CATALOG[catalog.DEFAULT_MODEL]


def _tokenizer_path() -> Path:
    """Where the default model's tokenizer.json is cached, under `/data/`
    with the rest of the runtime-written state."""
    m = _model()
    return (
        config.REPO_ROOT
        / "data"
        / "tokenizers"
        / f"{m.repo.replace('/', '_')}-{m.revision[:12]}.json"
    )


def _download() -> Path:
    """Fetch the pinned tokenizer.json once; a fresh clone with no cache
    downloads it, like the model facts are fetched on first resolve."""
    path = _tokenizer_path()
    if path.exists():
        return path
    m = _model()
    url = (
        f"https://huggingface.co/{m.repo}/resolve/{m.revision}/tokenizer.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # url is built from the catalog's own pinned repo and revision, never
    # from user input; the download is public and unauthenticated.
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        path.write_bytes(resp.read())
    return path


def tokenizer() -> Any:
    """The active tokenizer: the seam's fake when one is set, else the real
    one, loaded once and cached for the process."""
    global _loaded
    if TOKENIZER is not None:
        return TOKENIZER
    if _loaded is None:
        from tokenizers import Tokenizer

        _loaded = Tokenizer.from_file(str(_download()))
    return _loaded


def count_row_for(messages_field: str = "messages") -> Callable[[dict], int]:
    """The `count_row` the core counting pass calls: one parsed row -> the sum
    of its message contents' token counts under the active tokenizer."""
    tok = tokenizer()

    def count_row(obj: dict) -> int:
        total = 0
        for m in obj.get(messages_field) or []:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                total += len(
                    tok.encode(m["content"], add_special_tokens=False).ids
                )
        return total

    return count_row
