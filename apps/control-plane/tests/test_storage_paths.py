"""Every path the product writes to sits under the repo root's `data/`.

The reorg broke this and nothing caught it. `db.DB_PATH` and `datasets.UPLOADS`
were `Path(__file__).parent.parent / "data"`, which meant one thing when this
module lived at `api/db.py` and something else entirely once it moved to
`apps/control-plane/src/temper_control_plane/db.py`. They resolved to
`apps/control-plane/src/data/`, three directories below where `.gitignore`
anchors `/data/`.

Two consequences, and the second is the bad one. The database and every
uploaded dataset moved without anyone asking, and they moved somewhere git
would happily commit them -- in a repository that goes public at submission,
carrying user training data.

The suite stayed green throughout, because every test monkeypatches these two
constants to a `tmp_path`. That is the right thing for a test to do and it is
exactly why nothing noticed: the only assertion that could have caught this is
one about the unpatched value, which did not exist. It does now.
"""

from __future__ import annotations

from temper_control_plane import config, datasets, db, orchestrator


def test_every_writable_path_is_under_the_repo_root_data_directory():
    for name, path in [
        ("db.DB_PATH", db.DB_PATH),
        ("datasets.UPLOADS", datasets.UPLOADS),
        ("orchestrator.ARTIFACTS", orchestrator.ARTIFACTS),
    ]:
        assert path.resolve().is_relative_to(
            (config.REPO_ROOT / "data").resolve()
        ), f"{name} resolves to {path}, outside the repo root's data directory"


def test_the_gitignore_that_protects_them_is_anchored_to_the_root():
    """A bare `data/` matches at any depth; the leading slash is what makes the
    rule mean the root and only the root. That anchoring is what turned a moved
    constant into a committable database rather than a merely misplaced one."""
    rules = (
        (config.REPO_ROOT / ".gitignore")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert "/data/" in rules, "the anchored data rule is gone from .gitignore"
