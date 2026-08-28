"""Every writable *filesystem* location the product owns sits under the repo
root's `data/`.

The reorg broke this and nothing caught it. `db.DB_PATH` and the dataset
uploads directory were `Path(__file__).parent.parent / "data"`, which meant one
thing when this module lived at `api/db.py` and something else entirely once it
moved to `apps/control-plane/src/temper_control_plane/`. They resolved to
`apps/control-plane/src/data/`, three directories below where `.gitignore`
anchors `/data/`.

Two consequences, and the second is the bad one. The database and every
uploaded dataset moved without anyone asking, and they moved somewhere git
would happily commit them -- in a repository that goes public at submission,
carrying user training data.

The suite stayed green throughout, because every test monkeypatches these
constants to a `tmp_path`. That is the right thing for a test to do and it is
exactly why nothing noticed: the only assertion that could have caught this is
one about the unpatched value, which did not exist. It does now.

**A finding from issue #43.** This file's whole premise -- "every writable
location is a path under `/data/`" -- assumed the database was a file, which
was true for exactly as long as it was SQLite. `db.DATABASE_URL` is a
connection string to a networked server now, not a location on this
filesystem at all, so "is it under `/data/`" is not a question a URL can
answer; asking it would be testing the wrong shape of value. This file keeps
its original assertion for the one location it still applies to -- the
filesystem object store's root -- and adds the honest replacement for the
database: its address is *not* a local path, on purpose, and connects to
whatever `TEMPER_DATABASE_URL` names rather than anything this checkout owns
or `.gitignore` needs to protect.
"""

from __future__ import annotations

from temper_control_plane import config, db


def test_every_writable_filesystem_path_is_under_the_repo_root_data_directory():
    assert config.STORAGE_ROOT.resolve().is_relative_to(
        (config.REPO_ROOT / "data").resolve()
    ), (
        f"config.STORAGE_ROOT resolves to {config.STORAGE_ROOT}, outside "
        "the repo root's data directory"
    )


def test_the_database_address_is_a_connection_string_not_a_local_path():
    """The database is a server, addressed by URL -- there is no local file
    for `.gitignore`'s `/data/` rule to protect anymore, and no path for this
    file's own boundary check to apply to."""
    assert db.DATABASE_URL.startswith(("postgresql://", "postgres://"))


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
