"""What the trainer image is built from, and how those files are prepared.

One module because there are two build paths and they must not diverge. A real
job flattens these sources into a tar and builds on the machine; `just image`
flattens them into a directory and builds here. If "it worked locally" is to
mean anything about a job that is about to cost money, both have to start from
the same bytes.

This lives beside the orchestrator rather than inside it because the orchestrator
is about running jobs, and it was already being edited for build-context reasons
that have nothing to do with that.
"""

from __future__ import annotations

from pathlib import Path

from .config import REPO_ROOT

TRAINER_DIR = REPO_ROOT / "apps" / "trainer"

# Named rather than globbed.
#
# Globbing the trainer directory was fine when it held nothing but the image's
# sources. It now also holds a pyproject, a README and this project's
# instructions, none of which belong in a build context. Naming them also makes
# the list checkable against the Dockerfile's COPY, which is the specific
# failure this has already had: `thinking.py` was missing from the COPY for
# weeks, and because it is a top-level import the container died before the
# `finally` that writes result.json, surfacing as "trainer produced no
# result.json" -- an error pointing at training and saying nothing about the
# image. `test_trainer_context.py` asserts the two agree.
#
# `thinking.py` comes from the domain rather than from beside the entrypoint:
# the control plane validates thinking mode with the same module the image runs,
# and a second copy beside the entrypoint is the kind of hand-mirrored
# definition ADR-0010 forbids.
#
# `trainer-defaults.json` is the single definition of the trainer's defaults
# (#82). The control plane reads it through `temper_core.hyperparams` to
# resolve the spec before launch (#83); the trainer validates that spec, and
# the image ships the same file so data and code bake at one digest once the
# pipeline builds it.
#
# `axolotl-schema.json` (issue #33) is the pinned image's own configuration
# schema. The trainer's known-key set is derived from it -- a key unknown to
# the trainer is refused loudly and echoed back -- and it ships in the image so
# the trainer's guard and the generated surface read the same universe.
TRAINER_SOURCES = (
    TRAINER_DIR / "Dockerfile",
    TRAINER_DIR / "entrypoint.py",
    REPO_ROOT / "packages" / "core" / "src" / "temper_core" / "thinking.py",
    REPO_ROOT / "packages" / "contracts" / "trainer-defaults.json",
    REPO_ROOT / "packages" / "contracts" / "axolotl-schema.json",
)


def normalised(source: Path) -> bytes:
    """Read a source with LF endings, whichever way it sits on this disk.

    A CRLF Dockerfile fails inside the container in ways that read as anything
    but a line-ending bug. `.gitattributes` is the first defence; this is the
    second, and it stays because a checkout is not the only way a file gets
    onto this machine.
    """
    return source.read_text(encoding="utf-8").replace("\r\n", "\n").encode()


def write_context(destination: Path) -> Path:
    """Flatten every named source into one directory and return it."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in TRAINER_SOURCES:
        (destination / source.name).write_bytes(normalised(source))
    return destination
