"""Build the trainer image locally, from the same sources the machine builds from.

The build context is assembled rather than pointed at: `entrypoint.py` and the
Dockerfile come from `apps/trainer`, and `thinking.py` comes from the domain,
because the control plane validates thinking mode with the module the image runs
and a second copy beside the entrypoint would be a hand-mirrored definition.

This exists so that `just image` and a real job build the same thing.
`orchestrator.TRAINER_SOURCES` is the single list; this writes it to a directory
and the orchestrator writes it to a tar, and both flatten to the same names.

Issue #44 moves the published build into the pipeline. This stays afterwards as
the local path, because "build it and see" should not require a pull request.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .orchestrator import TRAINER_SOURCES

DEFAULT_TAG = "temper-trainer:local"


def write_context(destination: Path) -> Path:
    """Flatten every named source into one directory and return it."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in TRAINER_SOURCES:
        # Same normalisation as the tar: a CRLF Dockerfile fails inside the
        # container in ways that read as anything but a line-ending bug.
        text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
        (destination / source.name).write_text(
            text, encoding="utf-8", newline="\n"
        )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="write the build context and print it, without building",
    )
    args = parser.parse_args(argv)

    if args.context_only:
        print(write_context(Path(".build") / "trainer"))
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        context = write_context(Path(tmp) / "trainer")
        if shutil.which("docker") is None:
            print(
                "docker is not on PATH; wrote no image. "
                f"Context would have been: {sorted(p.name for p in context.iterdir())}",
                file=sys.stderr,
            )
            return 1
        return subprocess.run(
            ["docker", "build", "-t", args.tag, str(context)]
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
