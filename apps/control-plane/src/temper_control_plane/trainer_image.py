"""Build the trainer image locally, from the same sources the machine builds from.

The build context is assembled rather than pointed at: `entrypoint.py` and the
Dockerfile come from `apps/trainer`, and `thinking.py` comes from the domain,
because the control plane validates thinking mode with the module the image runs
and a second copy beside the entrypoint would be a hand-mirrored definition.

This exists so that `just image` and a real job build the same thing.
`trainer_build.TRAINER_SOURCES` is the single list. This writes it to a
directory, the orchestrator writes it to a tar, and both flatten to the same
names from the same bytes.

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

from .trainer_build import write_context

DEFAULT_TAG = "temper-trainer:local"


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
