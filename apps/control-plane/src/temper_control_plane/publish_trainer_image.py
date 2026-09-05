"""Build, publish and record the trainer image from the pipeline.

Issue #44. The pipeline workflow (`.github/workflows/image.yml`) runs this one
command: it assembles the same build context a job used to build from
(`trainer_build.TRAINER_SOURCES`), tags and pushes it to the container
registry, captures the digest, verifies the published image is pullable by
that digest, and rewrites the checked-in contract
(`packages/contracts/trainer-image.json`) that the orchestrator reads at
launch. The workflow then opens the pull request that lands the new digest,
which is what makes a digest change a deliberate, visible change rather than a
silent one.

The digest is the contract and the tag is a comment, exactly like the
Dockerfile's FROM line: the machine pulls `<image>@<digest>` -- never a tag --
so what runs is exactly what was built and verified here.

Nothing here installs packages into the image. The Dockerfile bakes the
entrypoint and the contract data; a `RUN pip install` would reintroduce the
dependency resolution problem the pinned base exists to avoid.

Only the build, registry and file operations live here; the git/PR dance that
lands the new digest stays in the workflow, where a reviewer can see it as
plain steps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .config import REPO_ROOT
from .trainer_build import write_context

CONTRACT = REPO_ROOT / "packages" / "contracts" / "trainer-image.json"
BODY_PATH = REPO_ROOT / ".build" / "publish-body.md"

# The fallback for a manual run where GITHUB_REPOSITORY is absent; in CI the
# name is derived from the Actions-provided repository, so a fork publishes
# into its own namespace rather than this one.
DEFAULT_IMAGE = "ghcr.io/thp728/temper/trainer"

# One comment for the contract whatever its state, so a publish rewrites the
# file deterministically rather than restating prose.
_CONTRACT_COMMENT = (
    "The published trainer image the orchestration pulls and runs. Written "
    "only by the pipeline's publish step (`.github/workflows/image.yml`, "
    "`temper_control_plane.publish_trainer_image`), which builds the image "
    "from `trainer_build.TRAINER_SOURCES`, pushes it, and verifies it is "
    "pullable by digest. The digest is the contract and the tag is a comment, "
    "exactly like the Dockerfile's FROM line: the machine pulls `<reference>` "
    "-- image@digest -- never a tag, so what runs is exactly what was built "
    "and verified by the pipeline. A digest change is a deliberate, visible "
    "change because it lands as its own pull request. When `published` is "
    "false there is no reference and a real job refuses loudly rather than "
    "pulling a made-up image."
)


def image_name() -> str:
    """The registry reference the image is published under.

    Computed from the Actions-provided `GITHUB_REPOSITORY` (owner/name) so the
    package lands in the right namespace, including on a fork; the literal
    default covers a manual local run where that variable is absent.
    """
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo and "/" in repo:
        return f"ghcr.io/{repo}/trainer"
    return DEFAULT_IMAGE


def _tag() -> str:
    """The tag for this publish. A comment only -- the digest is the contract."""
    sha = (os.environ.get("GITHUB_SHA") or "").strip()
    if sha:
        return f"main-{sha[:12]}"
    try:
        short = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "manual-unknown"
    return f"main-{short}" if short else "manual-unknown"


def _run(*args: str) -> subprocess.CompletedProcess:
    """One docker/registry command, failing loudly on a nonzero exit.

    Loudly means *with the reason*. `CalledProcessError`'s own message names
    the command and the exit code and nothing else, so a captured build that
    failed printed a traceback ending in the docker command line while the
    error docker actually reported -- the missing base, the exhausted disk,
    the refused push -- stayed in the captured stderr nobody read. Two
    publish runs were diagnosed as "docker build exited 1" because of it.
    The output is captured (these commands are chatty and the caller prints
    what matters), so failure has to re-raise it deliberately.
    """
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = "\n".join(
            part.strip() for part in (exc.stdout, exc.stderr) if part
        )
        raise RuntimeError(
            f"{' '.join(args)} exited {exc.returncode}\n{detail}"
        ) from exc


def build(tag: str) -> None:
    """Build the image from the same named sources a job used to build from.

    `write_context` is the single assembler both `just image` and this path
    call, so the pipeline can only ever publish what a local build would have
    produced -- the two cannot diverge.
    """
    with tempfile.TemporaryDirectory() as tmp:
        context = write_context(Path(tmp) / "trainer")
        _run("docker", "build", "-t", tag, str(context))


def push(tag: str) -> None:
    _run("docker", "push", tag)


def digest_of(image: str, tag: str) -> str:
    """The repository digest of the just-pushed image, as `image@sha256:...`.

    Read from the daemon's record of what was pushed (`RepoDigests`), filtered
    to the image name being published, so the reference the contract records
    is the one the registry accepted -- not a locally-invented string.

    The braces are doubled because that is what a Go template needs. Written
    once as `{json .RepoDigests}`, docker treated it as a literal and printed
    the string back verbatim, so this raised `JSONDecodeError` at character 1
    on every run -- after the build and the push had both already succeeded.
    That is why the image was never published despite the expensive half of
    the work completing each time.
    """
    out = _run(
        "docker",
        "inspect",
        "--format={{json .RepoDigests}}",
        f"{image}:{tag}",
    ).stdout.strip()
    digests = json.loads(out)
    matches = [d for d in digests if d.startswith(image + "@")]
    if not matches:
        raise RuntimeError(
            f"no repository digest for {image}:{tag}; RepoDigests was {digests}"
        )
    return matches[0]


def verify_pullable(reference: str) -> None:
    """Prove the published image resolves in the registry by digest.

    `docker manifest inspect` fetches the manifest for the exact digest from
    the registry -- the same resolution a `docker pull <reference>` would do --
    without re-downloading the ~8.5 GB of layers. A nonzero exit means the
    image is not pullable by the reference the product will use, and the
    publish fails rather than recording a reference nothing can pull.
    """
    _run("docker", "manifest", "inspect", reference)


def contract_document(image: str, tag: str, reference: str) -> dict:
    """The contract document for one publish. Deterministic for its inputs."""
    digest = reference.split("@", 1)[1] if "@" in reference else reference
    return {
        "_comment": _CONTRACT_COMMENT,
        "image": image,
        "published": True,
        "tag": tag,
        "digest": digest,
        "reference": reference,
    }


def write_contract(image: str, tag: str, reference: str) -> Path:
    """Rewrite `packages/contracts/trainer-image.json` for this publish."""
    document = contract_document(image, tag, reference)
    CONTRACT.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return CONTRACT


def write_body(image: str, tag: str, reference: str) -> Path:
    """The pull-request body the workflow attaches to the digest-change PR."""
    BODY_PATH.parent.mkdir(parents=True, exist_ok=True)
    BODY_PATH.write_text(
        "The pipeline rebuilt and re-published the trainer image.\n\n"
        f"- image: `{image}`\n"
        f"- tag: `{tag}` (a comment; the digest is the contract)\n"
        f"- reference: `{reference}`\n\n"
        "The machine pulls this reference by digest, so what runs is exactly "
        "what the publish workflow built and verified. The publish workflow "
        "confirmed the image is pullable by this digest (`docker manifest "
        "inspect`) before this pull request was opened.\n",
        encoding="utf-8",
    )
    return BODY_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=None,
        help="override the registry image name (default: derived from "
        "GITHUB_REPOSITORY)",
    )
    args = parser.parse_args(argv)

    image = args.image or image_name()
    tag = _tag()
    full = f"{image}:{tag}"
    print(f"building {full}")
    build(full)
    print(f"pushing {full}")
    push(full)
    reference = digest_of(image, tag)
    print(f"published {reference}")
    verify_pullable(reference)
    contract = write_contract(image, tag, reference)
    body = write_body(image, tag, reference)
    print(f"contract written to {contract}")
    print(f"pull-request body written to {body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
