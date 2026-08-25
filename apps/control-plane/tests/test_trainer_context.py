"""The image is built from a named list, and the Dockerfile has to agree with it.

`thinking.py` was missing from the Dockerfile's COPY for weeks. The image built
fine and died at import on the GPU, before the `try/finally` that writes
`result.json`, so it surfaced to the orchestrator as `training_failed: "Trainer
produced no result.json"` -- an error pointing at training and saying nothing
about the image. It was caught by reading, not by paying for it.

The reorg made that failure easier to reach, not harder: `thinking.py` now lives
in `temper_core` because the domain validates thinking mode with it, so the
build context is assembled from two directories rather than one. These tests are
what keeps the two statements of that fact in step.
"""

from __future__ import annotations

import io
import re
import tarfile

from temper_control_plane import orchestrator


def _copied_by_the_dockerfile() -> set[str]:
    text = (orchestrator.TRAINER_DIR / "Dockerfile").read_text(
        encoding="utf-8"
    )
    line = re.search(r"^COPY\s+(.+?)\s+\S+/\s*$", text, re.M)
    assert line, "no COPY line found in the trainer Dockerfile"
    return set(line.group(1).split())


def test_every_source_shipped_is_a_source_the_image_copies():
    shipped = {
        p.name for p in orchestrator.TRAINER_SOURCES if p.suffix == ".py"
    }
    assert shipped == _copied_by_the_dockerfile()


def test_the_tarball_carries_every_named_source_flat():
    """Flat on purpose: the machine untars into one directory and builds there,
    so the archive's member names are the build context's file names."""
    with tarfile.open(
        fileobj=io.BytesIO(orchestrator._trainer_tarball()), mode="r:gz"
    ) as tar:
        names = set(tar.getnames())
    assert names == {p.name for p in orchestrator.TRAINER_SOURCES}


def test_the_module_the_domain_validates_with_is_the_module_the_image_runs():
    """One file, two consumers. A copy in the trainer directory would drift."""
    import temper_core.thinking as domain_side

    shipped = next(
        p for p in orchestrator.TRAINER_SOURCES if p.name == "thinking.py"
    )
    assert (
        shipped.resolve()
        == __import__("pathlib").Path(domain_side.__file__).resolve()
    )


def test_sources_are_normalised_to_lf():
    """A CRLF Dockerfile fails inside the container in ways that read as
    anything but a line-ending bug. .gitattributes is the first defence and
    this is the second."""
    with tarfile.open(
        fileobj=io.BytesIO(orchestrator._trainer_tarball()), mode="r:gz"
    ) as tar:
        for member in tar.getmembers():
            assert b"\r\n" not in tar.extractfile(member).read(), member.name


def test_the_local_build_context_matches_what_the_machine_receives(tmp_path):
    """`just image` and a real job must build the same thing, or "it worked
    locally" stops meaning anything about the job that is about to cost money."""
    from temper_control_plane import trainer_image

    context = trainer_image.write_context(tmp_path / "ctx")
    on_disk = {p.name: p.read_bytes() for p in context.iterdir()}

    with tarfile.open(
        fileobj=io.BytesIO(orchestrator._trainer_tarball()), mode="r:gz"
    ) as tar:
        in_transit = {
            m.name: tar.extractfile(m).read() for m in tar.getmembers()
        }

    assert on_disk == in_transit
