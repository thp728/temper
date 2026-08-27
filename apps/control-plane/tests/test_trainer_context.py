"""The image is built from a named list, and the Dockerfile has to agree with it.

`thinking.py` was missing from the Dockerfile's COPY for weeks. The image built
fine and died at import on the GPU, before the `try/finally` that writes
`result.json`, so it surfaced to the orchestrator as `training_failed: "Trainer
produced no result.json"` -- an error pointing at training and saying nothing
about the image. It was caught by reading, not by paying for it.

The reorg made that failure easier to reach, not harder: `thinking.py` now lives
in `temper_core` because the domain validates thinking mode with it, so the
build context is assembled from two directories rather than one. Since issue
#44 the machine no longer builds -- the pipeline does, from this same named
list -- so `write_context` is what both the local `just image` and the pipeline
publish step call, and the two cannot diverge. These tests are what keeps the
two statements of that fact in step.
"""

from __future__ import annotations

import re

from temper_control_plane import trainer_build


def _context_names(tmp_path) -> set[str]:
    """The build context's file names, exactly as a build would receive them."""
    return {
        p.name for p in trainer_build.write_context(tmp_path / "ctx").iterdir()
    }


def _copied_by_the_dockerfile() -> set[str]:
    text = (trainer_build.TRAINER_DIR / "Dockerfile").read_text(
        encoding="utf-8"
    )
    line = re.search(r"^COPY\s+(.+?)\s+\S+/\s*$", text, re.M)
    assert line, "no COPY line found in the trainer Dockerfile"
    return set(line.group(1).split())


def test_every_source_shipped_is_a_source_the_image_copies():
    """Everything in the context except the Dockerfile itself must be copied
    into the image -- data files included. The contract is the single
    definition read by the resolver path (#82, #83); a missed source would
    mean code and data no longer bake at one digest, or a container that
    dies at import because `thinking.py` was omitted."""
    shipped = {
        p.name for p in trainer_build.TRAINER_SOURCES if p.name != "Dockerfile"
    }
    assert shipped == _copied_by_the_dockerfile()


def test_the_context_carries_every_named_source_flat(tmp_path):
    """Flat on purpose: the context is one directory of flat names, so a build
    context file name is what the Dockerfile's COPY reads."""
    assert _context_names(tmp_path) == {
        p.name for p in trainer_build.TRAINER_SOURCES
    }


def test_the_module_the_domain_validates_with_is_the_module_the_image_runs():
    """One file, two consumers. A copy in the trainer directory would drift."""
    import temper_core.thinking as domain_side

    shipped = next(
        p for p in trainer_build.TRAINER_SOURCES if p.name == "thinking.py"
    )
    assert (
        shipped.resolve()
        == __import__("pathlib").Path(domain_side.__file__).resolve()
    )


def test_sources_are_normalised_to_lf(tmp_path):
    """A CRLF Dockerfile fails inside the container in ways that read as
    anything but a line-ending bug. .gitattributes is the first defence and
    this is the second."""
    context = trainer_build.write_context(tmp_path / "ctx")
    for path in context.iterdir():
        assert b"\r\n" not in path.read_bytes(), path.name


def test_the_build_context_is_deterministic(tmp_path):
    """Two assemblies of the same sources are the same bytes, so `just image`
    and the pipeline publish step -- both of which call `write_context` --
    cannot disagree about what is being built (issue #44)."""
    a = trainer_build.write_context(tmp_path / "a")
    b = trainer_build.write_context(tmp_path / "b")
    assert {p.name: p.read_bytes() for p in a.iterdir()} == {
        p.name: p.read_bytes() for p in b.iterdir()
    }
