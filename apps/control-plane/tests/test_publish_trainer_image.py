"""The pipeline's publish step and the contract it writes (issue #44).

The money-free halves of publishing: deriving the registry name, parsing the
pushed digest out of the daemon's record, the pullability check that proves
"the published image is pullable by the same digest the product references",
and the deterministic contract the orchestrator reads. The docker/registry
calls themselves run only in the pipeline workflow (they need a daemon, a
registry and credentials), so what is tested here is everything that is pure,
plus the reader's contract with the checked-in file.

The reader is `trainer_build.published_reference`; the writer is
`publish_trainer_image.write_contract`. They are tested as one pair because a
value two components must agree on is defined once and read, never retyped.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from temper_control_plane import publish_trainer_image as publish
from temper_control_plane import trainer_build

# --- registry name -----------------------------------------------------------


def test_image_name_derives_from_the_actions_repository(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/temper")
    assert publish.image_name() == "ghcr.io/acme/temper/trainer"


def test_image_name_falls_back_to_the_default(monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert publish.image_name() == publish.DEFAULT_IMAGE


# --- digest capture and pullability ------------------------------------------


def test_digest_of_reads_the_pushed_repo_digest(monkeypatch):
    image = "ghcr.io/acme/temper/trainer"
    digests = [f"{image}@sha256:{'a' * 64}"]
    monkeypatch.setattr(
        publish,
        "_run",
        lambda *a: subprocess.CompletedProcess(
            list(a), 0, stdout=json.dumps(digests)
        ),
    )
    assert publish.digest_of(image, "main-x") == digests[0]


def test_digest_of_asks_docker_for_a_real_go_template(monkeypatch):
    """The format argument must be a Go template, not a literal.

    Written once as `{json .RepoDigests}`, docker printed the string back
    verbatim and the publish died parsing it -- every time, after the build
    and the push had already succeeded, which is why the image went
    unpublished while the expensive half of the work kept completing. The
    other tests here stub `_run` and hand back valid JSON, so they assert the
    parsing and never the argument; this asserts the argument.
    """
    image = "ghcr.io/acme/temper/trainer"
    seen: list[tuple[str, ...]] = []

    def record(*args: str) -> subprocess.CompletedProcess:
        seen.append(args)
        return subprocess.CompletedProcess(
            list(args), 0, stdout=json.dumps([f"{image}@sha256:{'a' * 64}"])
        )

    monkeypatch.setattr(publish, "_run", record)
    publish.digest_of(image, "main-x")
    fmt = next(a for a in seen[0] if a.startswith("--format="))
    assert fmt == "--format={{json .RepoDigests}}"


def test_digest_of_refuses_an_image_with_no_repo_digest(monkeypatch):
    monkeypatch.setattr(
        publish,
        "_run",
        lambda *a: subprocess.CompletedProcess(
            list(a), 0, stdout=json.dumps(["ghcr.io/other@sha256:b"])
        ),
    )
    with pytest.raises(RuntimeError, match="no repository digest"):
        publish.digest_of("ghcr.io/acme/temper/trainer", "main-x")


def test_verify_pullable_inspects_the_registry_manifest(monkeypatch):
    calls = []

    def fake_run(*args):
        calls.append(args)
        return subprocess.CompletedProcess(list(args), 0)

    monkeypatch.setattr(publish, "_run", fake_run)
    reference = f"{publish.DEFAULT_IMAGE}@sha256:{'c' * 64}"
    publish.verify_pullable(reference)
    assert calls == [("docker", "manifest", "inspect", reference)]


# --- the contract the orchestrator reads -------------------------------------


def test_contract_document_carries_image_tag_digest_and_reference():
    doc = publish.contract_document("img", "main-x", "img@sha256:abcdef")
    assert doc["image"] == "img"
    assert doc["tag"] == "main-x"
    assert doc["digest"] == "sha256:abcdef"
    assert doc["reference"] == "img@sha256:abcdef"
    assert doc["published"] is True
    assert doc["_comment"]


def test_write_contract_is_deterministic(tmp_path, monkeypatch):
    target = tmp_path / "trainer-image.json"
    monkeypatch.setattr(publish, "CONTRACT", target)

    publish.write_contract("img", "main-x", "img@sha256:abc")
    first = target.read_bytes()
    publish.write_contract("img", "main-x", "img@sha256:abc")

    assert target.read_bytes() == first
    doc = json.loads(first)
    assert doc["reference"] == "img@sha256:abc"
    assert doc["published"] is True


def test_write_contract_carries_delivery_formats_forward_across_a_republish(
    tmp_path, monkeypatch
):
    """Which formats a digest can produce is measured on real hardware, not
    derived from the build -- so a republish that changes only the digest
    must not silently revert that measurement to "everything producible".
    """
    target = tmp_path / "trainer-image.json"
    target.write_text(
        json.dumps(
            {
                "_comment": "x",
                "_delivery_comment": "quantised needs the converter",
                "delivery_formats": ["adapter", "merged"],
                "image": "img",
                "published": True,
                "reference": "img@sha256:old",
                "tag": "main-old",
                "digest": "sha256:old",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(publish, "CONTRACT", target)

    publish.write_contract("img", "main-new", "img@sha256:new")

    doc = json.loads(target.read_text(encoding="utf-8"))
    assert doc["digest"] == "sha256:new"
    assert doc["delivery_formats"] == ["adapter", "merged"]
    assert doc["_delivery_comment"] == "quantised needs the converter"


def test_write_body_names_the_reference(tmp_path, monkeypatch):
    target = tmp_path / "publish-body.md"
    monkeypatch.setattr(publish, "BODY_PATH", target)

    publish.write_body("img", "main-x", "img@sha256:abc")

    body = target.read_text(encoding="utf-8")
    assert "img@sha256:abc" in body
    assert "main-x" in body


def _contract(tmp_path, **overrides) -> Path:
    doc = {
        "_comment": "fixture",
        "digest": None,
        "image": "img",
        "published": False,
        "reference": None,
        "tag": None,
    }
    doc.update(overrides)
    path = tmp_path / "trainer-image.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_published_reference_is_none_until_published(tmp_path, monkeypatch):
    monkeypatch.setattr(
        trainer_build, "PUBLISHED_IMAGE_CONTRACT", _contract(tmp_path)
    )
    assert trainer_build.published_reference() is None


def test_published_reference_is_the_image_at_the_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        trainer_build,
        "PUBLISHED_IMAGE_CONTRACT",
        _contract(
            tmp_path,
            published=True,
            digest="sha256:abc",
            reference="img@sha256:abc",
            tag="main-x",
        ),
    )
    assert trainer_build.published_reference() == "img@sha256:abc"


def test_a_published_doc_without_a_reference_is_treated_as_unpublished(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        trainer_build,
        "PUBLISHED_IMAGE_CONTRACT",
        _contract(tmp_path, published=True, reference=""),
    )
    assert trainer_build.published_reference() is None


def test_the_checked_in_contract_has_the_expected_shape():
    """The committed contract pins the honest initial state: unpublished, so a
    real job refuses loudly until the pipeline's first publish lands. A
    publish rewrites it via `write_contract`, so the shape here must be the
    shape the writer produces (same keys, reference = image@digest when
    published)."""
    doc = json.loads(
        trainer_build.PUBLISHED_IMAGE_CONTRACT.read_text(encoding="utf-8")
    )
    assert set(doc) == {
        "_comment",
        "_delivery_comment",
        "delivery_formats",
        "digest",
        "image",
        "published",
        "reference",
        "tag",
    }
    assert doc["image"].startswith("ghcr.io/")
    if doc["published"]:
        assert doc["reference"] == f"{doc['image']}@{doc['digest']}"
    else:
        assert doc["reference"] is None
        assert doc["digest"] is None
