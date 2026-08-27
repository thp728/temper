"""Dataset import by reference (issue #45) at the HTTP seam.

The remote repository is a local double (`FakeRemoteDatasets`, the
datasets-seam equivalent of `FakeProvider`), seeded per test -- nothing here
reaches the network. Each acceptance criterion is exercised:

* a dataset imports by reference, optionally with a configuration and split;
* imported rows run through the identical validation path -- this file pins
  that by importing a dataset and an upload of the same bytes and asserting
  the two reports agree row for row;
* a repository that cannot be fetched surfaces the reason as a coded 400,
  before anything is stored;
* a split resolving to nothing is refused with that reason;
* an imported dataset that fails validation is still stored with its report;
* the size ceiling applies to imports exactly as it does to uploads.
"""

import json

import pytest
from helpers import wait_validated

from temper_control_plane import config, db, remote_datasets, storage
from temper_control_plane.fake_remote_datasets import (
    FakeRemoteDatasets,
    chat_rows,
)


@pytest.fixture()
def server(isolated, monkeypatch):
    from temper_control_plane import main, orchestrator

    monkeypatch.setattr(orchestrator, "launch", lambda job_id: None)
    return main


@pytest.fixture()
def client(server):
    from fastapi.testclient import TestClient

    with TestClient(server.app) as c:
        yield c


def import_reference(client, repo, config=None, split=None):
    return client.post(
        "/v1/datasets/import",
        json={"repo": repo, "config": config, "split": split},
    )


def with_resolver(monkeypatch, resolver):
    monkeypatch.setattr(remote_datasets, "RESOLVER", resolver)


# --- import by reference, optionally config and split -------------------------


def test_a_dataset_imports_by_reference(client, monkeypatch):
    with_resolver(
        monkeypatch,
        FakeRemoteDatasets({("acme/demo-chat", None, None): chat_rows(12)}),
    )

    r = import_reference(client, "acme/demo-chat")

    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "validating"
    assert body["filename"] == "acme/demo-chat/default/train.jsonl"
    record = wait_validated(client, body["id"])
    assert record["status"] == "valid"
    assert record["report"]["valid"] is True
    assert record["report"]["usable_rows"] == 12


def test_import_honours_an_explicit_config_and_split(client, monkeypatch):
    """Configuration and split are optional but honoured when given: the
    reference is resolved as (repo, config, split), not flattened into a
    repo name."""
    with_resolver(
        monkeypatch,
        FakeRemoteDatasets(
            {("acme/demo-chat", "sft", "validation"): chat_rows(11)}
        ),
    )

    r = import_reference(
        client, "acme/demo-chat", config="sft", split="validation"
    )

    assert r.status_code == 202
    body = r.json()
    assert body["filename"] == "acme/demo-chat/sft/validation.jsonl"
    record = wait_validated(client, body["id"])
    assert record["report"]["valid"] is True
    assert record["report"]["usable_rows"] == 11


def test_import_response_is_exactly_the_published_model(client, monkeypatch):
    from temper_control_plane.contracts_models import DatasetAccepted

    with_resolver(
        monkeypatch,
        FakeRemoteDatasets({("acme/demo-chat", None, None): chat_rows(12)}),
    )

    body = import_reference(client, "acme/demo-chat").json()

    parsed = DatasetAccepted.model_validate(body)
    assert set(body) == set(DatasetAccepted.model_fields)
    # The stored object address stays behind the seam, like every other row.
    wait_validated(client, parsed.id)


def test_imported_dataset_is_stored_under_the_seam(client, monkeypatch):
    """The fetched bytes land in object storage exactly as an upload's would,
    keyed the same way -- the import is the same ingest path, not a side
    store."""
    rows = chat_rows(12)
    with_resolver(
        monkeypatch, FakeRemoteDatasets({("acme/demo-chat", None, None): rows})
    )

    ds_id = import_reference(client, "acme/demo-chat").json()["id"]

    from temper_control_plane.remote_datasets import jsonl_chunks

    expected = b"".join(jsonl_chunks(rows))
    stored = storage.STORE.get(storage.dataset_key(ds_id))
    assert stored == expected
    # And validation read that same stored object (the seam reads what the
    # import wrote).
    wait_validated(client, ds_id)


# --- the identical validation path --------------------------------------------
# The issue's central sentence: imported rows go through exactly the same
# validation as an upload -- same schema detection, same line-numbered errors,
# same thinking-mode detection. The strongest proof is byte-for-byte: import a
# dataset and upload the very same JSONL, and the reports must agree.


def report_from_upload(client, rows):
    data = ("\n".join(json.dumps(r) for r in rows)).encode("utf-8")
    r = client.post("/v1/datasets", files={"file": ("d.jsonl", data)})
    assert r.status_code == 202
    return wait_validated(client, r.json()["id"])["report"]


def test_import_and_upload_of_the_same_rows_report_identically(
    client, monkeypatch
):
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(12)
    ]
    with_resolver(
        monkeypatch, FakeRemoteDatasets({("acme/demo-chat", None, None): rows})
    )

    imported = wait_validated(
        client, import_reference(client, "acme/demo-chat").json()["id"]
    )["report"]
    uploaded = report_from_upload(client, rows)

    assert imported == uploaded


def test_imported_thinking_mode_is_detected_not_chosen(client, monkeypatch):
    """Thinking-mode detection runs on imports too: an all-thinking import
    enables thinking, a mixed one blocks with its lines named -- the same
    behaviour, decided by the same pass."""

    def think(i: int) -> str:
        return f"{chr(60)}think{chr(62)}r{i}{chr(60)}/think{chr(62)}a{i}"

    rows = [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": think(i)},
            ]
        }
        for i in range(11)
    ]
    with_resolver(
        monkeypatch, FakeRemoteDatasets({("acme/thinky", None, None): rows})
    )

    report = wait_validated(
        client, import_reference(client, "acme/thinky").json()["id"]
    )["report"]

    assert report["valid"] is True
    assert report["enable_thinking"] is True


def test_imported_errors_name_their_lines(client, monkeypatch):
    """A remote dataset broken on some rows gets the same line-numbered errors
    an upload of those rows would -- the line number is the row's position in
    the split, which is exactly what a user can act on."""
    rows = chat_rows(11) + [{"content": "no messages list"}]
    with_resolver(
        monkeypatch, FakeRemoteDatasets({("acme/broken", None, None): rows})
    )

    report = wait_validated(
        client, import_reference(client, "acme/broken").json()["id"]
    )["report"]

    err = [e for e in report["errors"] if e["code"] == "missing_messages"]
    assert err and err[0]["line"] == 12


# --- fetch failures surface their reason --------------------------------------


def test_an_unfetchable_repository_is_refused_with_the_reason(
    client, monkeypatch
):
    with_resolver(monkeypatch, FakeRemoteDatasets({}))

    r = import_reference(client, "nope/nowhere")

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "dataset_not_found"
    assert "nope/nowhere" in detail["message"]
    # Refused before anything was stored: no row, no object.
    assert db.list_datasets() == []


def test_a_fetch_that_dies_mid_stream_surfaces_its_reason(client, monkeypatch):
    """A repository that cannot be fetched must not fail mid-job: even a
    failure discovered part-way through streaming answers as a coded 400 with
    the reason, and the half-written import is cleaned up -- the same contract
    a lying Content-Length earns on an upload."""
    from temper_control_plane.remote_datasets import RemoteDatasetError

    class DyingSource:
        repo, config, split = "acme/flaky", None, None

        def display_name(self):
            return "acme/flaky/default/train.jsonl"

        def stream(self):
            yield b'{"messages": [{"role": "user", "content": "q"}]}\n'
            raise RemoteDatasetError("fetch_failed", "rate limited mid-stream")

    class FlakyResolver:
        def resolve(self, repo, config=None, split=None):
            return DyingSource()  # type: ignore[return-value]

    with_resolver(monkeypatch, FlakyResolver())

    r = import_reference(client, "acme/flaky")

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "fetch_failed"
    assert "rate limited" in detail["message"]
    assert db.list_datasets() == []


# --- a split resolving to nothing is refused ----------------------------------


def test_a_split_that_resolves_to_nothing_is_refused_with_that_reason(
    client, monkeypatch
):
    with_resolver(
        monkeypatch,
        FakeRemoteDatasets({("acme/empty-split", None, "train"): []}),
    )

    r = import_reference(client, "acme/empty-split", split="train")

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "split_empty"
    assert "no rows" in detail["message"]
    assert "train" in detail["message"]
    assert db.list_datasets() == []


# --- a failing import is stored with its report -------------------------------


def test_an_import_that_fails_validation_is_stored_with_its_report(
    client, monkeypatch
):
    with_resolver(
        monkeypatch,
        FakeRemoteDatasets(
            {
                ("acme/demo-broken", None, None): [
                    {"content": "no messages list"},
                    {"messages": [{"role": "user", "content": "q"}]},
                ]
            }
        ),
    )

    ds_id = import_reference(client, "acme/demo-broken").json()["id"]
    record = wait_validated(client, ds_id)

    assert record["status"] == "invalid"
    codes = [e["code"] for e in record["report"]["errors"]]
    assert "missing_messages" in codes
    assert "no_assistant_turn" in codes
    # The failed import is still stored: the object the report describes is
    # the object the user is being told to fix.
    assert storage.STORE.get(storage.dataset_key(ds_id))


# --- the size ceiling applies to imports --------------------------------------
# The ceiling is a configured product limit (config.MAX_DATASET_BYTES, derived
# in ADR-0036 from measured throughput). Imports read the same value -- never
# a retyped number -- and enforce it mid-stream, since a remote reference has
# no declared size to refuse on ahead of time.


def test_the_size_ceiling_applies_to_imports(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_DATASET_BYTES", 128)
    with_resolver(
        monkeypatch,
        FakeRemoteDatasets({("acme/big", None, None): chat_rows(400)}),
    )

    r = import_reference(client, "acme/big")

    assert r.status_code == 413
    detail = r.json()["detail"]
    assert detail["code"] == "dataset_too_large"
    assert detail["limit_bytes"] == 128
    assert detail["actual_bytes"] > 128
    assert db.list_datasets() == []
