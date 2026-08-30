"""The zero-cost tier's demonstration content (spec 012 / issue #63).

The seeding is what "something to show immediately" means: boot the zero-cost
tier against an empty database and a reviewer finds a sample dataset and one
completed run already there. These tests pin the three properties that make
that honest:

* seeding happens only in the zero-cost tier, and only on an empty database —
  it must never plant a fake run into a database a human is using, and never
  run in a mode where a launched job costs money;
* the seeded run is complete and shaped exactly like a run the journeys
  produce (a real `run_job` against the fake provider), with a downloadable
  artifact;
* the single mode setting is parsed as a boolean, so `=0` really means the
  real tier and the web shell's banner reads the same value through the same
  rule.
"""

import json

import pytest

from temper_control_plane import config, db, seed_demo, storage
from temper_control_plane.fake_provider import DEMO_ADAPTER_BYTES


def _chat_rows(n: int):
    return [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(n)
    ]


def _upload_bytes(rows) -> bytes:
    return "\n".join(json.dumps(r) for r in rows).encode("utf-8") + b"\n"


# --- the sample dataset -------------------------------------------------------


def test_the_checked_in_sample_dataset_is_valid_chat_jsonl():
    """The file a reviewer opens is the file the seeding stores — one
    definition, so the two cannot drift — and it is a usable chat dataset."""
    raw = seed_demo.SAMPLE_DATASET_PATH.read_bytes()
    assert raw.startswith(b'{"messages":')
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert len(rows) == 12
    for row in rows:
        assert set(row) == {"messages"}
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][-1]["role"] == "assistant"
    assert raw.endswith(b"\n")


# --- seeding behaviour ---------------------------------------------------------


def test_seeding_plants_a_valid_sample_dataset_and_a_completed_run(
    isolated, monkeypatch
):
    """The whole point of the seed: after `maybe_seed`, the database holds a
    sample dataset and one *complete* job with a downloadable artifact."""
    monkeypatch.setattr(config, "FAKE_PROVIDER", True)
    seed_demo.maybe_seed()

    datasets = db.list_datasets()
    assert len(datasets) == 1, datasets
    ds = datasets[0]
    assert ds["filename"] == seed_demo.SAMPLE_DATASET_FILENAME
    assert ds["status"] == "valid"
    # The stored object is byte-for-byte the checked-in sample.
    assert (
        storage.STORE.get(storage.dataset_key(ds["id"]))
        == seed_demo.SAMPLE_DATASET_PATH.read_bytes()
    )

    jobs = db.list_jobs(limit=None)
    assert len(jobs) == 1, jobs
    job = jobs[0]
    assert job["status"] == "complete", job
    assert job["dataset_id"] == ds["id"]
    # A frozen quote and measured actuals — the record of a real run, not a
    # hand-written row.
    assert job["quote"] is not None
    assert job["actuals"] is not None
    # The run asked for the optional delivery formats, so the finished record
    # shows the whole journey.
    assert set(job["delivery_request"]) == {"merged", "quantised"}

    # The artifact is downloadable, and is the demo adapter the fake shipped.
    members = db.canonical_artifact_members(job)
    assert members, "completed job must have an artifact"
    payload = b"".join(storage.STORE.get(key) for _name, key in members)
    assert DEMO_ADAPTER_BYTES in payload

    # The run's own history is present, not an empty shell.
    assert db.count_events(job["id"]) > 0
    assert db.get_progress(job["id"]), "a complete run has per-phase progress"


def test_seeding_is_a_no_op_outside_the_zero_cost_tier(isolated, monkeypatch):
    """The real tier seeds nothing: it has no demo to show, and a job created
    by the seed would be claimed by a worker and run on real hardware."""
    monkeypatch.setattr(config, "FAKE_PROVIDER", False)
    seed_demo.maybe_seed()
    assert db.list_datasets() == []
    assert db.list_jobs() == []


def test_seeding_never_touches_a_database_a_human_is_using(
    isolated, monkeypatch
):
    """Seeding is for the first boot. A database that already holds a dataset
    or a job belongs to someone who has been using the stack, and a fake run
    planted into their history would be the dishonesty the mode exists to
    avoid."""
    monkeypatch.setattr(config, "FAKE_PROVIDER", True)
    # A human's data already exists.
    ds_id = db.create_dataset("mine.jsonl", "keys/mine", ds_id="ds_mine")
    db.finish_dataset(ds_id, {"valid": True, "row_count": 1})
    db.create_job(ds_id, "qwen/Qwen3-4B", {}, quote=None)

    seed_demo.maybe_seed()

    datasets = db.list_datasets()
    assert [d["filename"] for d in datasets] == ["mine.jsonl"]
    jobs = db.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["status"] == "queued"  # the human's job, untouched


# --- the single mode setting is parsed as a boolean ----------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
    ],
)
def test_the_mode_flag_is_parsed_as_a_boolean(monkeypatch, value, expected):
    """`TEMPER_FAKE_PROVIDER` means what it is documented to mean: `=0` is the
    real tier. This is the same rule the web shell's banner reads, so a
    deployment that disables the fake cannot find its pages still marked as a
    demonstration (ADR-0071)."""
    if value:
        monkeypatch.setenv("TEMPER_FAKE_PROVIDER", value)
    else:
        monkeypatch.delenv("TEMPER_FAKE_PROVIDER", raising=False)
    assert config._flag("TEMPER_FAKE_PROVIDER", False) is expected


def test_the_mode_flag_refuses_a_value_it_cannot_honour(monkeypatch):
    monkeypatch.setenv("TEMPER_FAKE_PROVIDER", "perhaps")
    with pytest.raises(ValueError):
        config._flag("TEMPER_FAKE_PROVIDER", False)
