"""The feasibility warning — spec 002, issue #10.

The size limit and the duration ceiling answer different questions and do not
reconcile: 1 GB is within memory and far beyond what finishes in 24 hours.
Job creation warns while the user can still act on it. A warning, not a
refusal: the estimate is crude, and a wrong block is worse than a wrong
warning.

Two seams, per the spec's testing decisions:

* the estimate itself, as pure functions over numbers — no I/O, no framework,
  so it survives the Phase B migration unchanged;
* job creation, through the HTTP seam — a dataset that plainly cannot finish
  yields a warning **and a launched job**, never a refusal.
"""

import json

import pytest
from fastapi.testclient import TestClient

from temper_control_plane import config  # noqa: E402
from temper_core import feasibility  # noqa: E402

# --- the estimate ------------------------------------------------------------


def test_throughput_is_the_measured_one_not_a_round_number():
    """Derived from the one real run of 2026-08-19: 64 rows x 3 epochs =
    192 row-passes in the measured 161.4s training phase (trainer/README).
    If this ever becomes a chosen number, the derivation comment in
    feasibility.py has rotted."""
    assert feasibility.ROWS_PER_SECOND == pytest.approx(192 / 161.4)


def test_estimate_is_rows_times_epochs_at_measured_throughput():
    est = feasibility.estimated_duration_s(usable_rows=100, hyperparams={})
    assert est == pytest.approx(300 / feasibility.ROWS_PER_SECOND)


def test_estimate_honours_a_num_epochs_override():
    est = feasibility.estimated_duration_s(
        usable_rows=100, hyperparams={"num_epochs": 1}
    )
    assert est == pytest.approx(100 / feasibility.ROWS_PER_SECOND)


def test_estimate_ignores_a_junk_num_epochs():
    """The trainer refuses bad overrides later; the estimate must not crash
    on one before that."""
    est = feasibility.estimated_duration_s(
        usable_rows=100, hyperparams={"num_epochs": "lots"}
    )
    assert est == pytest.approx(300 / feasibility.ROWS_PER_SECOND)


def test_no_estimate_when_max_steps_caps_the_run():
    """With max_steps set, run length is bounded by steps, not data volume --
    a data-volume estimate would be wrong by orders of magnitude, and no
    per-step throughput was ever measured. No estimate, so no warning."""
    assert (
        feasibility.estimated_duration_s(
            usable_rows=10**9, hyperparams={"max_steps": 10}
        )
        is None
    )


# --- the warning -------------------------------------------------------------


def test_usable_rows_prefers_the_validation_report():
    ds = {"report": {"usable_rows": 90}, "row_count": 100}
    assert feasibility.usable_rows(ds) == 90


def test_usable_rows_falls_back_and_never_guesses_upward():
    assert feasibility.usable_rows({"row_count": 100}) == 100
    assert feasibility.usable_rows({}) == 0


def test_warning_fires_when_estimate_exceeds_the_ceiling():
    w = feasibility.warning(
        usable_rows=10**6,
        hyperparams={},
        max_duration_s=config.MAX_JOB_DURATION_S,
    )
    assert w is not None
    assert w["code"] == "duration_feasibility"
    assert w["estimated_duration_s"] > config.MAX_JOB_DURATION_S


def test_warning_names_itself_an_estimate_wherever_it_would_be_shown():
    w = feasibility.warning(
        usable_rows=10**6,
        hyperparams={},
        max_duration_s=config.MAX_JOB_DURATION_S,
    )
    assert "estimate" in w["message"].lower()
    assert "measured" in w["message"].lower()


def test_no_warning_when_the_run_plausibly_fits():
    assert (
        feasibility.warning(
            usable_rows=500,
            hyperparams={},
            max_duration_s=config.MAX_JOB_DURATION_S,
        )
        is None
    )


def test_warning_at_exact_equality_does_not_fire():
    """The estimate is crude; a boundary value is inside its own error bar.
    Only plainly-over fires."""
    rows = int(
        config.MAX_JOB_DURATION_S
        * feasibility.ROWS_PER_SECOND
        / feasibility.DEFAULT_EPOCHS
    )
    assert (
        feasibility.warning(
            usable_rows=rows,
            hyperparams={},
            max_duration_s=config.MAX_JOB_DURATION_S,
        )
        is None
    )


def test_default_epochs_is_pinned_to_the_trainer_default():
    """Duplicated from the trainer entrypoint on purpose -- the domain does not
    import application code -- but the duplication is pinned here, so a trainer
    default change fails this test instead of silently skewing every estimate.

    Issue #82 removes the duplication itself; until then this is what makes it
    safe."""
    import entrypoint

    assert feasibility.DEFAULT_EPOCHS == entrypoint.DEFAULTS["num_epochs"]


# --- job creation, through the HTTP seam --------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from temper_control_plane import datasets, db, main, orchestrator

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(datasets, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(orchestrator, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(orchestrator, "launch", lambda job_id: None)
    db.init()
    datasets.UPLOADS.mkdir(parents=True, exist_ok=True)
    with TestClient(main.app) as c:
        yield c


def chat_rows(n):
    for i in range(n):
        yield {
            "messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ]
        }


def upload(client, rows):
    data = ("\n".join(json.dumps(r) for r in rows)).encode("utf-8")
    r = client.post("/v1/datasets", files={"file": ("d.jsonl", data)})
    assert r.status_code == 201
    return r.json()["id"]


def test_large_dataset_yields_a_warning_and_a_launched_job(
    client, monkeypatch
):
    """The acceptance criterion, verbatim: a large dataset yields a warning
    and a launched job, not a refusal. The ceiling is pulled down to seconds
    so a generated fixture -- never a committed one -- trips it."""
    monkeypatch.setattr(config, "MAX_JOB_DURATION_S", 1.0)
    ds_id = upload(client, chat_rows(50))

    r = client.post("/v1/jobs", json={"dataset_id": ds_id})

    assert r.status_code == 201, "a warning must not veto the user's judgement"
    body = r.json()
    assert body["status"] == "queued"
    warnings = body["warnings"]
    assert len(warnings) == 1
    assert warnings[0]["code"] == "duration_feasibility"
    assert "estimate" in warnings[0]["message"].lower()


def test_warning_is_on_the_job_afterwards_and_in_its_history(
    client, monkeypatch
):
    """Shown wherever the job is shown, and recorded as an event, so the
    warning the user accepted is part of the run's own account of itself."""
    monkeypatch.setattr(config, "MAX_JOB_DURATION_S", 1.0)
    ds_id = upload(client, chat_rows(50))
    job_id = client.post("/v1/jobs", json={"dataset_id": ds_id}).json()["id"]

    fetched = client.get(f"/v1/jobs/{job_id}").json()
    assert fetched["warnings"][0]["code"] == "duration_feasibility"

    events = client.get(f"/v1/jobs/{job_id}/events").json()["events"]
    assert any("estimate" in (e["message"] or "").lower() for e in events)


def test_normal_dataset_gets_no_warning(client):
    ds_id = upload(client, chat_rows(50))

    r = client.post("/v1/jobs", json={"dataset_id": ds_id})

    assert r.status_code == 201
    assert r.json()["warnings"] == []
