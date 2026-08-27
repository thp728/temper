"""The measured ("actual") half of issue #77 -- `temper_core.actuals`.

What the comparison rests on: a job's own record yields three measured figures
(duration twice -- wall total and per-stage -- and peak memory) and one derived
one (cost from measured duration times the frozen rate), with the measured/
derived distinction preserved in the shape rather than in a comment. A good
test asserts on what `measure` returns for a given record, never on how the
record was produced -- the same function records a real run's terminal state
and a test's canned events.

Four properties are load-bearing and each is asserted directly:

* **Duration is measured, twice.** The wall total comes from the row's own
  timestamps; each stage's duration is the gap between its state event and
  the next one, so a stage never entered records None rather than a guess.
* **Peak memory is measured, or honestly absent.** It comes from the
  trainer's result document; a run with no result records None, never a
  number invented for it.
* **Cost is derived, and marked so.** Measured duration times the frozen
  hourly rate, converted into the currency's smallest unit the same way the
  quote does -- one definition of the minor unit, never retyped.
* **The job's stages are not the quote's phases.** `measure` records what
  the machine actually did (`preparing` bundles readiness, image pull and
  model download); reconciling the two vocabularies is
  `temper_core.calibration`'s job, in the one place they meet.
"""

from __future__ import annotations

from temper_core import actuals


def job(**overrides) -> dict:
    """A completed job row, shaped as the orchestrator reads it back."""
    base = {
        "id": "job_abc",
        "created_at": 1000.0,
        "finished_at": 1360.0,
        "status": "complete",
        "price_per_hour": 41.31,
        "currency": "INR",
        "storage_cost_usd_per_hour": 0.10
        * 100
        / 730,  # 100 GB floor, documented rate
    }
    base.update(overrides)
    return base


def state_event(ts: float, message: str, id: int = 1) -> dict:
    # The control plane records the state it entered as data (db.set_state),
    # so a state event says which state it entered even when its message is
    # the human narration ("Selecting hardware", not "provisioning").
    return {
        "id": id,
        "job_id": "job_abc",
        "ts": ts,
        "kind": "state",
        "message": message,
        "data": {"state": message},
    }


# The state events a small successful run appends, in order. created_at=1000;
# provisioning begins 1000.0, preparing 1005.0, training 1060.0, packaging
# 1300.0, complete (terminal) 1360.0.
def run_events() -> list[dict]:
    return [
        state_event(1000.0, "queued", 1),
        state_event(1000.0, "provisioning", 2),
        state_event(1005.0, "preparing", 3),
        state_event(1060.0, "training", 4),
        state_event(1300.0, "packaging", 5),
        state_event(1360.0, "complete", 6),
    ]


# --- duration is measured, twice ---------------------------------------------


def test_wall_duration_is_finished_minus_created():
    m = actuals.measure(job(), run_events(), None)
    assert m.duration_s == 360.0


def test_each_stage_duration_is_the_gap_to_the_next_state_transition():
    m = actuals.measure(job(), run_events(), None)
    by_name = {p.name: p.duration_s for p in m.phases}
    assert by_name == {
        "provisioning": 5.0,  # 1000 -> 1005
        "preparing": 55.0,  # 1005 -> 1060
        "training": 240.0,  # 1060 -> 1300
        "packaging": 60.0,  # 1300 -> 1360 (terminal)
    }


def test_a_stage_never_reached_measures_none_not_a_guess():
    # Cancelled during provisioning: no preparing event follows.
    events = [
        state_event(1000.0, "queued", 1),
        state_event(1000.0, "provisioning", 2),
        state_event(1007.0, "cancelled", 3),
    ]
    m = actuals.measure(
        job(status="cancelled", finished_at=1007.0), events, None
    )
    by_name = {p.name: p.duration_s for p in m.phases}
    assert by_name["provisioning"] == 7.0
    assert by_name["preparing"] is None
    assert by_name["training"] is None
    assert by_name["packaging"] is None


def test_a_run_with_no_finished_at_measures_no_duration():
    m = actuals.measure(job(finished_at=None), run_events(), None)
    assert m.duration_s is None
    # But the stages it passed through are still measurable from its events.
    assert m.phases[1].duration_s == 55.0


def test_the_state_name_reads_from_data_not_from_the_narration():
    """A state event's message is the human line ("Selecting hardware");
    the state it entered is the data. Reading the message would measure
    nothing on a real run, which is the bug this shape exists to prevent."""
    events = [
        {
            "id": 1,
            "ts": 1000.0,
            "kind": "state",
            "message": "Selecting hardware",
            "data": {"state": "provisioning"},
        },
        {
            "id": 2,
            "ts": 1055.0,
            "kind": "state",
            "message": "Machine 4242 running; waiting for SSH",
            "data": {"state": "preparing"},
        },
        {
            "id": 3,
            "ts": 1060.0,
            "kind": "state",
            "message": "complete",
            "data": {"state": "complete"},
        },
    ]
    m = actuals.measure(job(finished_at=1060.0), events, None)
    by_name = {p.name: p.duration_s for p in m.phases}
    assert by_name["provisioning"] == 55.0
    assert by_name["preparing"] == 5.0


def test_an_event_without_data_falls_back_to_its_message():
    """Older rows or hand-built events carry no data; the message is the
    state name then (create_job writes "queued" with no narration)."""
    events = [
        {
            "id": 1,
            "ts": 1000.0,
            "kind": "state",
            "message": "queued",
            "data": None,
        },
        {
            "id": 2,
            "ts": 1005.0,
            "kind": "state",
            "message": "provisioning",
            "data": None,
        },
        {
            "id": 3,
            "ts": 1010.0,
            "kind": "state",
            "message": "preparing",
            "data": None,
        },
    ]
    m = actuals.measure(job(finished_at=1010.0), events, None)
    by_name = {p.name: p.duration_s for p in m.phases}
    assert by_name["provisioning"] == 5.0


# --- peak memory is measured, or honestly absent ------------------------------


def test_peak_memory_comes_from_the_trainers_result_document():
    m = actuals.measure(
        job(), run_events(), {"ok": True, "peak_memory_gb": 5.31}
    )
    assert m.peak_memory_gb == 5.31


def test_a_run_with_no_result_records_no_peak():
    m = actuals.measure(job(), run_events(), None)
    assert m.peak_memory_gb is None


def test_a_result_without_a_peak_records_none_not_a_zero():
    m = actuals.measure(job(), run_events(), {"ok": True})
    assert m.peak_memory_gb is None


# --- cost is derived, never measured ------------------------------------------


def test_cost_is_measured_duration_times_the_frozen_rate():
    # 360s at INR 41.31/hr = 360/3600 * 41.31 = 4.131 INR = 413.1 paisa.
    m = actuals.measure(job(), run_events(), None)
    assert m.cost_minor == 413
    assert m.currency == "INR"


def test_cost_carries_the_minor_unit_definition_from_the_quote():
    # A job billed in USD derives in cents against the same minor-unit rule.
    m = actuals.measure(
        job(currency="USD", price_per_hour=0.5), run_events(), None
    )
    assert m.cost_minor == 5  # 360/3600 * 0.5 = 0.05 USD = 5 cents


def test_no_duration_or_no_rate_derives_no_cost():
    assert (
        actuals.measure(job(finished_at=None), run_events(), None).cost_minor
        is None
    )
    assert (
        actuals.measure(
            job(price_per_hour=None), run_events(), None
        ).cost_minor
        is None
    )


def test_storage_is_derived_against_the_separate_usd_rate():
    # Two hours, so the floor-priced disk accrues a visible USD line:
    # 2 * (0.10*100/730) USD * 100 cents = ~2.74 cents.
    m = actuals.measure(job(finished_at=1000.0 + 7200.0), run_events(), None)
    assert m.storage_cost_usd_minor == 3


def test_no_storage_rate_derives_no_storage_line():
    m = actuals.measure(
        job(storage_cost_usd_per_hour=None), run_events(), None
    )
    assert m.storage_cost_usd_minor is None


# --- shape --------------------------------------------------------------------


def test_to_dict_matches_the_frozen_shape():
    m = actuals.measure(
        job(),
        run_events(),
        {"ok": True, "peak_memory_gb": 5.31},
    )
    d = actuals.to_dict(m)
    assert d["duration_s"] == 360.0
    assert d["peak_memory_gb"] == 5.31
    assert d["cost_minor"] == 413
    assert d["currency"] == "INR"
    assert [p["name"] for p in d["phases"]] == list(actuals.MEASURED_STAGES)
    assert d["phases"][2] == {"name": "training", "duration_s": 240.0}
