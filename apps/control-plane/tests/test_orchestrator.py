"""The money-spending path, exercised without spending money.

Every test here drives a job through the HTTP seam with a fake provider
injected. What is asserted is what a user or an operator can observe: the job
record, and the ordered event log. How many times an internal helper was called
is not behaviour. The one exception is teardown, where the observable outcome
*is* an event and the fact that destroy was attempted is the thing under test.
"""

import hashlib
import io
import json
import re
import tarfile
import time
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from temper_control_plane.fake_provider import (
    MACHINE_ID,
    FakeClock,
    FakeProvider,
    simulated_limits,
)
from temper_control_plane.limits import RunLimits

ADAPTER_BYTES = b"weights"
RESULT = {
    "ok": True,
    "stage": "train",
    "adapter_path": "run/adapter_model.safetensors",
    "adapter_sha256": hashlib.sha256(ADAPTER_BYTES).hexdigest(),
    "adapter_config": {"r": 16, "lora_alpha": 32},
}
TRAINING_LINES = [
    "[10:00:01] building trainer image",
    "[10:03:04] image built in 183s",
    "[10:03:04] running training",
    "33%|###       | 10/30 [00:20<00:40,  2.00s/it]",
    "{'loss': 1.9042, 'grad_norm': 1.5, 'epoch': 0.5}",
]


def chat(user, assistant):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


class Harness:
    """Upload a dataset, launch a job on a given provider, read it back."""

    def __init__(self, client, monkeypatch, tmp_path):
        self._client = client
        self._monkeypatch = monkeypatch
        self._tmp_path = tmp_path

    def run(self, provider, hyperparameters=None, limits=None) -> str:
        from temper_control_plane import orchestrator

        # Run inline rather than on a thread: the job is the system under test,
        # so the test should observe its finished state rather than race it.
        self._monkeypatch.setattr(
            orchestrator,
            "launch",
            lambda job_id: orchestrator.run_job(
                job_id, provider=provider, limits=limits
            ),
        )
        return self._create(hyperparameters)

    def run_on_a_thread(
        self, provider, hyperparameters=None, limits=None
    ) -> str:
        """Start the job and return while it is still going.

        The inline form above cannot answer this ticket's question: output that
        appears only once a job has finished is indistinguishable from output
        that appeared while it was working, unless the test reads mid-flight.
        It is also the only form in which a job can be cancelled at all — a
        request that cancels one has to arrive while it is running.
        """
        from temper_control_plane import orchestrator

        on_a_thread = orchestrator.launch  # before it is replaced below
        self._monkeypatch.setattr(
            orchestrator,
            "launch",
            lambda job_id: on_a_thread(
                job_id, provider=provider, limits=limits
            ),
        )
        return self._create(hyperparameters)

    def queued_job(self) -> str:
        """A job created and never started, as one is between the two."""
        from temper_control_plane import orchestrator

        self._monkeypatch.setattr(orchestrator, "launch", lambda job_id: None)
        return self._create()

    def _create(self, hyperparameters=None) -> str:
        path = self._tmp_path / "d.jsonl"
        path.write_text(
            "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(12)),
            encoding="utf-8",
        )
        with open(path, "rb") as f:
            ds = self._client.post(
                "/v1/datasets", files={"file": (path.name, f)}
            ).json()["id"]

        r = self._client.post(
            "/v1/jobs",
            json={"dataset_id": ds, "hyperparameters": hyperparameters or {}},
        )
        assert r.status_code == 201
        return r.json()["id"]

    def job(self, job_id) -> dict:
        return self._client.get(f"/v1/jobs/{job_id}").json()

    def events(self, job_id) -> list[dict]:
        return self._client.get(f"/v1/jobs/{job_id}/events").json()["events"]

    def messages(self, job_id) -> list[str]:
        return [e["message"] for e in self.events(job_id)]

    def poll_until_terminal(self, job_id) -> tuple[list[dict], list[dict]]:
        """Read events the way a client does: incrementally, by last id seen.

        Returns the events that were provably appended while the job was still
        working, and everything that was polled. A client that has to wait for
        a terminal status before it can read anything is exactly the four
        minutes of silence being fixed.

        The status is read *after* each page, not before: an event in a page
        fetched before a status that still said "working" was appended before
        that status was read, so it cannot be output that only arrived at the
        end. Reading the status first proves nothing -- the job may have
        finished in between.
        """
        from temper_control_plane import db

        during, everything, last = [], [], 0
        deadline = time.time() + POLL_DEADLINE_S
        while time.time() < deadline:
            page = self._client.get(
                f"/v1/jobs/{job_id}/events", params={"after": last}
            ).json()
            batch = page["events"]
            assert all(e["id"] > last for e in batch), (
                "after= re-delivered events the client already held"
            )
            last = page["last_id"]
            everything += batch
            if self.job(job_id)["status"] not in db.TERMINAL_STATES:
                during += batch
            elif not batch:
                return during, everything
            time.sleep(0.01)
        raise AssertionError(
            f"job never reached a terminal state: {everything}"
        )


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    from temper_control_plane import datasets, db, main, orchestrator

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(datasets, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(orchestrator, "ARTIFACTS", tmp_path / "artifacts")
    # Teardown retries sleep between attempts. Tests do not need to.
    monkeypatch.setattr(orchestrator, "DESTROY_RETRY_DELAY_S", 0)
    with TestClient(main.app) as c:
        yield Harness(c, monkeypatch, tmp_path)


def index_of(events, predicate) -> int:
    for i, e in enumerate(events):
        if predicate(e):
            return i
    raise AssertionError("no event matched")


def terminal_index(events) -> int:
    """Where the job reported that it was over."""
    states = [i for i, e in enumerate(events) if e["kind"] == "state"]
    return states[-1]


# --- the whole loop, no GPU ------------------------------------------------


def test_a_job_runs_to_completion_against_a_fake_provider(harness):
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "complete"
    assert job["gpu_type"] == "L4"
    assert job["currency"] == "INR"
    assert job["machine_id"] == MACHINE_ID
    assert job["result"]["adapter_path"] == RESULT["adapter_path"]

    # The artifact is the adapter *and* the config that makes it loadable.
    directory = Path(job["adapter_path"]).parent
    assert (
        directory / "adapter_model.safetensors"
    ).read_bytes() == ADAPTER_BYTES
    assert (
        json.loads((directory / "adapter_config.json").read_text())
        == RESULT["adapter_config"]
    )

    assert provider.destroyed
    # An injected provider belongs to whoever injected it; the job does not
    # close a resource it did not open.
    assert not provider.closed


def test_scripted_output_lines_reach_the_event_log(harness):
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider)
    messages = harness.messages(job_id)
    for line in TRAINING_LINES:
        assert line in messages
    # The result marker and the JSON behind it are machinery, not output.
    assert "---RESULT---" not in messages


def test_the_job_spec_reaches_the_machine(harness):
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider, hyperparameters={"lora_r": 32})
    script = provider.script.decode("utf-8")
    assert job_id in script
    assert '"lora_r": 32' in script
    # Sources and dataset each travel as one binary payload rather than one
    # round trip per file or an inline encoding.
    assert len(provider.pushed) == 2


# --- dataset transport: bytes, not hex ---------------------------------------


def upload_raw(harness, data: bytes, name="d.jsonl") -> str:
    """Upload arbitrary bytes and return the dataset id, asserting validity."""
    r = harness._client.post("/v1/datasets", files={"file": (name, data)})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["valid"], body["errors"]
    return body["id"]


def launch_dataset(harness, provider, dataset_id) -> str:
    from temper_control_plane import orchestrator

    harness._monkeypatch.setattr(
        orchestrator,
        "launch",
        lambda job_id: orchestrator.run_job(job_id, provider=provider),
    )
    r = harness._client.post("/v1/jobs", json={"dataset_id": dataset_id})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def pushed_dataset_bytes(provider) -> bytes:
    """What arrived at the far end of the transport, as the machine sees it.

    Scans everything that was pushed for an archive carrying the dataset,
    rather than assuming which push or which member name — the assertion is
    about the bytes, not about the container they travelled in.
    """
    found = []
    for _dest, payload in provider.pushed:
        try:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".jsonl"):
                        found.append(tar.extractfile(member).read())
        except tarfile.ReadError:
            continue
    assert len(found) == 1, (
        f"expected exactly one dataset in the pushed payloads, got {len(found)}"
    )
    return found[0]


def crlf_dataset() -> bytes:
    rows = [json.dumps(chat(f"q{i}", f"a{i}")) for i in range(12)]
    return ("\r\n".join(rows) + "\r\n").encode("utf-8")


def run_dataset(harness, data: bytes):
    """Upload bytes, launch a job on a fresh fake provider, finish it."""
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = launch_dataset(harness, provider, upload_raw(harness, data))
    assert harness.job(job_id)["status"] == "complete"
    return provider


def test_the_dataset_reaches_the_machine_byte_identical(harness):
    """The ticket: user bytes arrive exactly as uploaded."""
    data = crlf_dataset().replace(b"a11", "a11 ☕ café".encode())
    provider = run_dataset(harness, data)
    assert pushed_dataset_bytes(provider) == data


def test_a_dataset_containing_carriage_returns_arrives_unmodified(harness):
    """The line-ending hazard, pinned.

    The sources archive normalises CRLF, and rightly so -- a stray carriage
    return breaks a Dockerfile in ways that read as anything but a
    line-ending bug. The same rewriting applied to user data silently changes
    every row of it.
    """
    data = crlf_dataset()
    provider = run_dataset(harness, data)

    arrived = pushed_dataset_bytes(provider)
    assert arrived == data
    assert b"\r\n" in arrived, "the fixture itself carried no carriage returns"


def test_a_dataset_containing_non_ascii_text_arrives_unmodified(harness):
    data = (
        "\n".join(
            json.dumps(chat(f"frage {i}?", f"Antwort ☕ 日本語 {i}"))
            for i in range(12)
        )
        + "\n"
    ).encode("utf-8")
    provider = run_dataset(harness, data)
    assert pushed_dataset_bytes(provider) == data


def test_the_dataset_is_not_embedded_in_the_remote_script(harness):
    """Hex-in-script was the defect: twice the size as text, several copies
    in memory, and a mechanism forty lines away already shipping binaries."""
    data = crlf_dataset()
    provider = run_dataset(harness, data)

    script = provider.script.decode("utf-8")
    assert data.hex() not in script
    assert data.decode("utf-8") not in script


def test_training_numbers_arrive_as_metric_events(harness):
    """The loss has to be a number in a field, not a substring of a log line.

    Kept distinct from log output so that a chart is a read-only addition
    later, rather than a second pass over prose that has since changed shape.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider)

    metrics = [
        e["data"] for e in harness.events(job_id) if e["kind"] == "metric"
    ]
    assert metrics == [{"loss": 1.9042, "epoch": 0.5}]


def test_narration_and_progress_bars_stay_log_output(harness):
    """Only the framework's own log dict is a measurement.

    The bar is in the scripted output on purpose: an undescribed tqdm bar is
    the line most likely to be promoted by mistake, because the evaluation bar
    looks exactly like the training one.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider)

    kinds = {e["message"]: e["kind"] for e in harness.events(job_id)}
    assert kinds["[10:03:04] running training"] == "log"
    assert kinds["33%|###       | 10/30 [00:20<00:40,  2.00s/it]"] == "log"


# --- teardown ordering: the bug this ticket fixes ---------------------------


def test_destroy_confirmation_precedes_the_terminal_state(harness):
    """A client that stops polling on a terminal status still sees teardown.

    Asserted explicitly because it is the regression: teardown used to run in a
    `finally` that executed after the terminal transition, so the one
    confirmation an operator most wants arrived after everyone stopped reading.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider)

    events = harness.events(job_id)
    destroyed = index_of(events, lambda e: "destroyed" in (e["message"] or ""))
    assert destroyed < terminal_index(events)
    assert events[terminal_index(events)] is events[-1], (
        "the terminal state must be the last word on a job"
    )


def test_teardown_confirmation_precedes_the_terminal_state_on_failure(harness):
    provider = FakeProvider(fail_at="push", fail_code="source_upload_failed")
    job_id = harness.run(provider)

    events = harness.events(job_id)
    assert harness.job(job_id)["status"] == "failed"
    destroyed = index_of(events, lambda e: "destroyed" in (e["message"] or ""))
    assert destroyed < terminal_index(events)


def test_teardown_runs_when_the_run_raises_unexpectedly(harness):
    """The path nobody anticipated still destroys the machine."""
    provider = FakeProvider(fail_at="await_ready", fail_unexpectedly=True)
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "internal_error"
    assert (
        "RuntimeError" in job["error_message"]
        or "exploded" in job["error_message"]
    )
    assert provider.destroyed


def test_teardown_is_confirmed_by_listing_not_by_the_destroy_call(harness):
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    harness.run(provider)
    assert "list_machine_ids" in provider.calls, (
        "a destroy call's return value is a claim, not evidence"
    )


def test_a_machine_that_survives_teardown_is_reported_loudly(harness):
    provider = FakeProvider(
        fail_at="push", destroy_failures=99, stays_listed=True
    )
    job_id = harness.run(provider)

    errors = [
        e["message"] for e in harness.events(job_id) if e["kind"] == "error"
    ]
    assert any("STRAY" in m and str(MACHINE_ID) in m for m in errors)
    assert provider.destroy_attempts > 1, "one transient error is not give-up"


def test_a_transient_destroy_failure_is_retried_rather_than_given_up_on(
    harness,
):
    """One flaky provider call must not be what leaves a machine billing."""
    provider = FakeProvider(fail_at="push", destroy_failures=1)
    job_id = harness.run(provider)

    assert provider.destroy_attempts == 2
    assert provider.destroyed
    errors = [
        e["message"] for e in harness.events(job_id) if e["kind"] == "error"
    ]
    assert any("Destroy attempt failed" in m for m in errors)
    assert not any("STRAY" in m for m in errors)


def test_nothing_is_destroyed_when_no_machine_was_created(harness):
    provider = FakeProvider(
        fail_at="select_gpu", fail_code="provider_capacity_unavailable"
    )
    job_id = harness.run(provider)

    assert harness.job(job_id)["error_code"] == "provider_capacity_unavailable"
    assert "destroy" not in provider.calls


# --- failure at each stage --------------------------------------------------


@pytest.mark.parametrize(
    "stage,code",
    [
        ("select_gpu", "provider_capacity_unavailable"),
        ("create", "provider_capacity_unavailable"),
        ("await_ready", "ssh_unreachable"),
        ("await_ready", "ssh_auth_failed"),
        ("push", "source_upload_failed"),
        ("stream", "training_failed"),
    ],
)
def test_failure_at_a_stage_fails_the_job_with_its_code(harness, stage, code):
    provider = FakeProvider(fail_at=stage, fail_code=code)
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == code
    assert job["error_message"]
    assert job["adapter_path"] is None
    # A machine only exists from `create` onwards; before that there is
    # nothing to tear down, and after it there always is.
    if stage in ("select_gpu", "create"):
        assert "destroy" not in provider.calls
    else:
        assert provider.destroyed, "a created machine is always destroyed"


def test_a_provider_that_stops_producing_output_fails_the_job(harness):
    """No result marker means no result, whatever the output said."""
    provider = FakeProvider(lines=TRAINING_LINES, result=RESULT, stop_after=2)
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "training_failed"
    assert provider.destroyed


def test_a_trainer_that_reports_failure_fails_the_job(harness):
    provider = FakeProvider(
        lines=["[10:00:01] BUILD FAILED"],
        result={
            "ok": False,
            "stage": "build",
            "error": "no space left on device",
        },
    )
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "training_failed"
    assert "no space left on device" in job["error_message"]


def test_a_failure_before_training_keeps_its_own_error_code(harness):
    """An unpacking failure is not a training failure.

    The machine reports the stage that failed and the code that names it;
    telling a user their training failed sends them to read the wrong logs.
    """
    provider = FakeProvider(
        lines=["[10:00:01] SOURCE UNPACK FAILED"],
        result={
            "stage": "source",
            "ok": False,
            "error_code": "source_upload_failed",
            "error": "The trainer sources did not unpack.",
        },
    )
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "source_upload_failed"


def test_a_corrupt_adapter_download_is_refused(harness):
    """The container reported a hash; a truncated transfer must not pass."""
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=b"truncated"
    )
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "artifact_corrupt"


def test_an_unreadable_adapter_is_reported_without_failing_the_job(harness):
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=b""
    )
    job_id = harness.run(provider)

    job = harness.job(job_id)
    assert job["status"] == "complete"
    assert job["adapter_path"] is None
    assert any(
        "fetch failed" in (e["message"] or "").lower()
        for e in harness.events(job_id)
        if e["kind"] == "error"
    )


# --- the default provider ---------------------------------------------------


def test_missing_credentials_fail_the_job_before_anything_is_provisioned(
    harness, monkeypatch
):
    """No provider passed means the real one — which refuses without a key."""
    from temper_control_plane import config, orchestrator

    monkeypatch.setattr(config, "provider_credentials_present", lambda: False)
    monkeypatch.setattr(orchestrator, "launch", orchestrator.run_job)

    path = harness._tmp_path / "creds.jsonl"
    path.write_text(
        "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(12)),
        encoding="utf-8",
    )
    with open(path, "rb") as f:
        ds = harness._client.post(
            "/v1/datasets", files={"file": (path.name, f)}
        ).json()["id"]
    job_id = harness._client.post("/v1/jobs", json={"dataset_id": ds}).json()[
        "id"
    ]

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "provider_unauthenticated"
    assert job["machine_id"] is None


def test_the_suite_refuses_to_build_a_real_provider_client():
    """The guard itself is tested: a test that escapes its fake must fail."""
    from temper_control_plane import config, provider

    original = config.provider_credentials_present
    config.provider_credentials_present = lambda: True
    try:
        with pytest.raises(AssertionError, match="real provider client"):
            provider.new_provider()
    finally:
        config.provider_credentials_present = original


# --- the output channel: lines arrive while the job is running --------------

# Slow enough that a test can read between lines, fast enough that the suite
# does not notice. Real output is minutes apart; the shape is what matters.
LINE_DELAY_S = 0.05
POLL_DEADLINE_S = 20


def _command_block(script: str, command: str) -> str:
    """The named command with its line continuations, as one string."""
    lines = script.splitlines()
    start = next(i for i, line in enumerate(lines) if command in line)
    block = [lines[start]]
    while block[-1].rstrip().endswith("\\"):
        block.append(lines[start + len(block)])
    return "\n".join(block)


def streaming_provider():
    """A provider whose output is spaced out in time, as real output is."""
    return FakeProvider(
        lines=TRAINING_LINES,
        result=RESULT,
        adapter_bytes=ADAPTER_BYTES,
        line_delay=LINE_DELAY_S,
    )


def test_output_reaches_the_control_plane_while_the_job_is_still_running(
    harness,
):
    """The ticket, in one assertion.

    Before this, the whole build-and-train phase was one blocking call: 251
    measured seconds in which a user could not tell a working job from a hung
    one, because nothing left the machine until it was over.
    """
    job_id = harness.run_on_a_thread(streaming_provider())

    during, _ = harness.poll_until_terminal(job_id)
    arrived_early = [e["message"] for e in during]
    assert set(TRAINING_LINES) & set(arrived_early), (
        f"no training output arrived before the job ended: {arrived_early}"
    )
    assert harness.job(job_id)["status"] == "complete"


def test_events_are_retrievable_incrementally_while_the_job_runs(harness):
    """`after=` is a cursor, not a snapshot: nothing repeated, nothing lost."""
    job_id = harness.run_on_a_thread(streaming_provider())

    _, polled = harness.poll_until_terminal(job_id)
    ids = [e["id"] for e in polled]
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    # What a client assembled by polling is what one arriving at the end sees.
    assert polled == harness.events(job_id)


def test_each_event_is_stamped_when_its_line_was_read(harness):
    """Every event used to carry the same timestamp.

    They were all written after the remote command returned, so the record
    could not say how long anything took — which is a second failure on top of
    the silence, and the one that survives into the job's history.
    """
    job_id = harness.run(streaming_provider())

    events = harness.events(job_id)
    stamps = [e["ts"] for e in events]
    assert stamps == sorted(stamps), "the log is not in the order it happened"

    output = [e["ts"] for e in events if e["message"] in TRAINING_LINES]
    assert len(output) == len(TRAINING_LINES)
    assert len(set(output)) == len(output), "output lines share a timestamp"
    assert all(b - a >= LINE_DELAY_S * 0.5 for a, b in pairwise(output)), (
        "timestamps do not reflect when each line actually arrived"
    )


def test_the_machine_does_not_park_container_output_in_a_file(harness):
    """The outermost redirection, asserted because it silently undoes the rest.

    Normally the remote script's text is not something to assert on. This is
    the exception, and it is recorded as one in ADR-0001: a `> /tmp/run.log`
    reintroduced here would hold every line on the machine until the job was
    over, and every test above would still pass, because the fake provider
    yields lines the script never touched.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    harness.run(provider)
    script = provider.script.decode("utf-8")

    # A redirection to a path is the thing that closes the channel. `1>&2` and
    # `2>&1` are not redirections to a file -- they fold the container's two
    # streams into the one the provider reads.
    to_a_file = re.compile(r">\s*(?![&\s])")
    for command in ("docker build", "docker run"):
        block = _command_block(script, command)
        assert not to_a_file.search(block), (
            f"{command} output is being parked in a file on the machine: {block}"
        )

    assert "PYTHONUNBUFFERED=1" in _command_block(script, "docker run"), (
        "without this the container buffers and the channel is closed again"
    )


# --- runtime limits: a job that stops making progress stops itself ----------


def test_a_silent_job_is_killed_and_reports_that_it_stalled(harness):
    """The machine is up and the connection is open; nothing is coming.

    This is what a wedged trainer looks like from the control plane, and it is
    the case a wall-clock constant nobody read used to pretend to cover.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, silent_after=2
    )
    job_id = harness.run(
        provider, limits=simulated_limits(step=60.0, stall=900.0)
    )

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "gpu_stalled"
    assert "900" in job["error_message"]
    assert provider.destroyed, "a stalled job must not leave a machine billing"
    assert job["adapter_path"] is None


def test_a_job_that_runs_too_long_is_killed_and_says_so_differently(harness):
    """Still talking, just far past the ceiling. A different code on purpose.

    The clock advances well under the stall timeout per line, so the stall
    detector keeps resetting and this can only be the ceiling firing.
    """
    provider = FakeProvider(
        lines=[f"step {i}" for i in range(200)], result=RESULT
    )
    job_id = harness.run(
        provider,
        limits=simulated_limits(step=300.0, stall=900.0, maximum=3600.0),
    )

    job = harness.job(job_id)
    assert job["status"] == "failed"
    assert job["error_code"] == "gpu_max_duration_exceeded"
    assert "3600" in job["error_message"]
    assert provider.destroyed


def test_the_two_limits_are_configuration_not_constants(harness, monkeypatch):
    """Turning the knob down changes the behaviour, with nothing else touched.

    A limit that can only be changed by editing the module is the constant this
    ticket deleted, wearing a different name.
    """
    from temper_control_plane import config

    monkeypatch.setattr(config, "STALL_TIMEOUT_S", 120.0)
    monkeypatch.setattr(config, "MAX_JOB_DURATION_S", 86400.0)
    limits = replace(
        RunLimits.from_config(now=FakeClock(step=60.0)), poll_interval_s=0.005
    )

    job_id = harness.run(
        FakeProvider(lines=TRAINING_LINES, result=RESULT, silent_after=2),
        limits=limits,
    )

    job = harness.job(job_id)
    assert job["error_code"] == "gpu_stalled"
    assert "120" in job["error_message"], (
        "the configured value is the one used"
    )


def test_the_stall_detector_says_so_when_a_long_silence_ends(harness):
    """Observable rather than silent: a survived gap leaves a record.

    Not one event per line. The event log is the user's view of their own run,
    and a bookkeeping entry per training step would bury the training output it
    exists to make legible.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(
        provider, limits=simulated_limits(step=400.0, stall=900.0)
    )

    assert harness.job(job_id)["status"] == "complete"
    resumed = [
        e
        for e in harness.events(job_id)
        if "Output resumed" in (e["message"] or "")
    ]
    assert resumed, "a silence longer than a quarter of the budget went unsaid"
    assert resumed[0]["data"]["stall_timeout_s"] == 900.0
    assert resumed[0]["data"]["silence_s"] >= 400.0
    # That it reports *only* long gaps is the companion test below, and the
    # threshold itself is pinned in the guard's own tests.


def test_a_healthy_run_is_not_narrated_by_the_stall_detector(harness):
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider, limits=simulated_limits(step=1.0))

    assert harness.job(job_id)["status"] == "complete"
    assert not any(
        "Output resumed" in (e["message"] or "")
        for e in harness.events(job_id)
    )


def test_the_unread_wall_clock_constant_is_gone(harness):
    """A control that looks implemented and is not is worse than none at all.

    Asserted rather than trusted to review: the constant was readable, plausibly
    named, and never once consulted, and nothing about reading the module said
    so.
    """
    from temper_control_plane import orchestrator

    assert not hasattr(orchestrator, "MAX_GPU_MINUTES")


def test_a_job_stopped_by_a_limit_does_not_claim_teardown_it_has_not_done(
    harness,
):
    """The message is written before the machine is destroyed, so it cannot
    say the machine *has been* destroyed.

    The stray-machine path is real and tested: a destroy call can fail and the
    machine can go on being listed. A failure message asserting teardown as a
    completed fact would then be the one thing an operator trusted and the one
    thing that was false.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, silent_after=2
    )
    job_id = harness.run(
        provider, limits=simulated_limits(step=60.0, stall=900.0)
    )

    message = harness.job(job_id)["error_message"]
    assert "has been destroyed" not in message
    assert "being destroyed" in message
    # And the confirmation it points at is really there, before the job ends.
    events = harness.events(job_id)
    destroyed = index_of(events, lambda e: "destroyed" in (e["message"] or ""))
    assert destroyed < terminal_index(events)


def test_the_transport_backstop_cannot_pre_empt_the_duration_ceiling(harness):
    """The limit that fires must be the one that has a name.

    The SSH transport carries its own watchdog on the streaming subprocess. It
    was 90 minutes -- below a 24-hour ceiling -- so against a real machine the
    ceiling could never have fired, and a long job would have failed with an
    uncoded `TimeoutExpired` instead of `gpu_max_duration_exceeded`. The fake
    provider has no watchdog, so no other test here can see this.
    """
    from temper_control_plane import config
    from temper_control_plane import provider as provider_module

    assert provider_module._stream_timeout() > config.MAX_JOB_DURATION_S


# --- cancellation: the user's own decision to stop --------------------------


# The guard looks at the clock, the limits and the cancel flag once per trip
# round its loop, and blocks for a poll interval in between. In production that
# is a second; here it is small only so that these tests do not each spend one
# proving something that has nothing to do with time.
def responsive_limits() -> RunLimits:
    return replace(RunLimits.from_config(), poll_interval_s=0.005)


def cancel(harness, job_id):
    return harness._client.post(f"/v1/jobs/{job_id}/cancel")


def cancelled_at(harness, provider) -> dict:
    """Run a job, cancel it where the provider is paused, read it back.

    Paused rather than timed: a test that sleeps and hopes it caught the job
    mid-flight passes on an idle machine and fails on a loaded one, and the
    three points this exercises are minutes apart on a real run.
    """
    job_id = harness.run_on_a_thread(provider, limits=responsive_limits())
    assert provider.wait_until_paused(), "the provider never reached its pause"
    r = cancel(harness, job_id)
    assert r.status_code == 200, r.text
    provider.resume.set()
    harness.poll_until_terminal(job_id)
    return harness.job(job_id)


def test_cancelling_before_a_machine_exists_stops_the_job_cleanly(harness):
    """Cancellation is not a privilege of jobs that got as far as training."""
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, pause_at_stage="select_gpu"
    )
    job = cancelled_at(harness, provider)

    assert job["status"] == "cancelled"
    assert job["adapter_path"] is None
    assert provider.created == [], "no machine should have been provisioned"
    assert "destroy" not in provider.calls, "there was nothing to destroy"


def test_cancelling_during_provisioning_destroys_the_machine(harness):
    """The request lands inside the call that creates the machine.

    Nothing is interruptible inside a provider call, so this is the case that
    costs money: `create` runs to completion and hands back a machine that is
    already unwanted. What must not happen is that machine outliving the
    request, and the check at the next boundary is what stops it.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, pause_at_stage="create"
    )
    job = cancelled_at(harness, provider)

    assert job["status"] == "cancelled"
    assert provider.created, "the paused call still returned a machine"
    assert provider.destroyed, (
        "a cancelled job must not leave a machine billing"
    )
    assert job["adapter_path"] is None


def test_cancelling_while_waiting_for_ssh_destroys_the_machine(harness):
    """The longest wait before any output: minutes, with the machine billing.

    Cancellation has to be available here, or it is unavailable during the
    slowest part of a job that has not produced a single line yet.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, pause_at_stage="await_ready"
    )
    job = cancelled_at(harness, provider)

    assert job["status"] == "cancelled"
    assert job["machine_id"] == MACHINE_ID
    assert provider.destroyed, (
        "a cancelled job must not leave a machine billing"
    )
    assert "push" not in provider.calls, (
        "a cancelled job does not go on setting itself up"
    )


def test_cancelling_during_the_image_build_destroys_the_machine(harness):
    """Two lines in, the machine is building the image and nothing is trained."""
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, pause_at_line=2
    )
    job = cancelled_at(harness, provider)

    assert job["status"] == "cancelled"
    assert provider.destroyed
    assert job["adapter_path"] is None
    assert "fetch" not in provider.calls, (
        "nothing is retrieved from a cancelled job"
    )
    assert TRAINING_LINES[-1] not in harness.messages(job["id"]), (
        "the job went on producing output after it was cancelled"
    )


def test_cancelling_during_training_destroys_the_machine(harness):
    """Past the build, with the trainer talking and lines still to come.

    Deliberately not the last line: a pause at the end of the scripted output
    leaves the streaming loop one trip from finishing on its own, and a test
    that cannot tell cancellation from arriving at the end proves nothing. The
    final line never being logged is what says the job was abandoned.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES,
        result=RESULT,
        adapter_bytes=ADAPTER_BYTES,
        pause_at_line=3,
    )
    job = cancelled_at(harness, provider)

    assert job["status"] == "cancelled"
    assert provider.destroyed
    assert job["adapter_path"] is None
    assert job["result"] is None, (
        "a cancelled job has no result document to report"
    )
    assert TRAINING_LINES[-1] not in harness.messages(job["id"])


def test_cancelling_while_the_adapter_is_being_retrieved_produces_none(
    harness,
):
    """The last window in which an adapter could still appear.

    Training is over, and packaging is a download from a machine that is still
    billing — so `packaging` is a state a cancellation genuinely arrives in. A
    request answered with "no adapter will be produced" that then produced one
    would be the worst thing this path could tell a user, so what was already
    fetched is discarded rather than kept.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES,
        result=RESULT,
        adapter_bytes=ADAPTER_BYTES,
        pause_at_stage="fetch",
    )
    job = cancelled_at(harness, provider)

    assert job["status"] == "cancelled"
    assert job["adapter_path"] is None
    assert provider.destroyed
    assert not list(
        (harness._tmp_path / "artifacts").rglob("*.safetensors")
    ), "a cancelled job left an adapter on disk"


def test_a_cancelled_job_is_not_recorded_as_a_failure(harness):
    """The user's own decision is not a defect, and must not read as one."""
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, pause_at_line=2
    )
    job = cancelled_at(harness, provider)

    assert job["status"] == "cancelled"
    assert job["error_code"] is None
    assert job["error_message"] is None


def test_the_teardown_confirmation_precedes_a_cancelled_job_ending(harness):
    """The same ordering guarantee as every other terminal outcome.

    A user who cancels is the one most likely to stop reading the moment the
    job says it is over, and the most likely to want proof billing stopped.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, pause_at_line=2
    )
    job_id = harness.run_on_a_thread(provider, limits=responsive_limits())
    assert provider.wait_until_paused()
    cancel(harness, job_id)
    provider.resume.set()
    harness.poll_until_terminal(job_id)

    events = harness.events(job_id)
    destroyed = index_of(events, lambda e: "destroyed" in (e["message"] or ""))
    assert destroyed < terminal_index(events)


def test_cancelling_says_plainly_that_no_adapter_will_be_produced(harness):
    """The consequence is stated where it is chosen, not discovered later.

    Cancellation is destructive by decision: a partially trained adapter handed
    to someone who asked to stop invites them to mistake it for a finished
    model. Saying so is what makes that defensible rather than surprising.
    """
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, pause_at_line=2
    )
    job_id = harness.run_on_a_thread(provider, limits=responsive_limits())
    assert provider.wait_until_paused()

    body = cancel(harness, job_id).json()
    assert "no adapter" in body["message"].lower()
    assert body["cancel_requested"] is True
    assert body["status"] in (
        "queued",
        "provisioning",
        "preparing",
        "training",
        "cancelled",
    )

    provider.resume.set()
    harness.poll_until_terminal(job_id)


def test_cancelling_twice_succeeds_quietly(harness):
    """A double-clicked button is not an error condition."""
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, pause_at_line=2
    )
    job_id = harness.run_on_a_thread(provider, limits=responsive_limits())
    assert provider.wait_until_paused()

    first, second = cancel(harness, job_id), cancel(harness, job_id)
    assert first.status_code == second.status_code == 200
    assert second.json()["cancel_requested"] is True

    provider.resume.set()
    harness.poll_until_terminal(job_id)
    assert harness.job(job_id)["status"] == "cancelled"
    # The request is idempotent; its record is not repeated either.
    requests = [
        e
        for e in harness.events(job_id)
        if "Cancellation requested" in (e["message"] or "")
    ]
    assert len(requests) == 1


def test_cancelling_a_finished_job_is_refused_with_a_stable_code(harness):
    """Nothing was undone, and the refusal says so in a code, not in prose."""
    provider = FakeProvider(
        lines=TRAINING_LINES, result=RESULT, adapter_bytes=ADAPTER_BYTES
    )
    job_id = harness.run(provider)
    assert harness.job(job_id)["status"] == "complete"

    r = cancel(harness, job_id)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "job_already_terminal"

    job = harness.job(job_id)
    assert job["status"] == "complete"
    assert job["adapter_path"], (
        "a refused cancellation must not have touched the finished job"
    )


def test_cancelling_an_unknown_job_is_a_404(harness):
    assert cancel(harness, "job_nosuchthing").status_code == 404


def test_a_job_cancelled_before_it_starts_never_provisions_anything(harness):
    """The flag is honoured before the provider is even built.

    A job sitting in `queued` behind a thread that has not started is the one
    moment at which cancelling could plausibly still cost money.
    """
    from temper_control_plane import db, orchestrator

    provider = FakeProvider(lines=TRAINING_LINES, result=RESULT)
    job_id = harness.queued_job()
    assert db.request_cancel(job_id) == "accepted"
    orchestrator.run_job(job_id, provider=provider, limits=responsive_limits())

    job = db.get_job(job_id)
    assert job["status"] == "cancelled"
    assert provider.calls == [], "a cancelled job touched the provider"
