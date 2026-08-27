"""Tests for the browser pages: upload a dataset, read its validation report.

The spec's testing rule: assert that a user-visible fact reaches the page --
a rejected line's number, an error's stable code, the proceed action for a
usable dataset. Never markup structure, class names or layout.

The pages are server-rendered and must work without JavaScript, so "renders
with scripting unavailable" is asserted directly: no <script> tag may appear.
"""

import hashlib
import json
import time
from dataclasses import replace

from helpers import wait_validated

from temper_control_plane.fake_provider import FakeProvider, simulated_limits
from temper_control_plane.limits import RunLimits


def jsonl(tmp_path, rows, name="d.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def chat(user, assistant):
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def form_upload(client, path):
    with open(path, "rb") as f:
        return client.post(
            "/upload", files={"file": (path.name, f)}, follow_redirects=False
        )


def report_page(client, r):
    """Follow an upload's redirect and return the page once validation has
    finished -- validation runs in the background, so the report page shows
    progress and only renders the report when it lands."""
    url = r.headers["location"]
    wait_validated(client, url.rsplit("/", 1)[-1])
    return client.get(url).text


# --- the upload page --------------------------------------------------------


def test_upload_page_offers_a_labelled_file_form(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "<form" in body and 'type="file"' in body
    assert "<label" in body


# --- the report page --------------------------------------------------------


def test_form_upload_redirects_to_the_report(client, tmp_path):
    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(12)])
    r = form_upload(client, p)
    assert r.status_code == 303, "a form post must redirect, not return JSON"
    ds_id = r.headers["location"].rsplit("/", 1)[-1]
    wait_validated(client, ds_id)  # the report page needs the finished report
    report = client.get(r.headers["location"])
    assert report.status_code == 200


def test_report_shows_counts_thinking_and_preview(client, tmp_path):
    rows = [chat(f"q{i}", f"a{i}") for i in range(12)]
    body = report_page(client, form_upload(client, jsonl(tmp_path, rows)))
    assert "12" in body  # rows found / usable
    assert "not detected" in body.lower()  # thinking mode, in plain language
    assert "q0" in body and "a0" in body  # preview of rows as understood


def test_report_names_lines_and_codes_for_a_rejected_dataset(client, tmp_path):
    good = "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(11))
    p = tmp_path / "bad.jsonl"
    p.write_text(
        good
        + "\n{not json}\n"
        + "\n".join(json.dumps(chat("q", "   ")) for _ in range(1))
        + "\n",
        encoding="utf-8",
    )
    body = report_page(client, form_upload(client, p))
    assert "line 12" in body  # the offending line, named
    assert "invalid_json" in body  # its stable code
    assert "empty_target" in body


def test_mixed_thinking_block_is_explained(client, tmp_path):
    marker = f"{chr(60)}think{chr(62)}r{chr(60)}/think{chr(62)}"
    rows = [chat(f"q{i}", f"{marker}a{i}") for i in range(6)]
    rows += [chat(f"q{i}", f"a{i}") for i in range(6)]
    body = report_page(client, form_upload(client, jsonl(tmp_path, rows)))
    assert "mixed_thinking" in body
    assert "blocked" in body.lower()


def test_usable_dataset_with_warnings_still_offers_proceed(client, tmp_path):
    rows = [chat(f"q{i}", f"a{i}") for i in range(12)]  # below 50: warned
    body = report_page(client, form_upload(client, jsonl(tmp_path, rows)))
    assert "few_rows" in body  # the warning, with its code
    assert "/jobs/new" in body  # ...and the journey continues


def test_report_for_an_unknown_dataset_404s(client):
    assert client.get("/datasets/ds_nope").status_code == 404


def test_validating_dataset_renders_progress_not_a_report(client):
    """While validation runs the page shows a proportion complete and reloads
    itself -- a large upload must not look like a frozen page."""
    from temper_control_plane import db, storage

    ds_id = "ds_progress"
    db.create_dataset("big.jsonl", storage.dataset_key(ds_id), ds_id=ds_id)
    db.set_dataset_progress(
        ds_id, {"bytes_read": 50, "bytes_total": 100, "rows": 7}
    )
    body = client.get(f"/datasets/{ds_id}").text
    assert "Validating" in body
    assert "50" in body  # the percent
    assert 'http-equiv="refresh"' in body  # re-renders itself until done


def test_too_large_upload_renders_an_error_page_with_the_code(
    client, tmp_path, monkeypatch
):
    from temper_control_plane import config

    monkeypatch.setattr(config, "MAX_DATASET_BYTES", 10)
    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(12)])
    r = form_upload(client, p)
    assert r.status_code == 413
    assert "text/html" in r.headers["content-type"]
    assert "dataset_too_large" in r.text


def test_form_upload_without_a_filename_renders_the_refusal(client):
    """The form twin of the API's filename refusal: a part with no name is
    answered with the coded refusal rendered as a page, never a crash."""
    import io

    from starlette.requests import Request

    from temper_control_plane import web

    class Unnamed:
        filename = None
        file = io.BytesIO(b"{}\n")

    resp = web.upload_form(Request({"type": "http", "headers": []}), Unnamed())
    assert resp.status_code == 400
    assert resp.context["code"] == "unsupported_extension"


# --- no-JavaScript baseline -------------------------------------------------


def test_no_page_uses_scripting(client, tmp_path):
    p = jsonl(
        tmp_path, [chat("q", "a") for _ in range(3)]
    )  # invalid: errors shown
    r = form_upload(client, p)
    paths = ["/", r.headers["location"], "/datasets/ds_nope"]
    for path in paths:
        body = client.get(path).text.lower()
        assert "<script" not in body, f"{path} must render without JavaScript"


# --- the create-job page ----------------------------------------------------
# Issue #12. A user with a valid dataset must be able to choose a model and
# launch from the browser, and must see everything they are committing to --
# models and licences, the frozen hyperparameters, any feasibility warning --
# before the single action that starts them spending.


def valid_dataset(client, tmp_path):
    rows = [chat(f"q{i}", f"a{i}") for i in range(12)]
    r = form_upload(client, jsonl(tmp_path, rows))
    ds_id = r.headers["location"].rsplit("/", 1)[-1]
    wait_validated(client, ds_id)
    return ds_id


def create_page(client, ds_id):
    return client.get(f"/jobs/new?dataset_id={ds_id}")


def test_create_page_lists_models_with_licence_and_revision(client, tmp_path):
    from temper_core.catalog import is_pinned_revision, listing

    ds = valid_dataset(client, tmp_path)
    body = create_page(client, ds).text
    assert "Qwen/Qwen3-4B" in body
    assert "Qwen/Qwen3-8B" in body
    assert "Apache-2.0" in body  # licence, per model
    # The pinned revision, rendered as the code element the template wraps it in.
    assert "<code>main</code>" not in body
    for m in listing():
        assert is_pinned_revision(m["revision"])
        assert m["revision"] in body


def test_create_page_shows_the_frozen_hyperparameters(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    body = create_page(client, ds).text
    assert "lora_r" in body and "16" in body
    assert "lora_alpha" in body and "32" in body
    assert "num_epochs" in body
    assert "learning_rate" in body


def test_create_page_says_the_spec_freezes_at_launch(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    body = create_page(client, ds).text.lower()
    assert "frozen" in body
    assert "cannot" in body  # ...and cannot be changed after


def test_create_page_has_a_single_launch_action(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    body = create_page(client, ds).text.lower()
    assert body.count("<button") == 1, "launching must be unambiguous"
    assert "launch" in body


def test_feasibility_warning_appears_before_launch(
    client, tmp_path, monkeypatch
):
    from temper_control_plane import config

    monkeypatch.setattr(config, "MAX_JOB_DURATION_S", 10)
    ds = valid_dataset(client, tmp_path)
    body = create_page(client, ds).text
    assert "duration_feasibility" in body  # before launch, while actionable
    assert "<form" in body  # ...and launching is still offered


def test_form_launch_creates_the_job_and_redirects_to_it(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/jobs/new",
        data={"dataset_id": ds, "base_model": "qwen3-8b"},
        follow_redirects=False,
    )
    assert r.status_code == 303, "a form post must redirect, not return JSON"
    assert "/jobs/" in r.headers["location"]


def test_create_page_for_an_invalid_dataset_refuses(client, tmp_path):
    p = jsonl(tmp_path, [chat("q", "a")])  # too few rows: invalid
    r = form_upload(client, p)
    ds = r.headers["location"].rsplit("/", 1)[-1]
    resp = create_page(client, ds)
    assert resp.status_code == 400
    assert "dataset_invalid" in resp.text


def test_create_page_for_an_unknown_dataset_404s(client):
    assert create_page(client, "ds_nope").status_code == 404


def test_create_page_renders_without_javascript(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    body = create_page(client, ds).text.lower()
    assert "<script" not in body


# --- the watch page ----------------------------------------------------------
# Issue #13. The watch page and the finished-job page are the same page: a
# job's record is the same thing during and after the run. What is asserted is
# what a user can see -- state, spend, loss, outcome, teardown proof -- never
# markup structure.


WEIGHTS = b"weights"
RESULT = {
    "ok": True,
    "stage": "train",
    "adapter_path": "run/adapter_model.safetensors",
    "adapter_sha256": hashlib.sha256(WEIGHTS).hexdigest(),
    "adapter_config": {"r": 16, "lora_alpha": 32},
}
OUTPUT_LINES = [
    "[10:00:01] building trainer image",
    "[10:03:04] image built in 183s",
    "[10:03:04] running training",
    "{'loss': 1.9042, 'step': 10, 'epoch': 0.5}",
]


# The guard's poll interval is a second in production; small here so that a
# cancellation lands within the run rather than after it.
def responsive_limits():
    return replace(RunLimits.from_config(), poll_interval_s=0.005)


def launch_from_form(client, tmp_path) -> str:
    """The one launch block all three helpers share; they differ only in what
    `orchestrator.launch` has been patched to do."""
    ds = valid_dataset(client, tmp_path)
    r = client.post(
        "/jobs/new",
        data={"dataset_id": ds, "base_model": "qwen3-4b"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    return r.headers["location"]


def launch_running_job(
    client, monkeypatch, tmp_path, provider, limits=None
) -> str:
    """Launch from the browser form and let the job run on its thread."""
    import threading

    from temper_control_plane import fake_models, orchestrator

    def start(job_id):
        threading.Thread(
            target=orchestrator.run_job,
            args=(job_id,),
            kwargs={
                "provider": provider,
                "limits": limits,
                "models": fake_models.catalog_models(),
            },
            daemon=True,
            name=f"job-{job_id[:8]}",
        ).start()

    monkeypatch.setattr(orchestrator, "launch", start)
    return launch_from_form(client, tmp_path)


def launch_unstarted_job(client, monkeypatch, tmp_path) -> str:
    """A job created but not yet picked up: `queued`, as one is between the
    form post and the thread."""
    from temper_control_plane import orchestrator

    monkeypatch.setattr(orchestrator, "launch", lambda job_id: None)
    return launch_from_form(client, tmp_path)


def run_finished_job(
    client, monkeypatch, tmp_path, provider, limits=None
) -> str:
    """Launch and drive the job to a terminal state before returning."""
    from temper_control_plane import fake_models, orchestrator

    monkeypatch.setattr(
        orchestrator,
        "launch",
        lambda job_id: orchestrator.run_job(
            job_id,
            provider=provider,
            limits=limits,
            models=fake_models.catalog_models(),
        ),
    )
    path = launch_from_form(client, tmp_path)
    wait_until_terminal(client, path)
    return path


def watch(client, path):
    return client.get(path)


def wait_until_terminal(client, path):
    """Block until the job behind a watch path ends.

    A test that leaves its job thread running leaks it into the next test's
    database, so every test that starts a job also waits it out.
    """
    deadline = time.time() + 20
    while time.time() < deadline:
        if client.get(f"/v1/jobs/{path.rsplit('/', 1)[-1]}").json()[
            "status"
        ] in ("complete", "failed", "cancelled"):
            return
        time.sleep(0.01)
    raise AssertionError("job never reached a terminal state")


def test_watch_page_shows_state_elapsed_machine_and_price(
    client, monkeypatch, tmp_path
):
    provider = FakeProvider(
        lines=OUTPUT_LINES, result=RESULT, adapter_bytes=WEIGHTS
    )
    path = run_finished_job(client, monkeypatch, tmp_path, provider)
    body = watch(client, path).text
    assert "complete" in body.lower()
    assert "elapsed" in body.lower()
    assert "L4" in body  # machine type
    assert (
        "41.31" in body and "INR" in body
    )  # hourly price, in account currency


def test_watch_page_shows_the_latest_loss_with_its_step(
    client, monkeypatch, tmp_path
):
    provider = FakeProvider(
        lines=OUTPUT_LINES, result=RESULT, adapter_bytes=WEIGHTS
    )
    path = run_finished_job(client, monkeypatch, tmp_path, provider)
    body = watch(client, path).text
    assert "1.9042" in body  # the number itself
    assert "10" in body  # ...at its step
    assert "<svg" not in body.lower() and "<canvas" not in body.lower(), (
        "a chart is Phase B; the number is what Phase A shows"
    )


def test_watch_page_for_a_queued_job_offers_cancel_and_warns_first(
    client, monkeypatch, tmp_path
):
    path = launch_unstarted_job(client, monkeypatch, tmp_path)
    body = watch(client, path).text
    assert 'action="' + path + '/cancel"' in body
    assert "no adapter" in body.lower(), (
        "the consequence must be stated before cancelling happens"
    )


def test_watch_page_offers_cancel_in_every_working_state(
    client, monkeypatch, tmp_path
):
    # provisioning (before the machine exists), preparing (waiting for SSH),
    # training (mid-stream): one pause point per working state.
    providers = [
        FakeProvider(pause_at_stage="gpu_availability"),
        FakeProvider(
            lines=OUTPUT_LINES, result=RESULT, pause_at_stage="await_ready"
        ),
        FakeProvider(
            lines=OUTPUT_LINES,
            result=RESULT,
            adapter_bytes=WEIGHTS,
            pause_at_line=2,
        ),
    ]
    for i, provider in enumerate(providers):
        path = launch_running_job(
            client, monkeypatch, tmp_path, provider, limits=responsive_limits()
        )
        assert provider.wait_until_paused(), "the job never reached its pause"
        body = watch(client, path).text
        assert 'action="' + path + '/cancel"' in body
        assert "no adapter" in body.lower()
        if i == 1:
            # Provisioned but not yet training: spend is visible while it
            # accrues, not only once the run is over.
            assert "L4" in body and "41.31" in body
        provider.resume.set()
        wait_until_terminal(client, path)


def test_terminal_jobs_hide_the_cancel_control(client, monkeypatch, tmp_path):
    cases = [
        FakeProvider(lines=OUTPUT_LINES, result=RESULT, adapter_bytes=WEIGHTS),
        FakeProvider(fail_at="push", fail_code="source_upload_failed"),
    ]
    for provider in cases:
        path = run_finished_job(client, monkeypatch, tmp_path, provider)
        body = watch(client, path).text
        assert 'action="' + path + '/cancel"' not in body


def test_completed_job_offers_the_adapter_with_its_config(
    client, monkeypatch, tmp_path
):
    provider = FakeProvider(
        lines=OUTPUT_LINES, result=RESULT, adapter_bytes=WEIGHTS
    )
    path = run_finished_job(client, monkeypatch, tmp_path, provider)
    job_id = path.rsplit("/", 1)[-1]
    body = watch(client, path).text
    assert f"/v1/jobs/{job_id}/adapter" in body
    assert "adapter_config" in body.lower(), (
        "the download must promise the file that makes it loadable"
    )


def test_failed_job_shows_its_code_and_keeps_its_log(
    client, monkeypatch, tmp_path
):
    provider = FakeProvider(
        lines=OUTPUT_LINES,
        result={"ok": False, "stage": "train", "error": "CUDA out of memory"},
    )
    path = run_finished_job(client, monkeypatch, tmp_path, provider)
    body = watch(client, path).text
    assert "training_failed" in body  # the stable code
    assert "CUDA out of memory" in body  # the reason, in plain language
    for line in OUTPUT_LINES:  # output from before the failure
        # Jinja escapes the quotes in a framework log-dict line; the fact
        # under test is retention of the line, not its entity encoding.
        assert line in body.replace("&#39;", "'")


def test_a_cancelled_job_is_described_distinctly_from_a_failure(
    client, monkeypatch, tmp_path
):
    provider = FakeProvider(lines=OUTPUT_LINES, result=RESULT, pause_at_line=2)
    path = launch_running_job(
        client, monkeypatch, tmp_path, provider, limits=responsive_limits()
    )
    assert provider.wait_until_paused()
    r = client.post(path + "/cancel")
    assert r.status_code == 200  # the form redirects to the page
    provider.resume.set()
    wait_until_terminal(client, path)
    assert (
        client.get(f"/v1/jobs/{path.rsplit('/', 1)[-1]}").json()["status"]
        == "cancelled"
    )
    body = watch(client, path).text
    assert "your request" in body.lower(), (
        "the user's own decision must not read as a defect"
    )
    assert "training_failed" not in body and "gpu_stalled" not in body


def test_a_stalled_job_says_it_was_stalled(client, monkeypatch, tmp_path):
    provider = FakeProvider(lines=OUTPUT_LINES, result=RESULT, silent_after=2)
    path = run_finished_job(
        client,
        monkeypatch,
        tmp_path,
        provider,
        limits=simulated_limits(step=60.0, stall=900.0),
    )
    body = watch(client, path).text
    assert "gpu_stalled" in body
    assert "stall" in body.lower() or "silence" in body.lower()


def test_an_over_long_job_says_it_hit_the_ceiling(
    client, monkeypatch, tmp_path
):
    provider = FakeProvider(
        lines=[f"step {i}" for i in range(200)], result=RESULT
    )
    path = run_finished_job(
        client,
        monkeypatch,
        tmp_path,
        provider,
        limits=simulated_limits(step=300.0, stall=900.0, maximum=3600.0),
    )
    body = watch(client, path).text
    assert "gpu_max_duration_exceeded" in body
    assert (
        "duration" in body.lower()
        or "ceiling" in body.lower()
        or "limit" in body.lower()
    )


def test_the_machine_destroyed_confirmation_is_visible(
    client, monkeypatch, tmp_path
):
    provider = FakeProvider(
        lines=OUTPUT_LINES, result=RESULT, adapter_bytes=WEIGHTS
    )
    path = run_finished_job(client, monkeypatch, tmp_path, provider)
    body = watch(client, path).text
    assert "destroyed" in body.lower(), (
        "proof that billing stopped must reach the page"
    )


def test_the_state_change_is_announced_to_screen_readers(
    client, monkeypatch, tmp_path
):
    path = launch_unstarted_job(client, monkeypatch, tmp_path)
    body = watch(client, path).text
    assert "aria-live" in body


def test_only_live_updates_need_javascript(client, monkeypatch, tmp_path):
    """The core view is server-rendered; the script exists only while there is
    something live to follow, and is gone once the job is terminal."""
    provider = FakeProvider(
        lines=OUTPUT_LINES,
        result=RESULT,
        adapter_bytes=WEIGHTS,
        pause_at_line=2,
    )
    running = launch_running_job(
        client, monkeypatch, tmp_path, provider, limits=responsive_limits()
    )
    assert provider.wait_until_paused()
    assert "<script" in watch(client, running).text.lower()
    provider.resume.set()
    wait_until_terminal(client, running)

    finished = run_finished_job(
        client,
        monkeypatch,
        tmp_path,
        FakeProvider(lines=OUTPUT_LINES, result=RESULT, adapter_bytes=WEIGHTS),
    )
    assert "<script" not in watch(client, finished).text.lower()


def test_watch_page_for_an_unknown_job_404s(client):
    assert client.get("/jobs/job_nosuchthing").status_code == 404


# --- the jobs list ------------------------------------------------------------
# Issue #14. A user who closes the tab must be able to find their adapter
# again: the list shows every job with its outcome and links to its record,
# which is the watch page -- already carrying the full event history and the
# download for a finished job.


def test_jobs_list_shows_each_job_with_its_outcome(
    client, monkeypatch, tmp_path
):
    done = run_finished_job(
        client,
        monkeypatch,
        tmp_path,
        FakeProvider(lines=OUTPUT_LINES, result=RESULT, adapter_bytes=WEIGHTS),
    )
    failed = run_finished_job(
        client,
        monkeypatch,
        tmp_path,
        FakeProvider(fail_at="push", fail_code="source_upload_failed"),
    )
    # A cancelled job too: "cancelled" appears in no error code, so its
    # presence on the page can only come from the outcome column.
    provider = FakeProvider(lines=OUTPUT_LINES, result=RESULT, pause_at_line=2)
    cancelled = launch_running_job(
        client, monkeypatch, tmp_path, provider, limits=responsive_limits()
    )
    assert provider.wait_until_paused()
    client.post(cancelled + "/cancel")
    provider.resume.set()
    wait_until_terminal(client, cancelled)

    body = client.get("/jobs").text
    for path in (done, failed, cancelled):
        assert path.rsplit("/", 1)[-1] in body  # every job appears...
    assert "complete" in body.lower()  # ...with its outcome beside it
    assert "cancelled" in body.lower()
    assert "failed" in body.lower()
    assert "source_upload_failed" in body  # a failure names its code


def test_jobs_list_links_to_each_jobs_record(client, monkeypatch, tmp_path):
    path = run_finished_job(
        client,
        monkeypatch,
        tmp_path,
        FakeProvider(lines=OUTPUT_LINES, result=RESULT, adapter_bytes=WEIGHTS),
    )
    body = client.get("/jobs").text
    assert f'href="{path}"' in body, "an entry must lead to the full record"


def test_jobs_list_is_reachable_from_the_upload_page(client):
    body = client.get("/").text
    assert 'href="/jobs"' in body, (
        "a user who closes the tab needs a way back to their jobs"
    )


def test_empty_jobs_list_says_so_and_offers_the_way_in(client):
    body = client.get("/jobs").text
    assert "no jobs yet" in body.lower()
    assert 'href="/"' in body


def test_a_finished_jobs_record_shows_its_full_event_history(
    client, monkeypatch, tmp_path
):
    path = run_finished_job(
        client,
        monkeypatch,
        tmp_path,
        FakeProvider(lines=OUTPUT_LINES, result=RESULT, adapter_bytes=WEIGHTS),
    )
    body = watch(client, path).text
    # The lifecycle's early states are on the page even though the job ended
    # elsewhere: a finished run is as inspectable as a running one.
    assert "queued" in body.lower()
    assert "provisioning" in body.lower()


def test_jobs_list_renders_without_javascript(client, monkeypatch, tmp_path):
    run_finished_job(
        client,
        monkeypatch,
        tmp_path,
        FakeProvider(lines=OUTPUT_LINES, result=RESULT, adapter_bytes=WEIGHTS),
    )
    body = client.get("/jobs").text.lower()
    assert "<script" not in body
