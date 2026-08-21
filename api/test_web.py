"""Tests for the browser pages: upload a dataset, read its validation report.

The spec's testing rule: assert that a user-visible fact reaches the page --
a rejected line's number, an error's stable code, the proceed action for a
usable dataset. Never markup structure, class names or layout.

The pages are server-rendered and must work without JavaScript, so "renders
with scripting unavailable" is asserted directly: no <script> tag may appear.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def jsonl(tmp_path, rows, name="d.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def chat(user, assistant):
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


def form_upload(client, path):
    with open(path, "rb") as f:
        return client.post("/upload", files={"file": (path.name, f)},
                           follow_redirects=False)


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
    report = client.get(r.headers["location"])
    assert report.status_code == 200


def test_report_shows_counts_thinking_and_preview(client, tmp_path):
    rows = [chat(f"q{i}", f"a{i}") for i in range(12)]
    r = form_upload(client, jsonl(tmp_path, rows))
    body = client.get(r.headers["location"]).text
    assert "12" in body                      # rows found / usable
    assert "not detected" in body.lower()    # thinking mode, in plain language
    assert "q0" in body and "a0" in body     # preview of rows as understood


def test_report_names_lines_and_codes_for_a_rejected_dataset(client, tmp_path):
    good = "\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(11))
    p = tmp_path / "bad.jsonl"
    p.write_text(good + "\n{not json}\n" + "\n".join(
        json.dumps(chat("q", "   ")) for _ in range(1)) + "\n", encoding="utf-8")
    r = form_upload(client, p)
    body = client.get(r.headers["location"]).text
    assert "line 12" in body                 # the offending line, named
    assert "invalid_json" in body            # its stable code
    assert "empty_target" in body


def test_mixed_thinking_block_is_explained(client, tmp_path):
    rows = [chat(f"q{i}", f"<think>r</think>a{i}") for i in range(6)]
    rows += [chat(f"q{i}", f"a{i}") for i in range(6)]
    r = form_upload(client, jsonl(tmp_path, rows))
    body = client.get(r.headers["location"]).text
    assert "mixed_thinking" in body
    assert "blocked" in body.lower()


def test_usable_dataset_with_warnings_still_offers_proceed(client, tmp_path):
    rows = [chat(f"q{i}", f"a{i}") for i in range(12)]  # below 50: warned
    r = form_upload(client, jsonl(tmp_path, rows))
    body = client.get(r.headers["location"]).text
    assert "few_rows" in body                # the warning, with its code
    assert "/jobs/new" in body               # ...and the journey continues


def test_report_for_an_unknown_dataset_404s(client):
    assert client.get("/datasets/ds_nope").status_code == 404


def test_too_large_upload_renders_an_error_page_with_the_code(client, tmp_path, monkeypatch):
    from api import config
    monkeypatch.setattr(config, "MAX_DATASET_BYTES", 10)
    p = jsonl(tmp_path, [chat(f"q{i}", f"a{i}") for i in range(12)])
    r = form_upload(client, p)
    assert r.status_code == 413
    assert "text/html" in r.headers["content-type"]
    assert "dataset_too_large" in r.text


# --- no-JavaScript baseline -------------------------------------------------

def test_no_page_uses_scripting(client, tmp_path):
    p = jsonl(tmp_path, [chat("q", "a") for _ in range(3)])  # invalid: errors shown
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
    return r.headers["location"].rsplit("/", 1)[-1]


def create_page(client, ds_id):
    return client.get(f"/jobs/new?dataset_id={ds_id}")


def test_create_page_lists_models_with_licence_and_revision(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    body = create_page(client, ds).text
    assert "Qwen/Qwen3-4B" in body
    assert "Qwen/Qwen3-8B" in body
    assert "Apache-2.0" in body              # licence, per model
    # The pinned revision, rendered as the code element the template wraps it in.
    assert "<code>main</code>" in body


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
    assert "cannot" in body                  # ...and cannot be changed after


def test_create_page_has_a_single_launch_action(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    body = create_page(client, ds).text.lower()
    assert body.count("<button") == 1, "launching must be unambiguous"
    assert "launch" in body


def test_feasibility_warning_appears_before_launch(client, tmp_path, monkeypatch):
    from api import config
    monkeypatch.setattr(config, "MAX_JOB_DURATION_S", 10)
    ds = valid_dataset(client, tmp_path)
    body = create_page(client, ds).text
    assert "duration_feasibility" in body    # before launch, while actionable
    assert "<form" in body                   # ...and launching is still offered


def test_form_launch_creates_the_job_and_redirects_to_it(client, tmp_path):
    ds = valid_dataset(client, tmp_path)
    r = client.post("/jobs/new", data={"dataset_id": ds,
                                       "base_model": "qwen3-8b"},
                    follow_redirects=False)
    assert r.status_code == 303, "a form post must redirect, not return JSON"
    assert "/jobs/" in r.headers["location"]


def test_create_page_for_an_invalid_dataset_refuses(client, tmp_path):
    p = jsonl(tmp_path, [chat("q", "a")])    # too few rows: invalid
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
