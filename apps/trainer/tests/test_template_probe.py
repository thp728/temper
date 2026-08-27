"""The export-time template probe (Spec 009 / issue #59).

A fixed probe conversation is tokenised through the template used in training
and through the template serialised into the artifact; the token ids must be
identical. The probe runs on every export -- whether or not anything was
overridden -- because a wrong thinking-mode detection produces a wrong
template with no override involved.

The deliberately mismatched template is the point of this suite. Spec 009's
testing decisions are explicit: *a probe that has only ever passed has no
evidence of working*, and this one is load-bearing twice over. The mismatch
tests below construct a serialised template that is NOT the training template
and assert the probe fails, with the failure naming what differs rather than
only that something did.
"""

from __future__ import annotations

import json

import template_probe

# The deliberately mismatched template used throughout: a serialised template
# that is NOT the training template. Defined once so every test that needs a
# mismatch uses the same one, and the failure message can be asserted against
# a known string.
MISMATCHED_TEMPLATE = "a deliberately different template string"


class FakeTokenizer:
    """A tokenizer double for the probe's tokeniser seam.

    Renders a conversation through a template string and 'tokenises' by
    mapping each rendered character to its codepoint. Deterministic and
    network-free; the property that matters is that ids are a pure function of
    the rendered text, exactly as with a real tokenizer, so two templates that
    render differently produce different ids.
    """

    def __init__(self, template: str = "{{ messages }}"):
        self.chat_template = template

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize: bool,
        chat_template: str | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        tpl = self.chat_template if chat_template is None else chat_template
        kwargs = chat_template_kwargs or {}
        mode = "think" if kwargs.get("enable_thinking") else "nothink"
        text = "\n".join(
            [tpl, mode, *[f"{m['role']}:{m['content']}" for m in conversation]]
        )
        return text if not tokenize else [ord(c) for c in text]


# --- the probe at its own seam ----------------------------------------------


def _mismatched_outcome(tokenizer):
    """A real probe outcome from a deliberately mismatched serialised
    template: training used the model's own template, the artifact records a
    different one. This is the criterion's evidence -- the probe catches it."""
    outcome = template_probe.run_probe(
        tokenizer,
        training_template=tokenizer.chat_template,
        training_kwargs={"enable_thinking": True},
        serialised_template=MISMATCHED_TEMPLATE,
        serialised_kwargs={"enable_thinking": False},
    )
    assert outcome.ok is False
    return outcome


def test_a_probe_whose_templates_match_passes():
    """The happy path is not evidence, but it is the contract: when the
    training template and the serialised template render identically, the
    export may proceed and the probe records an ok result."""
    tokenizer = FakeTokenizer()
    outcome = template_probe.run_probe(
        tokenizer,
        training_template=tokenizer.chat_template,
        training_kwargs={"enable_thinking": True},
        serialised_template=tokenizer.chat_template,
        serialised_kwargs={"enable_thinking": True},
    )
    assert outcome.ok is True
    assert outcome.error_code is None
    assert outcome.message is None
    assert outcome.training_id_count == outcome.serialised_id_count


def test_a_deliberately_mismatched_serialised_template_fails():
    """THE criterion: a probe that has only ever passed has no evidence of
    working. A deliberately different serialised template must fail the probe
    with the stable code -- not pass because the code path is believed sound."""
    tokenizer = FakeTokenizer()
    outcome = _mismatched_outcome(tokenizer)
    assert outcome.ok is False
    assert outcome.error_code == template_probe.PROBE_MISMATCH_CODE
    assert outcome.serialised_id_count != outcome.training_id_count


def test_the_failure_message_names_what_differs():
    """A failure that only says 'the templates differ' does not tell the user
    (or a reviewer) what to fix. The message must name both sides."""
    tokenizer = FakeTokenizer()
    outcome = _mismatched_outcome(tokenizer)
    assert outcome.message is not None
    # the message names the training side and the serialised side...
    assert tokenizer.chat_template in outcome.message
    assert MISMATCHED_TEMPLATE in outcome.message
    # ...and says where the tokenisation first disagrees
    assert outcome.first_id_index is not None
    assert "id" in outcome.message


def test_an_override_template_can_be_the_serialised_template():
    """#80 will let a user override the template; the probe's serialised side
    must be able to carry an explicit template rather than only the
    tokenizer's default. Here an explicit template matches on both sides, so
    the probe passes."""
    tokenizer = FakeTokenizer()
    explicit = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}{% endfor %}"
    outcome = template_probe.run_probe(
        tokenizer,
        training_template=explicit,
        training_kwargs={"enable_thinking": True},
        serialised_template=explicit,
        serialised_kwargs={"enable_thinking": True},
    )
    assert outcome.ok is True


def test_resolve_template_turns_tokenizer_default_into_the_concrete_template():
    """`tokenizer_default` is Axolotl's directive for 'the model's own
    template'; the artifact records the concrete Jinja string, so a downloader
    never has to interpret the directive."""
    tokenizer = FakeTokenizer()
    tpl, kwargs = template_probe.resolve_template(
        tokenizer,
        chat_template="tokenizer_default",
        kwargs={"enable_thinking": True},
    )
    assert tpl == tokenizer.chat_template
    assert kwargs == {"enable_thinking": True}


def test_resolve_template_passes_an_explicit_template_through():
    tokenizer = FakeTokenizer()
    explicit = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}{% endfor %}"
    tpl, kwargs = template_probe.resolve_template(
        tokenizer, chat_template=explicit, kwargs={}
    )
    assert tpl == explicit


def test_outcome_serialises_for_the_artifact_record():
    """The probe result is recorded with the artifact (result.json): the
    record must carry the verdict and, on failure, the specifics."""
    tokenizer = FakeTokenizer()
    outcome = _mismatched_outcome(tokenizer)
    record = outcome.as_dict()
    assert record["ok"] is False
    assert record["error_code"] == template_probe.PROBE_MISMATCH_CODE
    assert record["serialised_template"] == MISMATCHED_TEMPLATE
    assert record["serialised_kwargs"] == {"enable_thinking": False}
    assert "message" in record
    # round-trips through JSON, since it lands in result.json
    json.dumps(record)


# --- the export path (entrypoint wiring) -------------------------------------


def _job_dir(tmp_path, *, recorded_template=None):
    """A job spec + dataset, shaped like what the control plane writes.

    `recorded_template` is written into the spec's hyperparameters as the
    recorded `chat_template` -- the seam #80's override surface writes to. The
    trainer's config ignores it (template resolution is not exposed to
    training yet), so the artifact's serialised template diverges from what
    training applied, which is exactly the divergence the probe must catch.
    """
    from temper_core import hyperparams

    job_dir = tmp_path / "job"
    out_dir = tmp_path / "out"
    job_dir.mkdir()
    hp = hyperparams.effective({})
    if recorded_template is not None:
        hp["chat_template"] = recorded_template
    job = {
        "job_id": "probe-test",
        "base_model": "Qwen/Qwen3-4B",
        "base_revision": "cafe",
        "hyperparameters": hp,
    }
    (job_dir / "job.json").write_text(json.dumps(job))
    rows = []
    for i in range(12):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"question {i}"},
                    {"role": "assistant", "content": f"answer {i}"},
                ]
            }
        )
    (job_dir / "dataset.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    return job_dir, out_dir


def _drive_export(tmp_path, monkeypatch, *, tokenizer, recorded_template=None):
    """Run the trainer's export path; return (exit_code, result_dict).

    The heavy training steps are stubbed; the real probe path runs through the
    fake tokenizer. `recorded_template` makes the artifact's serialised side
    genuinely diverge from training, so the whole cfg -> resolve -> run_probe
    -> compare path fails for real -- no stubbed outcome."""
    import entrypoint as ep

    job_dir, out_dir = _job_dir(tmp_path, recorded_template=recorded_template)
    monkeypatch.setattr(ep, "JOB_DIR", job_dir)
    monkeypatch.setattr(ep, "OUT_DIR", out_dir)
    # CONFIG / RESULT / LOG are resolved at import time; main() reads the
    # module globals, so a patched OUT_DIR alone still points them at /out.
    monkeypatch.setattr(ep, "CONFIG", out_dir / "config.yaml")
    monkeypatch.setattr(ep, "RESULT", out_dir / "result.json")
    monkeypatch.setattr(ep, "LOG", out_dir / "train.log")

    monkeypatch.setattr(ep, "load_probe_tokenizer", lambda job: tokenizer)
    monkeypatch.setattr(ep, "run_streaming", lambda cmd: (0, []))
    monkeypatch.setattr(
        ep,
        "collect_artifacts",
        lambda: {
            "adapter_path": "run/adapter_model.safetensors",
            "adapter_sha256": "x" * 64,
        },
    )
    exit_code = ep.main()
    result = json.loads((out_dir / "result.json").read_text())
    return exit_code, result


def test_every_export_runs_the_probe_and_records_an_ok_result(
    tmp_path, monkeypatch
):
    """The probe runs on every export, whether or not anything was overridden;
    its result is recorded with the artifact. Here nothing was recorded in the
    job spec -- the default case -- so the export proceeds and records a
    pass."""
    tokenizer = FakeTokenizer()
    exit_code, result = _drive_export(
        tmp_path, monkeypatch, tokenizer=tokenizer
    )
    assert "template_probe" in result
    assert result["template_probe"]["ok"] is True
    assert result["template_probe"]["serialised_template"] == (
        tokenizer.chat_template
    )
    assert exit_code == 0
    assert result["ok"] is True


def test_a_failing_probe_fails_the_export_with_the_stable_code(
    tmp_path, monkeypatch
):
    """A deliberately mismatched serialised template must fail the export: the
    result records the stable code, the probe's failure names the difference,
    and the job does not report success. The mismatch is real -- a template
    recorded in the job spec that training never applied -- and the whole
    probe path runs."""
    tokenizer = FakeTokenizer()
    exit_code, result = _drive_export(
        tmp_path,
        monkeypatch,
        tokenizer=tokenizer,
        # gitleaks:allow -- not a secret. The default generic-api-key rule
        # matches "key" inside "monkeypatch" above, then takes the next
        # identifier after an "=" within 40 characters as the value. Here that
        # is MISMATCHED_TEMPLATE, a constant defined at the top of this file.
        recorded_template=MISMATCHED_TEMPLATE,
    )
    assert result["error_code"] == template_probe.PROBE_MISMATCH_CODE
    assert result["template_probe"]["ok"] is False
    assert result["ok"] is False
    assert exit_code != 0
    message = result["template_probe"]["message"]
    assert MISMATCHED_TEMPLATE in message
    assert tokenizer.chat_template in message
