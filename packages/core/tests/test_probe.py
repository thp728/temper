"""The admission probe -- spec 009, issue #58.

A probe is a verdict plus findings; the tests assert on the decision and its
reason, not on the mechanism that produced it. The whole suite runs without a
network: facts arrive as `ModelFacts` fixtures, exactly as they would through
the `models` seam.

Spec 009's testing decisions require a fixture for **every** failure path, not
just the happy path: a clean model; one with no chat template; one where
padding and end-of-sequence collide; one that is mixture-of-experts; one whose
revision does not resolve; one whose licence is unresolvable; one too large for
anything available. Each is represented here by a fixture that drives exactly
one failure.
"""

from __future__ import annotations

from temper_core import probe
from temper_core.models import ModelFacts

# Qwen3-4B, from its own published config.json (matches test_selection.py): a
# clean, dense, chat-templated, distinct-pad model with a resolvable licence.
CLEAN = ModelFacts(
    architecture="qwen3",
    hidden_size=2560,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=9728,
    vocab_size=151936,
    tie_word_embeddings=True,
    is_moe=False,
    has_chat_template=True,
    pad_eos_distinct=True,
    context_length=40960,
    license="apache-2.0",
)

NO_TEMPLATE = ModelFacts(**{**CLEAN.__dict__, "has_chat_template": False})

PAD_EOS_COLLIDE = ModelFacts(**{**CLEAN.__dict__, "pad_eos_distinct": False})

NO_LICENCE = ModelFacts(**{**CLEAN.__dict__, "license": ""})

MOE = ModelFacts(**{**CLEAN.__dict__, "is_moe": True})

UNKNOWN_ARCHITECTURE = ModelFacts(
    **{**CLEAN.__dict__, "architecture": "unknown"}
)

# Big enough that even QLoRA's half-byte weights exceed the largest card this
# platform provisions (H200, 141 GB): weights alone sit past 240 GB for this
# shape, so no provisionable card can hold the predicted peak. The point is
# the verdict and its arithmetic, not the exact capacity that refused it.
TOO_LARGE = ModelFacts(
    architecture="qwen3",
    hidden_size=16384,
    num_hidden_layers=128,
    num_attention_heads=128,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=65536,
    vocab_size=151936,
    tie_word_embeddings=True,
    is_moe=False,
    has_chat_template=True,
    pad_eos_distinct=True,
    context_length=40960,
    license="apache-2.0",
)


def probe_facts(facts, **kwargs):
    return probe.probe(
        facts,
        repo="org/model",
        revision="deadbeef",
        **kwargs,
    )


def codes(result: probe.ProbeResult) -> set[str]:
    return {f.code for f in result.findings}


# --- a clean model ------------------------------------------------------------


def test_a_clean_model_passes_with_no_findings():
    result = probe_facts(CLEAN)
    assert result.ok is True
    assert result.verdict == "usable"
    assert result.findings == ()


def test_the_memory_check_reports_the_arithmetic_for_a_model_that_fits():
    result = probe_facts(CLEAN)
    assert result.memory is not None
    assert result.memory.fits is True
    assert result.memory.card == "L4"
    assert result.memory.capacity_gb == 24.0
    assert result.memory.headroom_gb is not None
    assert result.memory.headroom_gb > 0


# --- a missing chat template blocks ------------------------------------------


def test_a_missing_chat_template_blocks():
    result = probe_facts(NO_TEMPLATE)
    assert result.ok is False
    assert result.verdict == "blocked"
    assert probe.FINDING_MISSING_CHAT_TEMPLATE in codes(result)
    finding = next(
        f
        for f in result.findings
        if f.code == probe.FINDING_MISSING_CHAT_TEMPLATE
    )
    assert finding.severity == "block"
    # The reason is product language, kept visible: the alternative is
    # hand-writing role delimiters.
    assert "role" in finding.message


# --- padding colliding with end-of-sequence blocks ---------------------------


def test_padding_colliding_with_end_of_sequence_blocks():
    result = probe_facts(PAD_EOS_COLLIDE)
    assert result.ok is False
    assert result.verdict == "blocked"
    assert probe.FINDING_PAD_EOS_COLLIDE in codes(result)
    finding = next(
        f for f in result.findings if f.code == probe.FINDING_PAD_EOS_COLLIDE
    )
    assert finding.severity == "block"
    # The reason is product language, kept visible: unmasked it teaches the
    # model never to stop.
    assert "stop" in finding.message


# --- an untested architecture warns, labelled --------------------------------


def test_a_mixture_of_experts_model_warns_and_is_labelled_not_blocked():
    result = probe_facts(MOE)
    assert result.ok is True
    assert result.verdict == "usable_with_warnings"
    assert probe.FINDING_UNTESTED_ARCHITECTURE in codes(result)
    finding = next(
        f
        for f in result.findings
        if f.code == probe.FINDING_UNTESTED_ARCHITECTURE
    )
    assert finding.severity == "warn"
    assert "untested" in finding.message.lower()


def test_an_unknown_architecture_warns_and_is_labelled_not_blocked():
    result = probe_facts(UNKNOWN_ARCHITECTURE)
    assert result.ok is True
    assert result.verdict == "usable_with_warnings"
    assert probe.FINDING_UNTESTED_ARCHITECTURE in codes(result)


# --- a licence that does not resolve warns -----------------------------------


def test_an_unresolvable_licence_warns_and_is_labelled_not_blocked():
    result = probe_facts(NO_LICENCE)
    assert result.ok is True
    assert result.verdict == "usable_with_warnings"
    assert probe.FINDING_LICENSE_UNRESOLVABLE in codes(result)
    finding = next(
        f
        for f in result.findings
        if f.code == probe.FINDING_LICENSE_UNRESOLVABLE
    )
    assert finding.severity == "warn"
    # The posture #48/#59 established: a visibly missing licence beats a wrong
    # one, so this is shown, not guessed.
    assert "unknown" in finding.message.lower()


# --- the predicted memory fits something available ---------------------------


def test_a_model_too_large_for_anything_available_warns_with_the_arithmetic():
    result = probe_facts(TOO_LARGE)
    assert result.ok is True
    assert result.verdict == "usable_with_warnings"
    assert probe.FINDING_MEMORY_DOES_NOT_FIT in codes(result)
    assert result.memory is not None
    assert result.memory.fits is False
    assert result.memory.peak_gb is not None
    assert result.memory.capacity_gb is not None
    # The warning carries the same numbers the creation-time refusal (#54)
    # would refuse with -- the two cannot drift, because this is that
    # computation, not a copy.
    assert result.memory.peak_gb > result.memory.capacity_gb
    finding = next(
        f
        for f in result.findings
        if f.code == probe.FINDING_MEMORY_DOES_NOT_FIT
    )
    assert finding.severity == "warn"
    # Memory warns at admission and blocks at creation (ADR-0043): the message
    # says the launch would be refused, so the user is not told a warning when
    # the creation path would refuse.
    assert "refused" in finding.message


def test_an_unresolvable_revision_blocks():
    result = probe.unresolvable(
        "org/ghost", "deadbeef" * 5, "404 from the model hub"
    )
    assert result.ok is False
    assert result.verdict == "blocked"
    assert probe.FINDING_REVISION_UNRESOLVABLE in codes(result)


# --- the verdict is shown, not only enforced ---------------------------------


def test_the_result_serialises_for_persistence_and_display():
    result = probe_facts(MOE)
    record = result.as_dict()
    assert record["ok"] is True
    assert record["verdict"] == "usable_with_warnings"
    assert record["architecture"] == "qwen3"
    assert record["repo"] == "org/model"
    assert record["revision"] == "deadbeef"
    assert any(
        f["code"] == probe.FINDING_UNTESTED_ARCHITECTURE
        and f["severity"] == "warn"
        for f in record["findings"]
    )
    assert record["memory"]["fits"] is True
