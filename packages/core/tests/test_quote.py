"""The quote -- spec 005, issue #72.

The predictor's warning half: what a job is predicted to cost and how long it
should take, per phase, as ranges. A good test here asserts on what the quote
returns -- that durations are ranges, that cost is composed per phase, that
the download phase uses the measured rate -- never on the internal breakdown
that would break every time the model is refined.

Three properties are load-bearing and each is asserted directly:

* **A duration is a range, never a point.** The throughput figures behind it
  are the softest numbers in the model, so every phase (and the total) carries
  a low and a high, and the low is never the high.
* **Cost is composed per phase, not as one blended rate.** The phases have
  different durations and the first four are largely independent of the
  dataset; one blended rate would hide exactly the cold-start cost a large
  model's download represents.
* **The download phase uses the measured rate.** Spike 5 measured 364 MB/s
  mean with a 0.59 coefficient of variation; a bigger model downloads longer,
  and that is visible in the quote rather than buried.
"""

from __future__ import annotations

import pytest

from temper_core import disk, feasibility, quote
from temper_core.models import ModelFacts

# Qwen3-4B, matching test_memory.py / test_selection.py / test_disk.py.
QWEN3_4B = ModelFacts(
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

# Qwen3-8B, matching test_memory.py.
QWEN3_8B = ModelFacts(
    architecture="qwen3",
    hidden_size=4096,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=12288,
    vocab_size=151936,
    tie_word_embeddings=False,
    is_moe=False,
    has_chat_template=True,
    pad_eos_distinct=True,
    context_length=40960,
    license="apache-2.0",
)

# The L4 rate spike 5 measured; price and currency are inputs to the quote
# because they come from the provider seam live, not assumptions here.
L4_INR_PER_HOUR = 41.31
STORAGE_USD_PER_HOUR = 100 * 0.10 / 730  # 100 GB floored disk, documented rate

NOW = 1_752_000_000.0
TTL_S = 24 * 60 * 60


def estimate(
    facts=QWEN3_4B,
    *,
    rows=100,
    token_count=None,
    price=L4_INR_PER_HOUR,
    currency="INR",
    storage=STORAGE_USD_PER_HOUR,
    hyperparameters=None,
    dataset_created_at=NOW,
    base_revision="1cfa9a7208912126459214e8b04321603b3df60c",
    **overrides,
) -> quote.Quote:
    kwargs = dict(
        facts=facts,
        usable_rows=rows,
        token_count=token_count,
        hyperparameters=hyperparameters or {},
        price_per_hour=price,
        currency=currency,
        storage_cost_usd_per_hour=storage,
        dataset_id="ds_abc",
        dataset_created_at=dataset_created_at,
        base_revision=base_revision,
        expires_at=NOW + TTL_S,
    )
    kwargs.update(overrides)
    return quote.estimate(**kwargs)


# --- a duration is a range, never a point ------------------------------------


def test_every_phase_carries_a_range_never_a_point():
    q = estimate()
    assert len(q.phases) == len(quote.PHASES) == 6
    for phase in q.phases:
        assert phase.duration_low_s is not None
        assert phase.duration_high_s is not None
        assert phase.duration_low_s < phase.duration_high_s
        assert phase.cost_low_minor is not None
        assert phase.cost_high_minor is not None
        assert phase.cost_low_minor < phase.cost_high_minor


def test_the_total_duration_is_a_range_too():
    q = estimate()
    assert q.duration_low_s < q.duration_high_s
    assert q.duration_low_s == pytest.approx(
        sum(p.duration_low_s or 0 for p in q.phases)
    )
    assert q.duration_high_s == pytest.approx(
        sum(p.duration_high_s or 0 for p in q.phases)
    )


def test_phases_appear_in_the_order_a_job_passes_through_them():
    q = estimate()
    assert [p.name for p in q.phases] == list(quote.PHASES)


def test_training_duration_spans_a_full_order_of_magnitude():
    """The training throughput is the softest number in the model; the quote
    widens its range by sqrt(10) each way per the spec's risk note, which
    asserts as a ~10x low-to-high ratio."""
    q = estimate(rows=100)
    training = q.phases[quote.PHASES.index("training")]
    assert training.duration_high_s / training.duration_low_s == pytest.approx(
        10.0, rel=0.01
    )


def test_max_steps_means_training_is_not_estimable_not_guessed():
    """With max_steps set, run length is bounded by steps rather than data
    volume, and no per-step throughput was ever measured -- so the training
    phase reports None rather than a number invented for the occasion."""
    q = estimate(hyperparameters={"max_steps": 10})
    training = q.phases[quote.PHASES.index("training")]
    assert training.duration_low_s is None
    assert training.duration_high_s is None
    assert training.cost_low_minor is None


# --- cost is composed per phase ----------------------------------------------


def test_total_cost_is_the_sum_of_the_phase_costs():
    q = estimate()
    assert q.cost_low_minor == sum(p.cost_low_minor or 0 for p in q.phases)
    assert q.cost_high_minor == sum(p.cost_high_minor or 0 for p in q.phases)


def test_a_higher_hourly_rate_never_predicts_less_cost():
    cheap = estimate(price=41.31)
    dear = estimate(price=250.0)
    assert dear.cost_low_minor > cheap.cost_low_minor
    assert dear.cost_high_minor > cheap.cost_high_minor


def test_currency_and_minor_unit_travel_with_the_cost():
    q = estimate(currency="INR")
    assert q.currency == "INR"
    assert q.minor_unit == 100  # paisa
    # Cost is an integer in the smallest unit: never a float, never a
    # formatting choice applied after the fact.
    assert isinstance(q.cost_low_minor, int)
    assert isinstance(q.cost_high_minor, int)


def test_an_unknown_currency_is_refused_not_assumed():
    with pytest.raises(ValueError, match="cannot quote"):
        estimate(currency="XYZ")


# --- the download phase uses the measured rate --------------------------------


def test_a_bigger_model_downloads_longer():
    small = estimate(QWEN3_4B)
    big = estimate(QWEN3_8B)
    small_dl = small.phases[quote.PHASES.index("model_download")]
    big_dl = big.phases[quote.PHASES.index("model_download")]
    assert big_dl.duration_low_s > small_dl.duration_low_s
    assert big_dl.duration_high_s > small_dl.duration_high_s


def test_download_payload_is_the_download_precision_weights():
    """The phase that dominates a large job is priced at what crosses the
    wire -- bf16 weights (disk.WEIGHTS_DOWNLOAD_BYTES_PER_PARAM), the same
    figure disk.py sums -- not at some smaller training-precision number."""
    q = estimate(QWEN3_4B)
    dl = q.phases[quote.PHASES.index("model_download")]
    size_mb = QWEN3_4B.params * disk.WEIGHTS_DOWNLOAD_BYTES_PER_PARAM / 1e6
    low_rate = quote.DOWNLOAD_RATE_MBPS_MEAN * (1 - quote.DOWNLOAD_RATE_COV)
    assert dl.duration_high_s == pytest.approx(size_mb / low_rate, rel=1e-6)


# --- storage is a separate USD line -------------------------------------------


def test_storage_is_exposed_labelled_never_converted():
    """ADR-0030: no live per-account storage price exists to convert the
    documented USD rate against, so the quote carries it in USD, labelled --
    not silently folded into the account currency."""
    q = estimate()
    assert q.storage_cost_usd_per_hour > 0
    # A long-enough job accrues a visible (nonzero) storage total; the point
    # is that the total is derived from the USD rate times duration, in USD,
    # never converted into the account's currency.
    long_run = estimate(rows=10_000_000)
    assert long_run.storage_cost_usd_total_high_minor > 0


# --- references and expiry -----------------------------------------------------


def test_the_quote_pins_what_it_was_computed_against():
    q = estimate(dataset_created_at=123.0, base_revision="deadbeef" * 5)
    assert q.dataset_id == "ds_abc"
    assert q.dataset_created_at == 123.0
    assert q.base_revision == "deadbeef" * 5


def test_the_quote_expires_and_is_labelled_an_estimate():
    q = estimate()
    assert q.expires_at == NOW + TTL_S
    assert q.is_estimate is True


def test_token_count_is_carried_through():
    assert estimate(token_count=1_000_000).token_count == 1_000_000
    assert estimate().token_count is None


# --- pure, no I/O -------------------------------------------------------------


def test_the_quote_is_pure_arithmetic_no_framework_imports():
    """The quote must survive the Phase B migration unchanged (ADR-0010), so
    it must not import the framework or reach for a clock. The estimate
    function itself is what is under test: it takes its numbers, including
    the wall clock, as arguments."""
    import sys

    src = sys.modules[quote.__name__].__file__ or ""
    raw = open(src, encoding="utf-8").read()
    assert "fastapi" not in raw
    assert "import time" not in raw


# --- training duration reuse --------------------------------------------------


def test_training_duration_can_be_supplied_by_the_caller():
    """The orchestrator already estimates training duration via feasibility;
    the quote accepts that estimate and widens it, rather than re-deriving it
    and risking the two disagreeing."""
    supplied = feasibility.estimated_duration_s(100, {})
    q = estimate(training_duration_s=supplied)
    training = q.phases[quote.PHASES.index("training")]
    assert training.duration_low_s == pytest.approx(
        supplied * quote.TRAINING_DURATION_LOW_FACTOR
    )
