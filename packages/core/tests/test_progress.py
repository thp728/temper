"""The progress domain: byte sizes, a live measured rate, and per-phase
superseding (issue #49).

Progress is promoted out of the machine's output rather than filtered, and the
things this module owns are the ones a reviewer would grill: the byte units
(docker and huggingface_hub disagree in formatting but agree in meaning), the
rate measured between readings rather than read off a line, and the fact that
one progress line *replaces* the phase's record rather than appending to it.
"""

import pytest

from temper_core.progress import (
    PHASE_IMAGE_PULL,
    PHASE_MODEL_DOWNLOAD,
    ProgressReading,
    ProgressTracker,
    eta_seconds,
    parse_size,
)

# --- byte sizes ------------------------------------------------------------


def test_parse_size_reads_the_units_docker_and_tqdm_write():
    assert parse_size("0B") == 0
    assert parse_size("567") == 567
    assert parse_size("4.096kB") == pytest.approx(4096)
    assert parse_size("15.19MB") == pytest.approx(15.19e6)
    assert parse_size("42.42MB") == pytest.approx(42.42e6)
    assert parse_size("2.147GB") == pytest.approx(2.147e9)
    # huggingface_hub's tqdm unit_scale drops the B: 400M / 4.00G.
    assert parse_size("400M") == pytest.approx(400e6)
    assert parse_size("4.00G") == pytest.approx(4e9)
    # One convention for a byte: SI, the base both tools scale on (1000, not
    # 1024), so a figure rendered here means the same number everywhere.
    assert parse_size("1kB") == pytest.approx(1e3)
    assert parse_size("1MB") == pytest.approx(1e6)
    assert parse_size("1GB") == pytest.approx(1e9)
    assert parse_size("1TB") == pytest.approx(1e12)


def test_parse_size_refuses_what_is_not_a_size():
    assert parse_size("90/200") is None
    assert parse_size("2.00s/it") is None
    assert parse_size("examples/s") is None
    assert parse_size("abc") is None
    assert parse_size("") is None


# --- measured rate and estimate --------------------------------------------


def test_rate_is_measured_between_two_readings():
    rate = (10.0e6 - 5.0e6) / (10.0 - 5.0)
    assert rate == 1.0e6


def test_a_single_reading_has_no_rate():
    assert (
        ProgressTracker()
        .update(
            ProgressReading(
                PHASE_MODEL_DOWNLOAD, done=1.0e6, total=4.0e9, ts=0.0
            )
        )
        .rate
        is None
    )


def test_eta_uses_the_measured_rate_and_corrects_when_it_changes():
    assert eta_seconds(2.0e6, 10.0e6, 2.0e6) == pytest.approx(4.0)
    assert eta_seconds(9.0e6, 10.0e6, 1.0e6) == pytest.approx(1.0)


def test_eta_is_none_without_a_total_or_rate():
    assert eta_seconds(5.0, None, 1.0) is None
    assert eta_seconds(5.0, 10.0, None) is None
    assert eta_seconds(5.0, 10.0, 0.0) is None


def test_eta_is_zero_when_nothing_is_left():
    assert eta_seconds(10.0, 10.0, 1.0) == 0.0


# --- the tracker: supersede per phase ---------------------------------------


def test_a_phase_supersedes_itself_not_the_other_phase():
    tracker = ProgressTracker()
    pull = tracker.update(
        ProgressReading(PHASE_IMAGE_PULL, done=5.0e6, total=10.0e6, ts=0.0)
    )
    download = tracker.update(
        ProgressReading(PHASE_MODEL_DOWNLOAD, done=1.0e6, total=4.0e9, ts=1.0)
    )
    assert pull.done == 5.0e6
    assert download.done == 1.0e6


def test_an_update_replaces_the_phase_not_accumulates():
    tracker = ProgressTracker()
    tracker.update(
        ProgressReading(PHASE_MODEL_DOWNLOAD, done=1.0e6, total=4.0e9, ts=0.0)
    )
    latest = tracker.update(
        ProgressReading(PHASE_MODEL_DOWNLOAD, done=2.0e6, total=4.0e9, ts=1.0)
    )
    assert latest.done == 2.0e6  # one number, not a running total


def test_the_rate_is_measured_between_the_two_latest_readings():
    tracker = ProgressTracker()
    tracker.update(
        ProgressReading(PHASE_MODEL_DOWNLOAD, done=1.0e6, total=4.0e9, ts=0.0)
    )
    latest = tracker.update(
        ProgressReading(PHASE_MODEL_DOWNLOAD, done=3.0e6, total=4.0e9, ts=2.0)
    )
    assert latest.rate == pytest.approx(1.0e6)
    assert latest.eta_s == pytest.approx((4.0e9 - 3.0e6) / 1.0e6)


def test_a_status_line_without_bytes_keeps_the_phases_figures():
    tracker = ProgressTracker()
    tracker.update(
        ProgressReading(PHASE_IMAGE_PULL, done=5.0e6, total=10.0e6, ts=0.0)
    )
    latest = tracker.update(ProgressReading(PHASE_IMAGE_PULL, ts=1.0))
    assert latest.done == 5.0e6
    assert latest.total == 10.0e6


# --- image pull aggregates across layers ------------------------------------
#
# Docker pulls layers in parallel, so no single line is the image's progress:
# each names one layer's done/total. The phase's figures are the aggregate
# across every layer seen so far, which is what "image pull: 57MB/130MB" means.


def test_image_pull_sums_the_layers_it_has_seen():
    tracker = ProgressTracker()
    tracker.update(
        ProgressReading(
            PHASE_IMAGE_PULL,
            done=15.19e6,
            total=42.42e6,
            unit="9b829b73a52f",
            ts=0.0,
        )
    )
    latest = tracker.update(
        ProgressReading(
            PHASE_IMAGE_PULL,
            done=8.5e6,
            total=25.54e6,
            unit="1fe172e4850f",
            ts=1.0,
        )
    )
    assert latest.done == pytest.approx(15.19e6 + 8.5e6)
    assert latest.total == pytest.approx(42.42e6 + 25.54e6)


def test_a_completed_layer_keeps_its_full_size_in_the_total():
    tracker = ProgressTracker()
    tracker.update(
        ProgressReading(
            PHASE_IMAGE_PULL,
            done=42.42e6,
            total=42.42e6,
            unit="9b829b73a52f",
            ts=0.0,
        )
    )
    latest = tracker.update(
        ProgressReading(
            PHASE_IMAGE_PULL,
            done=15.19e6,
            total=42.42e6,
            unit="1fe172e4850f",
            ts=1.0,
        )
    )
    # Layer A is complete at its full size; layer B has only just started. The
    # phase's total carries both layers' sizes, not just the moving ones.
    assert latest.done == pytest.approx(42.42e6 + 15.19e6)
    assert latest.total == pytest.approx(42.42e6 + 42.42e6)


def test_a_layer_that_reports_more_keeps_its_latest_figure():
    """A layer's delivered bytes only grow: `Downloading` rises, then
    `Extracting`, then `Pull complete`, so the max is the honest count."""
    tracker = ProgressTracker()
    tracker.update(
        ProgressReading(
            PHASE_IMAGE_PULL,
            done=15.19e6,
            total=42.42e6,
            unit="9b829b73a52f",
            ts=0.0,
        )
    )
    latest = tracker.update(
        ProgressReading(
            PHASE_IMAGE_PULL,
            done=28.1e6,
            total=42.42e6,
            unit="9b829b73a52f",
            ts=1.0,
        )
    )
    assert latest.done == pytest.approx(28.1e6)


def test_image_pull_rate_is_measured_on_the_aggregate_done():
    tracker = ProgressTracker()
    tracker.update(
        ProgressReading(
            PHASE_IMAGE_PULL,
            done=15.19e6,
            total=42.42e6,
            unit="9b829b73a52f",
            ts=0.0,
        )
    )
    tracker.update(
        ProgressReading(
            PHASE_IMAGE_PULL,
            done=8.5e6,
            total=25.54e6,
            unit="1fe172e4850f",
            ts=1.0,
        )
    )
    latest = tracker.update(
        ProgressReading(
            PHASE_IMAGE_PULL,
            done=28.1e6,
            total=42.42e6,
            unit="9b829b73a52f",
            ts=2.0,
        )
    )
    moved = (28.1e6 + 8.5e6) - (15.19e6 + 8.5e6)
    assert latest.rate == pytest.approx(moved / 1.0)


def test_a_status_line_does_not_wipe_a_layers_figures():
    """`Pull complete` carries no bytes; the layer keeps what it had reached,
    and the phase's aggregate survives the status-only lines that follow."""
    tracker = ProgressTracker()
    tracker.update(
        ProgressReading(
            PHASE_IMAGE_PULL,
            done=35.2e6,
            total=42.42e6,
            unit="9b829b73a52f",
            ts=0.0,
        )
    )
    tracker.update(
        ProgressReading(
            PHASE_IMAGE_PULL,
            done=17.2e6,
            total=25.54e6,
            unit="1fe172e4850f",
            ts=1.0,
        )
    )
    latest = tracker.update(
        ProgressReading(PHASE_IMAGE_PULL, unit="9b829b73a52f", ts=2.0)
    )
    assert latest.done == pytest.approx(35.2e6 + 17.2e6)
    assert latest.total == pytest.approx(42.42e6 + 25.54e6)


def test_model_download_stays_alive_across_files():
    """`snapshot_download` moves from file to file, each with its own bar; a
    finished file keeps its bytes in the phase's figures, so done only grows
    and the measured rate survives the switch instead of resetting (a new
    file's small done would otherwise read as a backward move)."""
    tracker = ProgressTracker()
    tracker.update(
        ProgressReading(
            PHASE_MODEL_DOWNLOAD,
            done=400e6,
            total=4e9,
            unit="model.safetensors",
            ts=0.0,
        )
    )
    tracker.update(
        ProgressReading(
            PHASE_MODEL_DOWNLOAD,
            done=4e9,
            total=4e9,
            unit="model.safetensors",
            ts=40.0,
        )
    )
    latest = tracker.update(
        ProgressReading(
            PHASE_MODEL_DOWNLOAD,
            done=567,
            total=567,
            unit="config.json",
            ts=41.0,
        )
    )
    # The finished shard stays; the phase's done never goes backwards.
    assert latest.done == pytest.approx(4e9 + 567)
    assert latest.total == pytest.approx(4e9 + 567)
    assert latest.rate == pytest.approx(
        567.0
    )  # the config file's live reading, not a frozen shard rate
    assert latest.eta_s == 0.0  # done has reached the seen total
