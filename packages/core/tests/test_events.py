"""The line classifier, as a pure function over real training output.

Every string here is the shape the framework actually emits: the dict
transformers' progress callback writes once per logging step, the evaluation
and summary dicts that look almost the same, the tqdm bars it redraws between
them, and the prose Axolotl and the remote script narrate with.

The negative cases matter more than the positive ones. A false loss number is
worse than a missing one — a user reading a chart cannot tell that the figure
came from a line that merely looked like a measurement.
"""

from temper_core.events import classify


def metric(line: str) -> dict:
    event = classify(line)
    assert event.kind == "metric", f"expected a metric event, got {event!r}"
    return event.data


def is_log(line: str) -> bool:
    event = classify(line)
    return event.kind == "log" and event.data is None


# --- what becomes a metric --------------------------------------------------


def test_a_training_log_line_yields_loss_and_epoch():
    line = (
        "{'loss': 1.9042, 'grad_norm': 3.9971, "
        "'learning_rate': 4.5e-05, 'epoch': 0.13}"
    )
    assert metric(line) == {"loss": 1.9042, "epoch": 0.13}


def test_the_original_line_survives_as_the_message():
    """Structuring the numbers must not cost the reader the line they came in."""
    line = "{'loss': 1.9042, 'epoch': 0.13}"
    assert classify(line).message == line


def test_integers_are_read_as_numbers_not_dropped():
    assert metric("{'loss': 2, 'epoch': 1}") == {"loss": 2.0, "epoch": 1.0}


def test_a_negative_or_exponent_loss_is_still_a_number():
    assert metric("{'loss': 1.5e-03, 'epoch': 2.0}")["loss"] == 0.0015


def test_the_step_is_promoted_when_the_framework_states_it():
    """The one number that has to come from a field rather than a bar.

    See the classifier's note: an undescribed progress bar is not evidence of a
    training step, because the evaluation bar is drawn the same way.
    """
    assert metric("{'loss': 1.9, 'epoch': 0.13, 'step': 12}")["step"] == 12
    assert metric("{'loss': 1.9, 'global_step': 12}")["step"] == 12


def test_a_loss_that_is_not_a_number_does_not_suppress_the_rest_of_the_line():
    """A diverged job prints `nan`; the epoch beside it is still real."""
    assert metric("{'loss': nan, 'epoch': 0.13}") == {"epoch": 0.13}


# --- recognised, deliberately not promoted ----------------------------------


def test_grad_norm_and_learning_rate_are_not_promoted():
    """They travel on the same line as the loss and stay in the log text."""
    line = (
        "{'loss': 1.9042, 'grad_norm': 3.9971, "
        "'learning_rate': 4.5e-05, 'epoch': 0.13}"
    )
    assert "grad_norm" not in metric(line)
    assert "learning_rate" not in metric(line)


def test_a_line_carrying_only_diagnostics_is_log_output():
    assert is_log("{'grad_norm': 3.9971, 'learning_rate': 4.5e-05}")


def test_throughput_is_not_promoted():
    assert is_log("[axolotl] throughput 2.00s/it, 0.5 samples/s")


# --- what must never become a metric ----------------------------------------


def test_prose_that_merely_mentions_loss_is_log_output():
    assert is_log("[trainer] initial loss: 3.0 before any step")
    assert is_log("[axolotl] loss masking enabled, train_on_inputs=False")


def test_an_evaluation_row_is_not_a_training_step():
    """`eval_loss` is a different key, and the epoch beside it is not ours.

    The trainer holds back a validation split by default, so these lines are
    guaranteed to appear. Promoting their epoch would interleave a second
    series into the first.
    """
    assert is_log("{'eval_loss': 1.2, 'eval_runtime': 4.0, 'epoch': 1.0}")


def test_the_end_of_training_summary_is_not_a_training_step():
    assert is_log("{'train_runtime': 43.2, 'train_loss': 0.94, 'epoch': 3.0}")


def test_a_progress_bar_is_not_a_training_step():
    """The correction this classifier carries.

    An undescribed bar looks like the training bar and may be the evaluation
    bar, which counts batches and restarts every epoch. Described bars — dataset
    mapping, weight downloads — count something else again. None of them is
    promoted.
    """
    assert is_log(" 33%|███▎      | 10/30 [00:20<00:40,  2.00s/it]")
    assert is_log("  0%|          | 0/30 [00:00<?, ?it/s]")
    assert is_log(
        "Map:  45%|████▌     | 90/200 [00:01<00:01, 88.0 examples/s]"
    )
    assert is_log(
        "model.safetensors:  10%|█         | 400M/4.00G [00:05<00:45]"
    )


def test_a_partial_line_does_not_produce_a_metric():
    """Reads split mid-line; half a measurement is not a measurement."""
    assert is_log("{'loss': 1.90")
    assert is_log(" 33%|███")


def test_a_malformed_payload_produces_no_metric():
    assert is_log("{'loss': }")
    assert is_log("{'loss': None, 'epoch': None}")


def test_a_nested_payload_is_left_as_log_output():
    """The training log dict is flat; mis-parsing a nested one costs more."""
    assert is_log("{'loss': 1.9, 'extra': {'a': 1}}")


def test_ordinary_output_is_log_output():
    assert is_log("[10:03:04] building trainer image")
    assert is_log("#8 [4/6] RUN pip install -r requirements.txt")
    assert is_log("")


# --- what Axolotl actually emits, captured from job_05300098085f ------------
#
# Every line below is verbatim from a real run on an L4. They are here because
# the suite above was written against plain transformers, which logs live
# values; Axolotl formats each value to a fixed precision first and logs the
# *strings*. Nothing in the suite noticed, and a real job produced 24 of these
# lines and not one metric event.

AXOLOTL_STEP = (
    "{'loss': '0.7157', 'grad_norm': '12.79', 'learning_rate': '0', "
    "'ppl': '2.046', 'memory/max_active (GiB)': '2.98', "
    "'memory/max_allocated (GiB)': '2.98', "
    "'memory/device_reserved (GiB)': '4.12', 'tokens/trainable': 93, "
    "'tokens/total': 512, 'epoch': '0.1333'}"
)

AXOLOTL_SUMMARY = (
    "{'train_runtime': '55.71', 'train_samples_per_second': '3.303', "
    "'train_steps_per_second': '0.413', 'train_loss': '0.06318', "
    "'memory/max_active (GiB)': '2.86', 'epoch': '3.0'}"
)


def test_axolotl_writes_its_numbers_as_strings_and_they_still_promote():
    assert metric(AXOLOTL_STEP) == {"loss": 0.7157, "epoch": 0.1333}


def test_a_quoted_number_in_exponent_form_promotes():
    """Loss goes exponential on an easy dataset; this run reached 5.866e-06."""
    assert metric("{'loss': '5.866e-06', 'epoch': '2.933'}") == {
        "loss": 5.866e-06,
        "epoch": 2.933,
    }


def test_the_quoted_summary_line_is_still_not_a_training_step():
    """`train_loss`, not `loss` -- and its epoch must not become a data point."""
    assert is_log(AXOLOTL_SUMMARY)


def test_a_quoted_value_that_only_starts_with_a_number_is_not_a_measurement():
    """The closing quote is required, so a prefix cannot be truncated into one."""
    assert is_log("{'loss': '0.7 (estimated)'}")


def test_a_quoted_non_finite_loss_drops_out_but_its_epoch_does_not():
    """The same rule as the unquoted form, which the quoting must not change."""
    assert metric("{'loss': 'nan', 'epoch': '1.0'}") == {"epoch": 1.0}
