"""The resolver and the trainer agree on what a complete specification is.

These assertions used to pin two hand-copied tables against each other: one
here, one in `entrypoint.py`. Issue #83 deleted the trainer's copy -- it
resolves nothing now -- so what has to agree is narrower and sharper: every
key the trainer requires must be present in the set the control plane's
resolver produces, because resolution happens there before launch (#83) and a
key missing from both would otherwise fail on a paid machine instead of here.

The dependency still runs this way round on purpose: an app may import a
package, never the reverse (ADR-0010).
"""

from __future__ import annotations

import entrypoint

from temper_core import feasibility, hyperparams


def test_the_trainer_requires_exactly_what_the_resolver_produces():
    """`hyperparams.effective` output is the job spec's hyperparameters block.
    If a default is added to the resolver without the trainer learning to read
    it, this fails here rather than as `spec_incomplete` on the GPU."""
    assert set(entrypoint.REQUIRED_HYPERPARAMETERS) == set(
        hyperparams.effective({})
    )


def test_the_epoch_count_the_estimates_are_built_on_matches_the_resolver():
    """Every duration estimate multiplies by this number. A default that moved
    in the resolver without feasibility following would skew every estimate
    silently."""
    assert feasibility.DEFAULT_EPOCHS == hyperparams.DEFAULTS["num_epochs"]
