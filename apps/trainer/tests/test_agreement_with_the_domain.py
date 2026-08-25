"""The trainer and the domain hold the same table, and this is what makes that safe.

These two assertions used to live in `packages/core/tests`, where they had to
`import entrypoint` to run -- a pure package whose tests reached into an
application. The dependency belongs this way round: an app may import a package,
never the reverse (ADR-0010).

Both exist only because the table is duplicated at all. Issue #82 collapses it
to one definition in `packages/contracts`, at which point these become a check
that the trainer reads what it was given, and this file is where that check
already lives.
"""

from __future__ import annotations

import entrypoint

from temper_core import feasibility, hyperparams


def test_the_resolved_defaults_match_the_trainers():
    """The page's promise and the trainer's behaviour come from two copies of one
    table. If either copy changes alone, this fails and the drift is caught
    before a user is told something false."""
    assert hyperparams.DEFAULTS == entrypoint.DEFAULTS
    assert hyperparams.ALLOWED_OVERRIDES == entrypoint.ALLOWED_OVERRIDES


def test_the_epoch_count_estimates_are_built_on_matches_the_trainers():
    """Every duration estimate multiplies by this number. A trainer default that
    moved without it would skew every estimate silently."""
    assert feasibility.DEFAULT_EPOCHS == entrypoint.DEFAULTS["num_epochs"]
