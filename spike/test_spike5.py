"""The one part of spike 5 that can be tested without spending money.

Spike 5 provisions a billing GPU, so almost none of it belongs in a suite --
"tests do not cost money" is the rule, and a suite that provisions a 4 TB disk
does not get run. What CAN be tested is the parsing: the platform names its
storage ceiling inside a rejection message, and reading that number correctly
is the difference between reporting a measured ceiling and reporting the
largest value the ladder happened to try.

The real message below is transcribed verbatim from the 2026-08-23 run and is
the reason the pattern is anchored on `hdd:` -- the rejection complains about
four other fields as well, and matching the first number in the string would
have picked up whichever complaint the backend chose to list first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from spike5 import ceiling_from_rejection  # noqa: E402

# Verbatim, from findings-spike5.json, requesting 8000 GB on an L4 VM.
REAL_REJECTION = (
    "APIError: hdd: ensure this value is less than or equal to 7200; "
    "vcpus: field required; ram_gb: field required; "
    "__root__: Either cpus or gpus should be provided, not both or neither.; "
    "hdd: ensure this value is less than or equal to 7200; "
    "hdd: ensure this value is less than or equal to 7200"
)


def test_reads_the_ceiling_out_of_the_real_rejection():
    assert ceiling_from_rejection(REAL_REJECTION) == 7200


def test_is_not_fooled_by_the_other_complaints_in_the_same_message():
    """The message names four other problems. Only the hdd bound is the ceiling."""
    noisy = (
        "APIError: num_gpus: ensure this value is less than or equal to 8; "
        "hdd: ensure this value is less than or equal to 7200"
    )
    assert ceiling_from_rejection(noisy) == 7200


def test_returns_none_when_the_rejection_names_no_bound():
    assert ceiling_from_rejection("APIError: insufficient capacity") is None
    assert ceiling_from_rejection("") is None


def test_returns_none_when_the_bound_is_on_some_other_field():
    """A ceiling on num_gpus is not a ceiling on disk, and must not be read as one."""
    assert ceiling_from_rejection(
        "num_gpus: ensure this value is less than or equal to 8") is None
