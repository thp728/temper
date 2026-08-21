"""What the orchestration path raises: a failure, and a stop that is not one.

They live in their own module because both the orchestrator and the provider
implementations raise them, and neither should import the other: the whole
point of the provider seam is that the orchestrator depends on a protocol, not
on the module that implements it.
"""

from __future__ import annotations


class OrchestratorError(Exception):
    """A failure with a stable machine-readable code.

    Every API error carries a code the caller can branch on; a human message
    alone forces callers to match on prose that will change.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Cancelled(Exception):
    """The user asked for this job to stop, and something noticed.

    Deliberately *not* an `OrchestratorError`. Every code that type carries
    names a defect, and a job the user chose to stop is not one: it ends
    `cancelled`, with no error code, and anything that catches failures
    generically must not catch this by inheritance and quietly relabel it.
    """

