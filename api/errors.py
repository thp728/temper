"""The one error type the orchestration path raises.

It lives in its own module because both the orchestrator and the provider
implementations raise it, and neither should import the other: the whole point
of the provider seam is that the orchestrator depends on a protocol, not on the
module that implements it.
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
