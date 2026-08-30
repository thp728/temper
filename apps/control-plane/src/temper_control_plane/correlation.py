"""Correlation identifier threaded from request through job through machine.

Spec 008 / issue #52: "Structured logs with a correlation identifier threaded
from request through job through machine, so one failure reads as one story."

One request produces one identifier that appears on every log line the
resulting job emits, and the identifier is surfaced on user-visible errors so
a report can be traced. The identifier travels in three places:

* **In-memory** as a ``contextvars.ContextVar`` -- the request handler sets it,
  ``jobs.create`` reads it to freeze onto the row, the worker re-hydrates it
  from the row before driving the job. No thread-local, because the
  orchestrator already runs off the request thread (``run_job`` inside the
  worker) and ``contextvars`` is what survives that hop when explicitly
  propagated.
* **On the job row** as ``jobs.correlation_id`` -- so a worker that starts
  a minute later still knows the request it came from, and a log line emitted
  on the machine can name the same identifier the request's error carried.
* **On the wire** as ``X-Correlation-ID`` (and ``X-Request-ID`` alias) -- the
  response header a reporter pastes, and the request header a caller can set
  to continue a story across retries. When set, that header is honoured rather
  than overwritten, so a caller that retries with the same identifier threads
  its own retry into the same trace.

A value two components must agree on is defined once and read, never retyped:
the header names, the context var, and the generation rule live here and are
imported everywhere else.
"""

from __future__ import annotations

import contextvars
import uuid

# The in-memory holder. ``None`` means no request is in flight on this
# context -- a worker driving a job without a request header re-hydrates from
# the row instead, and a bare startup log has no correlation at all.
correlation_id_var: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("correlation_id", default=None)
)

# Header names, defined once and read by the middleware and the tests -- never
# retyped. Both spellings are honoured on ingress: ``X-Correlation-ID`` is the
# domain language (spec 008) and ``X-Request-ID`` the ubiquitous alias; on
# egress both are sent so a reporter can paste whichever they saw.
CORRELATION_HEADER = "x-correlation-id"
REQUEST_ID_HEADER = "x-request-id"


def generate_correlation_id() -> str:
    """A fresh identifier for a request that arrived without one.

    Prefixed so a grep for ``req_`` finds every correlation identifier without
    also matching job ids (``job_``) or dataset ids (``ds_``), and so a raw
    uuid is never mistaken for a correlation identifier in a log.
    """
    return f"req_{uuid.uuid4().hex[:12]}"


def get_correlation_id() -> str | None:
    """The correlation identifier on this context, if any."""
    return correlation_id_var.get()


def set_correlation_id(value: str | None) -> None:
    """Bind the correlation identifier for this context."""
    correlation_id_var.set(value)
    # Mirror into structlog's contextvars so ``merge_contextvars`` sees it
    # even when the logger was obtained before the middleware bound it, and
    # into Sentry's scope so the error report carries the same story.
    try:
        from structlog import contextvars as _sl_cv

        if value is None:
            _sl_cv.unbind_contextvars("correlation_id")
        else:
            _sl_cv.bind_contextvars(correlation_id=value)
    except Exception:  # noqa: S110
        pass
    try:
        import sentry_sdk

        if value is None:
            sentry_sdk.set_tag("correlation_id", "")
        else:
            sentry_sdk.set_tag("correlation_id", value)
    except Exception:  # noqa: S110
        pass


def ensure_correlation_id() -> str:
    """The correlation identifier for this context, generating one if absent.

    Used at the start of a request: a caller-provided identifier is honoured
    when present, otherwise a fresh one is generated and bound.
    """
    cid = get_correlation_id()
    if cid:
        return cid
    cid = generate_correlation_id()
    set_correlation_id(cid)
    return cid
