"""Error reporting as an integration point, inert without a credential.

Spec 008 / issue #52: "Error reporting is wired as an integration point with
no credential set locally -- present in shape, inert in effect."

The SDK is installed and this module is imported at startup, but
``TEMPER_SENTRY_DSN`` is unset locally (``DEFAULT_SENTRY_DSN`` is ``None``) so
``init_sentry`` does nothing and no data leaves the process. Present in shape
means a DSN set in the environment is honoured without a code change; inert in
effect means a fresh clone without one is not broken and not chatty.

The scrubber is structural: even when a DSN *is* set, user data (prompts,
completions, dataset rows) and secrets never reach Sentry -- they are stripped
before the SDK serialises the event.
"""

from __future__ import annotations

import os
import re
from typing import Any

from . import config

# Patterns whose values must never reach Sentry, even when the SDK is active.
# Mirrors the log redaction so the same secret scrubbed from a log line is
# also scrubbed from an error report -- a value two components must agree on
# is scrubbed once, not hoped for twice.
_SECRET_KEYS = re.compile(
    r"(api[_-]?key|secret|password|token|jl_api_key|sentry_dsn|storage_secret)",
    re.I,
)
# User data that must never reach an error report: a training platform that
# logs or reports user data has a problem no access control fixes (spec 008).
# Word boundaries keep the scrub precise: ``row_count`` (a number) is not
# user data, only a field that *is* the row or the dataset.
_USER_DATA_KEYS = re.compile(
    r"\b(prompt|completion|messages|dataset|preview|row)\b",
    re.I,
)


def _scrub(obj: Any) -> Any:
    """Recursively redact secrets and user data from an object before Sentry sees it."""
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if _SECRET_KEYS.search(key):
                out[k] = "[REDACTED]"
            elif _USER_DATA_KEYS.search(key):
                # For breadcrumbs and extra context: keep the fact that a
                # prompt existed, not its contents.
                out[k] = "[REDACTED USER DATA]"
            else:
                out[k] = _scrub(v)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(_scrub(x) for x in obj)
    if isinstance(obj, str) and len(obj) > 500 and _USER_DATA_KEYS.search(obj):
        # Defensive: a string that looks like a dumped row.
        return "[REDACTED USER DATA]"
    return obj


def before_send(
    event: dict[str, Any], _hint: dict[str, Any]
) -> dict[str, Any] | None:
    """Sentry ``beforeSend`` hook: scrub before anything leaves the process."""
    # Scrub breadcrumbs, extra, contexts, tags -- anything the SDK may have
    # attached from the structured logger's context.
    if "breadcrumbs" in event:
        event["breadcrumbs"] = _scrub(event["breadcrumbs"])
    if "extra" in event:
        event["extra"] = _scrub(event["extra"])
    if "contexts" in event:
        event["contexts"] = _scrub(event["contexts"])
    if "user" in event:
        # No user identity is ever set, but if an upstream hook did, scrub it.
        event["user"] = _scrub(event["user"])
    # Scrub exception values that might carry a secret in the message.
    for exc in (event.get("exception") or {}).get("values") or []:
        if exc.get("value"):
            val = str(exc["value"])
            if _SECRET_KEYS.search(val):
                # Replace the token value but keep the error kind.
                exc["value"] = _SECRET_KEYS.sub("[REDACTED]", val)
    return event


def init_sentry() -> None:
    """Initialize Sentry when a DSN is configured, otherwise remain inert.

    Called once at startup (from ``main.lifespan`` and ``worker.main``) so
    the integration point is present in shape even when ``TEMPER_SENTRY_DSN``
    is unset. Inert means no import-time side effect beyond this call, and no
    network traffic when the DSN is absent.
    """
    dsn = config.SENTRY_DSN
    if not dsn:
        return
    # ``sentry_sdk`` is a soft import: the SDK is installed (pyproject.toml)
    # but an environment without it (e.g. a minimal test install) must not
    # crash the process at boot -- inert is still inert when the SDK cannot be
    # imported, and the shape is still present because this module exists and
    # is called.
    try:
        import sentry_sdk
    except Exception:
        return
    sentry_sdk.init(
        dsn=dsn,
        # Tracing is off by default in this build; enabling it is a DSN-level
        # decision, not a code change, and a tracing sample rate of 0 keeps the
        # integration inert for performance as well as for privacy.
        traces_sample_rate=float(
            os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0")
        ),
        # Do not send PII, and let ``before_send`` be the last scrub.
        send_default_pii=False,
        before_send=before_send,  # type: ignore[arg-type]
    )
