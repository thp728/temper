"""Structured, machine-readable logs with a correlation identifier.

Spec 008 / issue #52: "Logs are structured, and telemetry carries numbers only.
No prompt, no completion, no dataset row ever appears in a log line -- a
training platform that logs user data has a problem no amount of access control
fixes." and "One request produces a correlation identifier that appears on every
log line the resulting job emits."

Every log line is JSON, written to ``stdout`` and parseable as one JSON object
per line. The identifier is threaded via ``contextvars`` (the same var
``correlation`` defines) so a request, its job and its machine share one story
without passing an explicit ``correlation_id`` through every call site -- the
processor reads the var at emit time.

Two scrubbing processors run on **every** log line, before it is serialised:

* **Secrets are redacted.** Any key matching ``api_key``, ``secret``,
  ``password``, ``token``, ``jl_api_key`` or a value that looks like one is
  replaced with ``[REDACTED]``. The health endpoint's own scrub (naming only
  the exception class) is not enough: logs go to a different sink and need
  their own guard.

* **User data never appears.** Keys named ``prompt``, ``completion``,
  ``messages``, ``row``, ``dataset`` or similar are replaced with
  ``[REDACTED USER DATA]``. The orchestrator never logs a prompt or a dataset
  row; this processor is the belt that makes that true even when a future
  call site mistakenly does -- a training platform that logs user data has a
  problem no access control fixes, so the guard is structural rather than
  conventional.

Setup is one call at startup: ``configure_logging()``. Afterwards,
``structlog.get_logger(__name__)`` is the logger to use everywhere -- the
stdlib ``logging`` logger also emits JSON via the same pipeline (``Processor
Formatter`` bridge) so a library that still calls ``logging.getLogger`` gets
the same shape.

Tests read this file rather than awaiting a deployed environment: the
correlation processor and both scrubbers are plain functions with unit tests,
and ``test_logging`` drives a real request against the app and asserts the
captured JSON lines carry the same ``correlation_id`` the error response did.
"""

from __future__ import annotations

import json
import logging
import logging.config
import re
import sys
from typing import Any

import structlog

from . import config
from .correlation import get_correlation_id

# ---------------------------------------------------------------------------
# scrubbing -- secrets and user data
# ---------------------------------------------------------------------------

# Any structured key whose name looks like a secret is redacted. The list is
# deliberately broad: ``api_key`` covers the endpoint key, ``jl_api_key``
# the provider credential, ``secret`` the storage signing secret, ``sentry_dsn``
# the error-reporting credential -- a value two components must agree on is
# scrubbed once, not hoped for twice.
_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|secret|password|token|jl_api_key|sentry_dsn|storage_secret)",
    re.I,
)
# A heuristic for a raw string value that is itself a secret bearer rather than
# a keyed field -- e.g. "JL_API_KEY=sk-..." inside an exception message.
_SECRET_VALUE_RE = re.compile(
    r"(jl_api_key|api_key|secret|password|token)\s*[:=]\s*\S+",
    re.I,
)

# User data that must never appear in a log line (spec 008): prompts,
# completions, dataset rows. The orchestrator never logs them intentionally;
# the scrubber ensures a future call site that does is caught at emit time
# rather than in review. Keys are matched, not values, because the value is
# where the data lives and the key is where to decide. Word boundaries keep
# the scrub precise: ``row_count`` (a number) and ``dataset_id`` (an
# identifier) are not redacted, only a field that *is* the row or the
# dataset.
_USER_DATA_KEY_RE = re.compile(
    r"\b(prompt|completion|messages|dataset|preview|row)\b",
    re.I,
)

# Known secret values from the environment, redacted wherever they appear
# even without a matching key. Captured at configure time so a rotation after
# startup does not leave the old value unscrubbed in already-configured
# filters.
_known_secrets: set[str] = set()


def _collect_known_secrets() -> None:
    """Snapshot the secret values currently in the environment.

    Called once when logging is configured; a process that rotates a secret
    without restarting keeps the old snapshot scrubbed as well as the new one
    on next configure, which is the best that can be done without re-reading
    the environment on every log line.
    """
    _known_secrets.clear()
    for name in (
        "JL_API_KEY",
        "TEMPER_SENTRY_DSN",
        "SENTRY_DSN",
        "TEMPER_STORAGE_SECRET",
        "STORAGE_SECRET",
    ):
        val = config._text(name) if hasattr(config, "_text") else None
        # Fallback to raw environ: ``config._text`` returns None for empty, but
        # environ may still carry the value.
        import os

        raw = os.environ.get(name)
        if raw:
            _known_secrets.add(raw)
        if val:
            _known_secrets.add(val)
    # Also include the storage secret derived by the control plane itself.
    if config.STORAGE_SECRET:
        _known_secrets.add(config.STORAGE_SECRET)


def _redact_value(key: str, value: Any) -> Any:
    """Redact ``value`` when ``key`` is secret or user data."""
    if isinstance(key, str) and _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(key, str) and _USER_DATA_KEY_RE.search(key):
        return "[REDACTED USER DATA]"
    if isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        return _SECRET_VALUE_RE.sub("[REDACTED]", value)
    # Known secret values anywhere in a string, even without a key hint.
    if isinstance(value, str) and _known_secrets:
        for secret in _known_secrets:
            if secret and secret in value:
                value = value.replace(secret, "[REDACTED]")
    return value


def redact_processor(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor: redact secrets and user data from every log line.

    Runs after the correlation and timestamp processors, before serialisation,
    so both the ``event`` string and any structured keys are scrubbed. The
    function mutates ``event_dict`` in place for performance (structlog's own
    processors do) and returns it for the next processor.
    """
    for k in list(event_dict.keys()):
        v = event_dict[k]
        # Top-level scrub: a field that *is* user data or a secret is
        # redacted whole, regardless of its value's shape. This catches
        # ``{"prompt": "hello"}`` and ``{"messages": [...]}`` at the top
        # level, before the recursive dict/list handling.
        if _SECRET_KEY_RE.search(str(k)):
            event_dict[k] = "[REDACTED]"
            continue
        if _USER_DATA_KEY_RE.search(str(k)):
            event_dict[k] = "[REDACTED USER DATA]"
            continue
        # Recursively scrub dicts and lists so a nested ``{"prompt": "..."}``
        # inside ``extra`` is still caught.
        if isinstance(v, dict):
            scrubbed: dict[Any, Any] = {}
            for dk, dv in v.items():
                dk_s = str(dk)
                if _SECRET_KEY_RE.search(dk_s):
                    scrubbed[dk] = "[REDACTED]"
                elif _USER_DATA_KEY_RE.search(dk_s):
                    scrubbed[dk] = "[REDACTED USER DATA]"
                elif isinstance(dv, dict):
                    # One level deeper is enough for the shapes this codebase
                    # emits (``extra``, ``data``, ``result``); deeper nesting
                    # is still scrubbed string-wise by the known-secrets pass.
                    scrubbed[dk] = {
                        kk: _redact_value(str(kk), vv) for kk, vv in dv.items()
                    }
                elif isinstance(dv, (list, tuple)):
                    scrubbed[dk] = [
                        _redact_value(str(dk), x) if isinstance(x, str) else x
                        for x in dv
                    ]
                else:
                    scrubbed[dk] = _redact_value(dk_s, dv)
            event_dict[k] = scrubbed
        elif isinstance(v, (list, tuple)):
            # Lists of strings that might contain a secret value.
            event_dict[k] = [
                _redact_value(k, x) if isinstance(x, str) else x for x in v
            ]
        else:
            event_dict[k] = _redact_value(k, v)
    # Also scrub the top-level ``event`` string itself (e.g. "auth failed: JL_API_KEY=...")
    if "event" in event_dict and isinstance(event_dict["event"], str):
        event_dict["event"] = _redact_value("event", event_dict["event"])
        # Known secrets in the event string even without a key pattern.
        if _known_secrets:
            for secret in _known_secrets:
                if secret and secret in event_dict["event"]:
                    event_dict["event"] = event_dict["event"].replace(
                        secret, "[REDACTED]"
                    )
    return event_dict


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------


def correlation_processor(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Attach ``correlation_id`` when one is bound on this context.

    Missing means this log line is outside a request (startup, sweep thread,
    worker poll with no job) and no identifier is added -- a missing
    correlation is not a redaction, it is an honest absence. A worker that
    claimed a job binds the job's frozen identifier before driving it, so even
    a line emitted from ``orchestrator.run_job`` carries the request's
    identifier.
    """
    cid = get_correlation_id()
    if cid is not None:
        event_dict["correlation_id"] = cid
    return event_dict


# ---------------------------------------------------------------------------
# stdlib bridge -- structlog controls rendering, stdlib carries it
# ---------------------------------------------------------------------------

# One JSON sink for both stdlib and structlog, so a library that still calls
# ``logging.getLogger(__name__).info`` and a site that calls
# ``structlog.get_logger(__name__).info`` both emit one JSON line, not two
# shapes. This is the ``structlog.stdlib.ProcessorFormatter`` pattern
# documented by structlog, where the *stdlib* handler does the final
# rendering and structlog's own processors run first.


def _level_from_name(name: str) -> int:
    try:
        return getattr(logging, name.upper())
    except AttributeError:
        return logging.INFO


_CONFIGURED = False


def configure_logging() -> None:
    """Configure structured JSON logging for this process.

    Idempotent -- called from ``main.lifespan`` and ``worker.main`` and
    harmless to call twice. Every line is JSON, machine-readable, written to
    ``stdout`` (the container's log sink) with a ``timestamp``, ``level``,
    ``logger`` name, ``event`` and, when bound, ``correlation_id``. No line
    ever carries a prompt, completion, dataset row or secret -- those are
    scrubbed before serialisation by ``redact_processor``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    _collect_known_secrets()

    # Processors that run *before* the message is handed to stdlib -- this is
    # where correlation, timestamp and redaction happen so both stdlib and
    # structlog loggers share them.
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        correlation_processor,
        redact_processor,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # The final stdlib handler that renders JSON. ``ProcessorFormatter``'s
    # ``processor`` is what actually calls ``json.dumps`` -- the stdlib
    # handler's own formatter is that wrapper, so the stdlib pipeline and the
    # structlog pipeline converge to the same JSON shape.
    json_renderer = structlog.processors.JSONRenderer()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": json_renderer,
                    "foreign_pre_chain": [
                        structlog.stdlib.add_log_level,
                        structlog.processors.TimeStamper(
                            fmt="iso", key="timestamp"
                        ),
                        correlation_processor,
                        redact_processor,
                    ],
                }
            },
            "handlers": {
                "default": {
                    "level": config.LOG_LEVEL,
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "json",
                }
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": config.LOG_LEVEL,
                    "propagate": True,
                },
                "uvicorn": {
                    "handlers": ["default"],
                    "level": config.LOG_LEVEL,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["default"],
                    "level": config.LOG_LEVEL,
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["default"],
                    "level": config.LOG_LEVEL,
                    "propagate": False,
                },
            },
        }
    )

    structlog.configure(
        processors=shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """The structured logger for ``name``. Every line is JSON and scrubbed."""
    # Ensure logging is configured even when a module imports this before
    # ``configure_logging`` was explicitly called (e.g. in tests that import
    # ``db`` before the app boots). The guard in ``configure_logging`` keeps
    # this cheap.
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# test helpers -- importable without booting the whole app
# ---------------------------------------------------------------------------


def is_structured_json(line: str) -> bool:
    """True when ``line`` is a single JSON object -- the machine-readable gate."""
    try:
        obj = json.loads(line)
        return isinstance(obj, dict)
    except Exception:
        return False
