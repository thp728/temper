"""Serving: a temporary authenticated endpoint that stops itself.

Spec 011's last flow. The product hands back a file; an endpoint lets a
user try the model without downloading anything. Endpoints that outlive
their usefulness are the top complaint against the commercial baseline --
training is cheap and the forgotten warm machine is the bill -- so stopping
itself is the feature, not a convenience.

Two timeouts, both domain constants (ADR-0062: derived against measured
cost, not a deployment knob):

* IDLE_TIMEOUT -- how long an endpoint sits idle before it stops. Each
  successful inference extends this window, so a user actively probing the
  model does not lose it mid-session.
* MAX_LIFETIME -- the hard ceiling from creation. Even a busy endpoint
  dies at this age, so traffic cannot keep a warm machine alive forever.
  The interaction of the two is the feature the second-most-fakeable
  criterion exists to catch: "extends on use" and "expires" must coexist.

Both are derived once here and read everywhere (ADR-0010): the control
plane reads them to mint expiries, the web reads them to show "stops at",
and the tests read them to prove the interaction. A value two components
must agree on is defined once and read, never retyped.

Keys are stored hashed (SHA-256). A key that can be read back out of the
store is a finding, not a feature.

Pure and typed (ADR-0010's package is pure): no I/O, no framework import.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

# ---------------------------------------------------------------------------
# The expiry window
# ---------------------------------------------------------------------------

# How long an idle endpoint lives before it stops itself.
#
# Derivation: 15 minutes (900s).
#
# * The same window as the training stall detector (15 min, ADR-0002)
#   because a silent training job and an idle serving machine are the same
#   failure mode -- a GPU billing while nobody watches it -- and a user who
#   learns one timeout has learned both.
# * Measured: the longest legitimately quiet stretch on a real training run
#   was 183s of image build; 15 minutes is an order of magnitude above that
#   and far below an unattended overnight (the failure this exists to prevent).
# * For serving specifically, a single inference returns in seconds; 15 min
#   lets a user try a handful of prompts without re-provisioning, while a
#   forgotten endpoint left overnight still dies in the first quarter-hour.
#
# Judgment, not measurement, so recorded as such. A busy endpoint keeps
# extending this window up to MAX_LIFETIME, so this is the idle grace, not
# the total bill.
ENDPOINT_IDLE_TIMEOUT_S: float = 15 * 60

# The hard ceiling from creation, regardless of use.
#
# Derivation: 2 hours (7200s).
#
# * At the cheapest GPU the platform provisions (L4 at Rs 41.31/hr), 2h
#   costs ~Rs 82.62 -- about 0.16% of the Rs 50,000 grant and ~0.8% of the
#   Rs 10,000 spend ceiling (ADR-0063). A forgotten endpoint that is kept
#   alive by traffic is still capped at a bill a reviewer can see is small.
# * 2h is enough to try a model interactively (the side-by-side comparison
#   is one prompt; a user trying ten prompts still fits) without forcing a
#   re-provision every few minutes, which a shorter ceiling (e.g., 30 min)
#   would do and which teaches users not to rely on the endpoint at all.
# * Considered and rejected: 30 min (too short for interactive use), 8h
#   (lets a busy endpoint bill 4x as much for no additional utility), and
#   "no ceiling, idle only" (the exact failure the criterion says to close:
#   a busy endpoint kept alive forever by traffic).
#
# Like the idle timeout, a domain constant with its derivation written beside
# it (ADR-0062), not a deployment setting.
ENDPOINT_MAX_LIFETIME_S: float = 2 * 60 * 60

# How many hex characters of the key to show as a prefix (for UI/listings
# that confirm "which key" without revealing the key). The full key is 32
# urlsafe bytes (~43 chars); 8 chars is enough to distinguish keys and short
# enough to not tempt a brute-force.
API_KEY_PREFIX_LEN: int = 8


def generate_api_key() -> str:
    """One fresh API key, urlsafe and unpredictable.

    32 bytes of entropy (~43 urlsafe chars) is well beyond brute-force
    for a temporary endpoint that lives at most 2h.
    """
    return secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """The stored form of an API key: SHA-256 hex, never the key itself.

    A key that can be read back out of the store is a finding, not a
    feature (spec 011: "keys are stored hashed"). SHA-256 is sufficient
    here: the key is already 256 bits of entropy, so a salted slow hash
    (bcrypt/argon2) buys nothing but latency on every request.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def verify_api_key(provided: str, stored_hash: str) -> bool:
    """Constant-time check of a presented key against a stored hash."""
    return hmac.compare_digest(hash_api_key(provided), stored_hash)


def key_prefix(key: str) -> str:
    """The display prefix of a key (first N chars), never the key itself."""
    return key[:API_KEY_PREFIX_LEN]


def compute_expiry(
    now: float, idle_timeout: float = ENDPOINT_IDLE_TIMEOUT_S
) -> float:
    """When an endpoint created or just used at `now` should expire idle."""
    return now + idle_timeout


def compute_max_expiry(
    now: float, max_lifetime: float = ENDPOINT_MAX_LIFETIME_S
) -> float:
    """The hard ceiling for an endpoint created at `now`."""
    return now + max_lifetime


def extend_expiry(
    now: float,
    max_expires_at: float,
    idle_timeout: float = ENDPOINT_IDLE_TIMEOUT_S,
) -> float:
    """Next idle expiry after a use at `now`, capped by the hard ceiling.

    The busy-endpoint-still-dies property: even if traffic keeps extending
    the idle window, the endpoint never lives past `max_expires_at`.
    """
    candidate = now + idle_timeout
    return candidate if candidate < max_expires_at else max_expires_at
