"""Serving domain constants and key handling (issue #78).

The endpoint is temporary by construction: it carries an expiry from the
moment it starts, extends on use, and stops itself. The forgotten warm
machine is the loudest complaint against the commercial baseline, so
stopping itself is the feature. Keys are stored hashed.
"""

from temper_core import serving


def test_idle_and_max_are_positive_and_idle_shorter_than_max():
    assert serving.ENDPOINT_IDLE_TIMEOUT_S > 0
    assert serving.ENDPOINT_MAX_LIFETIME_S > 0
    assert serving.ENDPOINT_IDLE_TIMEOUT_S < serving.ENDPOINT_MAX_LIFETIME_S


def test_idle_is_15_minutes_and_max_is_2_hours():
    # The derivation is in the module docstring; the test pins the numbers
    # so a later change is deliberate, not accidental.
    assert serving.ENDPOINT_IDLE_TIMEOUT_S == 15 * 60
    assert serving.ENDPOINT_MAX_LIFETIME_S == 2 * 60 * 60


def test_key_is_hashed_and_verifiable_constant_time():
    key = serving.generate_api_key()
    h = serving.hash_api_key(key)
    assert h != key
    assert len(h) == 64  # SHA-256 hex
    assert serving.verify_api_key(key, h) is True
    assert serving.verify_api_key(key + "x", h) is False
    assert serving.verify_api_key("", h) is False


def test_key_prefix_is_not_the_key():
    key = serving.generate_api_key()
    prefix = serving.key_prefix(key)
    assert prefix == key[: serving.API_KEY_PREFIX_LEN]
    assert len(prefix) == serving.API_KEY_PREFIX_LEN
    assert prefix != key


def test_extend_is_capped_by_max():
    now = 1000.0
    max_at = now + serving.ENDPOINT_MAX_LIFETIME_S
    # Far before max, extend is now+idle
    assert (
        serving.extend_expiry(now, max_at)
        == now + serving.ENDPOINT_IDLE_TIMEOUT_S
    )
    # Near max, extend is capped
    near_max = max_at - 10
    assert serving.extend_expiry(near_max, max_at) == max_at
    # At max, extend stays at max
    assert serving.extend_expiry(max_at, max_at) == max_at
    # Past max, extend stays at max (never beyond)
    assert serving.extend_expiry(max_at + 100, max_at) == max_at


def test_compute_expiry_and_max():
    now = 2000.0
    assert serving.compute_expiry(now) == now + serving.ENDPOINT_IDLE_TIMEOUT_S
    assert (
        serving.compute_max_expiry(now)
        == now + serving.ENDPOINT_MAX_LIFETIME_S
    )


def test_idle_and_max_interact_busy_endpoint_still_dies():
    """A busy endpoint that is kept alive by traffic still dies at the max.

    This is the second-most-fakeable criterion: "extends on use" and
    "expires" must coexist, so a busy endpoint cannot be kept alive forever.
    """
    start = 0.0
    max_at = serving.compute_max_expiry(start)
    # Simulate traffic every idle/2: each use extends idle but never beyond max
    now = start
    for _ in range(100):
        now += serving.ENDPOINT_IDLE_TIMEOUT_S / 2
        now_expires = serving.extend_expiry(now, max_at)
        assert now_expires <= max_at
        if now >= max_at:
            break
    # After enough extends, the next expiry is exactly max
    assert serving.extend_expiry(max_at - 1, max_at) == max_at
    # And after max, it stays at max
    assert serving.extend_expiry(max_at + 1000, max_at) == max_at
