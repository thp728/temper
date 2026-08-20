"""Suite-wide guards.

One rule, enforced rather than remembered: no test builds a real provider
client. A suite that can reach the billing account by accident is a suite that
eventually does.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def no_real_provider(monkeypatch):
    from api import provider

    def refuse():
        raise AssertionError(
            "A test tried to construct a real provider client. Pass a "
            "FakeProvider instead — the real one costs money.")

    monkeypatch.setattr(provider.JarvisLabsProvider, "_connect",
                        staticmethod(refuse))
