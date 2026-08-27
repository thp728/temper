"""Suite-wide guards.

One rule, enforced rather than remembered: no test builds a real provider
client. A suite that can reach the billing account by accident is a suite that
eventually does.
"""

import pytest


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Redirect every writable surface into this test's own directory.
    Nothing a test does may reach the checkout's real data/ tree.

    Fixtures that need the app add their TestClient on top of this one. This
    deliberately does NOT neuter `orchestrator.launch`: the orchestrator
    tests swap it for a driver they control, including the real threaded
    one, and a stub here would be what they captured.
    """
    from temper_control_plane import db, storage

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(
        storage,
        "STORE",
        storage.FilesystemStorage(root=tmp_path / "objects"),
    )
    db.init()


@pytest.fixture()
def client(isolated, monkeypatch):
    # Never launch a real VM from a test.
    from temper_control_plane import orchestrator

    monkeypatch.setattr(orchestrator, "launch", lambda job_id: None)
    from fastapi.testclient import TestClient

    from temper_control_plane import main

    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def peak_memory():
    """Peak Python-heap allocation, in bytes, while `run` executes.

    The measurement behind spec 006's flat-memory clause. tracemalloc traces
    this process's own allocations -- the control-plane side of a transfer,
    which is the thing under test; RSS would bury the signal under the
    interpreter baseline. Whatever `run` retains only in C-level buffers or
    in another process is invisible here by construction, which is the right
    blindness: an implementation that accumulates in Python objects is the
    failure mode the clause names.
    """

    def measure(run):
        import tracemalloc

        tracemalloc.start()
        try:
            run()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    return measure


@pytest.fixture(autouse=True)
def no_real_provider(monkeypatch):
    from temper_control_plane import provider

    def refuse():
        raise AssertionError(
            "A test tried to construct a real provider client. Pass a "
            "FakeProvider instead — the real one costs money."
        )

    monkeypatch.setattr(
        provider.JarvisLabsProvider, "_connect", staticmethod(refuse)
    )


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    """No test resolves model facts over the network.

    Unlike the provider, reaching Hugging Face costs nothing and risks no
    billing account, so this is narrower than `no_real_provider`: only
    `main.MODELS` -- the one seam application code actually calls through --
    is replaced with the catalog's real facts, seeded once and checked into
    `fake_models.py`, so `/v1/models` behaves exactly as it would in
    production. It does not touch `urllib` globally, because other suites
    (`test_storage.py`'s moto-backed HTTP round trip) legitimately make real
    loopback requests; a test exercising `HuggingFaceModels` itself
    monkeypatches `urllib.request.urlopen` locally instead.
    """
    from temper_control_plane import fake_models, main

    monkeypatch.setattr(main, "MODELS", fake_models.catalog_models())


@pytest.fixture(autouse=True)
def no_real_quote_provider(monkeypatch):
    """Quotes are priced against a fake provider and fake model facts, like
    everything else that could reach the account or the network.

    The plan screen computes a quote per catalog model on every load, so any
    test that renders it would construct a provider; the suite-wide refusal in
    `no_real_provider` is exactly what should catch a real one. Pointing the
    quote seams at the fakes keeps every quote deterministic (L4 at 41.31 INR,
    the spike-5 figures, and the catalog's real facts) without touching the
    account or Hugging Face.
    """
    from temper_control_plane import (
        fake_models,
        fake_provider,
    )
    from temper_control_plane import (
        quote as quote_mod,
    )

    monkeypatch.setattr(
        quote_mod, "QUOTE_PROVIDER", fake_provider.FakeProvider()
    )
    monkeypatch.setattr(
        quote_mod, "QUOTE_MODELS", fake_models.catalog_models()
    )
