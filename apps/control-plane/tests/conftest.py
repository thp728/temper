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
    from temper_control_plane import fake_provider, orchestrator

    monkeypatch.setattr(orchestrator, "launch", lambda job_id: None)
    # The checked-in image contract starts unpublished; tests that drive a
    # real `run_job` (e.g. the simulated-machine journeys) inject a reference
    # so the pull-by-digest path is exercised rather than the refusal.
    monkeypatch.setattr(
        orchestrator,
        "published_reference",
        lambda: fake_provider.PUBLISHED_IMAGE_REFERENCE,
    )
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
def no_real_remote_datasets(monkeypatch):
    """No test imports a public dataset over the network.

    Like `no_real_models`, fetching a public dataset costs nothing and risks
    no billing account, so this replaces only the one seam application code
    calls through -- `remote_datasets.RESOLVER` -- with an empty fake that
    refuses every reference as `dataset_not_found`. A test that wants to
    import something seeds the fake with its own references; a test that
    forgets fails loudly instead of silently reaching the network.
    """
    from temper_control_plane import fake_remote_datasets, remote_datasets

    monkeypatch.setattr(
        remote_datasets,
        "RESOLVER",
        fake_remote_datasets.FakeRemoteDatasets({}),
    )


@pytest.fixture(autouse=True)
def no_real_tokenizer(monkeypatch):
    """No test downloads or loads a real tokenizer over the network.

    The tokenizer seam (`tokenize.TOKENIZER`) is replaced with the deterministic
    fake from `fake_tokenizer.py` -- one token per character -- so background
    token counting in tests is exact, cheap and offline, mirroring how the
    model-facts and quote seams are replaced.
    """
    from temper_control_plane import tokenize
    from temper_control_plane.fake_tokenizer import fake_tokenizer

    monkeypatch.setattr(tokenize, "TOKENIZER", fake_tokenizer())


@pytest.fixture(autouse=True)
def no_leaked_counting_threads():
    """Join any in-flight counting thread before the test's database is torn
    down.

    Counting is a background phase (issue #42): a test that validates a
    dataset but does not wait for the count leaves its counting thread running
    into teardown. On Windows a still-alive thread that connects after the
    next test points `db.DB_PATH` at its own fresh file collides with that
    test's `PRAGMA journal_mode=WAL` and surfaces as a spurious "database is
    locked" setup error. Joining here keeps the phase inside its own test,
    which is also the honest version of what the test is asserting.
    """
    import threading

    yield
    for t in threading.enumerate():
        if t.name.startswith("count-") and t is not threading.current_thread():
            t.join(timeout=10)


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
