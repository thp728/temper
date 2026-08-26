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
