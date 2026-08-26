"""Suite-wide guards.

One rule, enforced rather than remembered: no test builds a real provider
client. A suite that can reach the billing account by accident is a suite that
eventually does.
"""

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from temper_control_plane import db, main, orchestrator, storage

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    # Stored objects land in the test's own directory: nothing a test does
    # may reach the checkout's data/ tree.
    monkeypatch.setattr(
        storage,
        "STORE",
        storage.FilesystemStorage(root=tmp_path / "objects"),
    )
    # Never launch a real VM from a test.
    monkeypatch.setattr(orchestrator, "launch", lambda job_id: None)
    from fastapi.testclient import TestClient

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
