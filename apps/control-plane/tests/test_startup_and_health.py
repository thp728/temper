"""Issue #29: startup, configuration and health.

The acceptance criteria this file pins, each against the mechanism that makes
it honest rather than asserted:

- **The system starts with no configuration set and no secrets present.** A
  subprocess boots the whole app from an environment with every `TEMPER_*` and
  `JL_*` variable removed, and the test asserts what *happens*: it boots, the
  health endpoint answers with every dependency ready, and the resolved
  defaults are the documented ones. Asserting defaults in-process instead
  would be asserting that defaults exist; running the process is showing it.
- **A health check reports on each dependency separately.** The healthy shape
  is per-dependency, and a broken database is reported as a broken database
  while the object store (and the app) keep answering.
- **Stopping and restarting preserves data.** One boot writes a dataset, its
  object and a job; a second boot over the same paths reads them all back.
- **Whether real compute is used is a single setting.** Flipping
  `TEMPER_FAKE_PROVIDER` changes which provider answers and nothing else in
  the health surface, and a default start lands on the real tier with the
  fault surface off (the safe mode, ADR-0051/ADR-0062).

**A finding from issue #43, load-bearing for the first criterion above.**
SQLite needed no running process to satisfy "starts with nothing configured":
the file simply existed or was created. PostgreSQL is a client-server system,
and there is no server-less default for it -- "nothing configured" can no
longer mean "no external process required," only "no `TEMPER_*`/`JL_*`
override required, given a database reachable at the documented default
address." `DEFAULT_DATABASE_URL` names that address
(`postgresql://temper:temper@localhost:5432/temper`, the same one
`compose.yaml`'s `postgres` service and `just db-up` bind to); this test now
stands one up at exactly that address before spawning the scrubbed-environment
subprocess, which is what makes the claim it asserts -- that the default
resolves to something real and reachable, not just that it prints a string --
still true rather than merely restated.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import psycopg
import pytest


def _remove_runtime_data(root: Path, existed_before: set[str]) -> None:
    """Remove exactly what a default boot wrote into `/data/`.

    The default boot creates the filesystem object root and nothing else
    now that the database is a server rather than a file under here.
    Only the names that were not present before this test ran are removed,
    so a developer's own object store is never touched.
    """
    data = root / "data"
    if not data.exists():
        return
    objects = data / "objects"
    if objects.is_dir() and "objects" not in existed_before:
        shutil.rmtree(objects, ignore_errors=True)
    if not any(data.iterdir()):
        data.rmdir()


# --- the system starts with no configuration and no secrets -----------------


def test_the_system_starts_with_no_configuration_and_no_secrets():
    """A fresh process, an environment with nothing set, and the product boots.

    The only honest proof of "starts with nothing configured" is to start it
    with nothing configured: this spawns a real interpreter that scrubs every
    `TEMPER_*` and `JL_*` variable, imports and boots the actual application,
    and hits its health endpoint. What the reviewer would see -- a healthy
    app that resolves to the documented defaults and refuses fault specs (the
    safe mode) -- is what is asserted.
    """
    from temper_control_plane import config, migrations

    if (config.REPO_ROOT / ".env").exists() or (
        config.REPO_ROOT / "spike" / ".env"
    ).exists():
        # With a .env present the environment is not empty, so the claim "no
        # configuration set" cannot be made exactly from this checkout. Say so
        # rather than quietly testing a configured start.
        pytest.skip(
            "this checkout carries a .env file, so 'no configuration set' "
            "cannot be demonstrated exactly; run on a clean clone"
        )

    program = textwrap.dedent(
        """\
        import json
        from fastapi.testclient import TestClient
        import temper_control_plane.config as c
        import temper_control_plane.main as m

        with TestClient(m.app) as tc:
            h = tc.get('/health')
            refusal = c.fault_surface_refusal('oom')
            print(json.dumps({
                'status': h.status_code,
                'health': h.json(),
                'fault_refusal_code': refusal['code'] if refusal else None,
                'defaults': {
                    'stall_timeout_s': c.STALL_TIMEOUT_S,
                    'max_job_duration_s': c.MAX_JOB_DURATION_S,
                    'fake_provider': c.FAKE_PROVIDER,
                    'fault_surface': c.FAULT_SURFACE,
                    'storage_backend': c.STORAGE_BACKEND,
                    'database_url': c.DATABASE_URL,
                },
            }))
        """
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TEMPER_") and not key.startswith("JL_")
    }
    data = config.REPO_ROOT / "data"
    existed_before = (
        {p.name for p in data.iterdir()} if data.exists() else set()
    )

    # Stand up a real server at exactly the documented zero-configuration
    # address, since a client-server database has no server-less default the
    # way the SQLite file did (see the module docstring's finding). Bound to
    # the fixed default port rather than testcontainers' usual random one,
    # on purpose: this proves the *default* is reachable, not merely that
    # some database somewhere is.
    from testcontainers.community.postgres import PostgresContainer

    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    container = PostgresContainer(
        "postgres:16",
        username="temper",
        password="temper",
        dbname="temper",
        port=5432,
    ).with_bind_ports(5432, 5432)
    container.start()
    try:
        with psycopg.connect(
            container.get_connection_url(driver=None)
        ) as conn:
            migrations.migrate_up(conn)

        try:
            proc = subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                env=env,
                cwd=config.REPO_ROOT,
            )
        finally:
            _remove_runtime_data(config.REPO_ROOT, existed_before)
    finally:
        container.stop()

    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])

    # The app answers, and every dependency is ready -- a broken dependency
    # would be visible here, not hidden behind a single ok.
    assert report["status"] == 200, report
    health = report["health"]
    assert health["ok"] is True
    assert set(health["dependencies"]) == {"database", "storage"}
    assert health["dependencies"]["database"]["ok"] is True
    assert health["dependencies"]["storage"]["ok"] is True

    # Nothing was required, and nothing was secretly provided: the resolved
    # defaults are the documented ones, on the real tier, with the fault
    # surface off -- the safe mode a default start must land in.
    assert report["defaults"]["stall_timeout_s"] == 15 * 60
    assert report["defaults"]["max_job_duration_s"] == 24 * 60 * 60
    assert report["defaults"]["fake_provider"] is False
    assert report["defaults"]["fault_surface"] is False
    assert report["defaults"]["storage_backend"] == "filesystem"
    assert report["defaults"]["database_url"] == config.DEFAULT_DATABASE_URL
    assert health["provider"] == "real"
    assert report["fault_refusal_code"] == "fault_surface_refused"


def test_a_no_configuration_start_resolves_to_the_documented_defaults(
    monkeypatch,
):
    """The typed Settings view of a start with nothing configured.

    `Settings()` with the environment cleared is exactly what a bare start
    resolves to; every field is its documented local default, defined once in
    the `DEFAULT_*` constants and read by both the environment readers and
    this view.
    """
    from temper_control_plane import config

    for key in [k for k in os.environ if k.startswith("TEMPER_")]:
        monkeypatch.delenv(key, raising=False)

    bare = config.Settings()
    assert bare.stall_timeout_s == config.DEFAULT_STALL_TIMEOUT_S
    assert bare.max_job_duration_s == config.DEFAULT_MAX_JOB_DURATION_S
    assert bare.quote_ttl_s == config.DEFAULT_QUOTE_TTL_S
    assert bare.max_dataset_mb == pytest.approx(config.DEFAULT_MAX_DATASET_MB)
    assert bare.fake_provider is False
    assert bare.fake_line_delay_s == 0.0
    assert bare.fault_surface is False
    assert bare.storage_backend == config.DEFAULT_STORAGE_BACKEND
    assert bare.storage_root == config.DEFAULT_STORAGE_ROOT
    assert bare.database_url == config.DEFAULT_DATABASE_URL
    assert bare.db_reset is False
    assert bare.s3_bucket is None
    assert bare.s3_endpoint_url is None
    assert bare.s3_region is None
    assert bare.storage_secret is None
    assert bare.checkpoint_retention == config.DEFAULT_CHECKPOINT_RETENTION


def test_the_settings_bundle_is_consistent_with_the_module_names():
    """The module-level names the codebase reads and the typed bundle that
    backs them cannot drift apart -- one definition, read twice."""
    from temper_control_plane import config

    assert config.settings.stall_timeout_s == config.STALL_TIMEOUT_S
    assert config.settings.max_job_duration_s == config.MAX_JOB_DURATION_S
    assert config.settings.quote_ttl_s == config.QUOTE_TTL_S
    assert config.settings.max_dataset_mb == pytest.approx(
        config.MAX_DATASET_BYTES / (1024 * 1024)
    )
    assert config.settings.fake_provider == config.FAKE_PROVIDER
    assert config.settings.fake_line_delay_s == config.FAKE_LINE_DELAY_S
    assert config.settings.fault_surface == config.FAULT_SURFACE
    assert config.settings.storage_backend == config.STORAGE_BACKEND
    assert config.settings.storage_root == config.STORAGE_ROOT
    assert config.settings.database_url == config.DATABASE_URL
    assert config.settings.db_reset == config.DB_RESET
    assert config.settings.s3_bucket == config.S3_BUCKET
    assert config.settings.s3_endpoint_url == config.S3_ENDPOINT_URL
    assert config.settings.s3_region == config.S3_REGION
    assert config.settings.storage_secret == config.STORAGE_SECRET
    assert config.settings.checkpoint_retention == config.CHECKPOINT_RETENTION


# --- a health check reports on each dependency separately --------------------


def test_health_reports_each_dependency_separately(client):
    """The healthy shape names every dependency and its own ok.

    One endpoint returning a single ok is the failure the criterion names: a
    reviewer must be able to tell a broken database from a broken application,
    and from a broken object store, from this one response.
    """
    body = client.get("/health").json()
    assert body["ok"] is True
    assert set(body["dependencies"]) == {"database", "storage"}
    assert body["dependencies"]["database"] == {"ok": True}
    assert body["dependencies"]["storage"] == {"ok": True}


def test_a_broken_database_is_reported_as_a_broken_database(
    client, monkeypatch
):
    """Point the database at an address nothing answers on: the app still
    answers (a dead process answers nothing) and names the database, not
    itself.

    **A finding from issue #43.** The SQLite version of this test pointed
    `DB_PATH` at a location that could not be created -- a filesystem
    failure, because the database *was* a file. A connection string has no
    equivalent "unwritable path"; what breaks a client-server database is an
    address nothing answers on, so that is what this now simulates. The
    failure kind reported (`OperationalError`, not `NotADirectoryError`) is
    a real, visible consequence of the migration, not an incidental
    rewording.
    """
    from temper_control_plane import db

    monkeypatch.setattr(
        db, "DATABASE_URL", "postgresql://nobody:nobody@127.0.0.1:1/nothing"
    )

    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()["detail"]
    assert body["ok"] is False
    assert body["dependencies"]["database"]["ok"] is False
    assert "database" in body["dependencies"]
    # The object store is untouched by the database being broken.
    assert body["dependencies"]["storage"]["ok"] is True


def test_a_broken_object_store_is_reported_as_a_broken_store(
    client, tmp_path, monkeypatch
):
    """The mirror image: the store cannot prepare, the database still can, and
    the response tells them apart."""
    from temper_control_plane import storage

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("occupied")
    monkeypatch.setattr(
        storage,
        "STORE",
        storage.FilesystemStorage(root=blocker / "objects"),
    )

    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()["detail"]
    assert body["ok"] is False
    assert body["dependencies"]["database"]["ok"] is True
    assert body["dependencies"]["storage"]["ok"] is False
    assert body["dependencies"]["storage"]["error"]


# --- stopping and restarting preserves data ----------------------------------


def test_stopping_and_restarting_preserves_data(isolated, tmp_path, monkeypatch):
    """One boot writes; a second boot over the same database and paths
    reads it all back.

    The restart is a second application boot (fresh lifespan, fresh storage
    instance) against the same database and object root -- which is exactly
    the seam a container restart crosses. `isolated` points both at this
    test's own throwaway database and directory for its whole duration, so
    "the same paths" now means the same database too, not a SQLite file
    reopened.
    """
    from fastapi.testclient import TestClient

    from temper_control_plane import db, main, storage

    def upload(client, path: Path) -> str:
        with open(path, "rb") as f:
            return client.post(
                "/v1/datasets", files={"file": (path.name, f)}
            ).json()["id"]

    from helpers import wait_counted, wait_validated

    dataset = tmp_path / "d.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                }
            )
            for i in range(12)
        ),
        encoding="utf-8",
    )

    # Boot one: write a dataset (row + stored object) and a job row.
    with TestClient(main.app) as boot_one:
        ds_id = upload(boot_one, dataset)
        wait_validated(boot_one, ds_id)
        wait_counted(boot_one, ds_id)
        job_id = db.create_job(ds_id, "Qwen/Qwen3-4B", {})
        assert (
            boot_one.get(f"/v1/datasets/{ds_id}").json()["status"] == "valid"
        )
        assert boot_one.get(f"/v1/jobs/{job_id}").json()["status"] == "queued"

    # Boot two: a fresh storage instance over the same paths, as a restarted
    # process would build it from configuration.
    monkeypatch.setattr(
        storage, "STORE", storage.FilesystemStorage(root=tmp_path / "objects")
    )
    with TestClient(main.app) as boot_two:
        dataset_record = boot_two.get(f"/v1/datasets/{ds_id}").json()
        assert dataset_record["id"] == ds_id
        assert dataset_record["status"] == "valid"
        assert dataset_record["report"]["row_count"] == 12
        # The counting phase's terminal state, read from the one definition.
        assert dataset_record["token_count_status"] == db.COUNT_PHASE_DONE

        job_record = boot_two.get(f"/v1/jobs/{job_id}").json()
        assert job_record["id"] == job_id
        assert job_record["dataset_id"] == ds_id
        # The row survived the restart; the job itself is surfaced as orphaned
        # rather than silently resumed -- the documented restart behaviour,
        # which a reviewer restarting mid-run would see.
        assert job_record["status"] == "failed"
        assert job_record["error_code"] == "orphaned_by_restart"

        # The stored object survived the restart too, byte for byte.
        key = storage.dataset_key(ds_id)
        assert storage.STORE.get(key) == dataset.read_bytes()


# --- whether real compute is used is a single setting ------------------------


def test_the_compute_switch_is_one_setting(monkeypatch):
    """Flipping TEMPER_FAKE_PROVIDER changes which provider answers, and the
    same call site routes both ways -- there is no second switch, and no other
    setting changes the mode (ADR-0024, ADR-0062)."""
    from temper_control_plane import config, provider
    from temper_control_plane.fake_provider import SimulatedMachine

    real = object()
    monkeypatch.setattr(provider, "JarvisLabsProvider", lambda: real)
    monkeypatch.setattr(config, "FAKE_PROVIDER", False)
    assert provider.new_provider() is real

    monkeypatch.setattr(config, "FAKE_PROVIDER", True)
    fake = provider.new_provider()
    assert isinstance(fake, SimulatedMachine)


def test_only_the_provider_field_differs_between_modes(client, monkeypatch):
    """The health surface is identical in both modes except the one field that
    says which mode is in force -- the observable form of 'nothing else
    differs between modes'."""
    from temper_control_plane import config

    monkeypatch.setattr(config, "FAKE_PROVIDER", False)
    real = client.get("/health").json()
    monkeypatch.setattr(config, "FAKE_PROVIDER", True)
    fake = client.get("/health").json()

    real_sans_provider = {k: v for k, v in real.items() if k != "provider"}
    fake_sans_provider = {k: v for k, v in fake.items() if k != "provider"}
    assert real_sans_provider == fake_sans_provider
    assert real["provider"] == "real"
    assert fake["provider"] == "fake"


def test_a_default_start_lands_in_the_safe_mode(monkeypatch):
    """With nothing set, the fault surface cannot be turned on by accident: a
    default process is the real tier, and a fault spec is refused there
    (ADR-0051). Pinned here so the boundary a sibling's fault surface relies
    on cannot drift."""
    from temper_control_plane import config

    bare = config.Settings()
    assert bare.fake_provider is False
    assert bare.fault_surface is False

    monkeypatch.setattr(config, "FAKE_PROVIDER", False)
    monkeypatch.setattr(config, "FAULT_SURFACE", False)
    refusal = config.fault_surface_refusal("oom")
    assert refusal is not None
    assert refusal["code"] == "fault_surface_refused"
