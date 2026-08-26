"""The storage seam, both implementations, and what a grant confers.

Spec 006 / issue #22. One interface holds every stored object: a filesystem
implementation for local runs and tests, and an object-store implementation
behind configuration. What is asserted here is the seam's external behaviour:

* bytes written through the seam read back unchanged, in both implementations
  -- including binary content and CRLF line endings, the exact corruption the
  transport tier once shipped;
* nothing outside the seam resolves a key to a location, so traversal-shaped
  keys are refused rather than escaped;
* a scoped write grant confers one write to one key and nothing else, and
  stops conferring it once expired;
* backend selection is configuration, with filesystem as the default when
  nothing is set.

The object-store tests run against an in-process S3 API mock (`moto`). That
proves wiring, key handling and byte fidelity through a real boto3 client; the
real-store tier -- MinIO through testcontainers, per the control plane's
testing rules -- is Phase B integration work and is deliberately not faked
here by pretending this is it.
"""

from __future__ import annotations

import base64
import socket
import time
import urllib.request
from urllib.parse import parse_qs, urlsplit

import boto3
import pytest
from botocore.config import Config as botocore_config
from moto import mock_aws
from moto.server import ThreadedMotoServer

from temper_control_plane import config, storage
from temper_control_plane.storage import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    FilesystemStorage,
    ObjectNotFound,
    S3Storage,
    WriteGrant,
    artifact_key,
    dataset_key,
)

# Binary content on purpose: NUL bytes and CRLF are what a naive transport
# quietly mangles, so they are the first thing a round-trip must survive.
PAYLOAD = b"\x00\x01binary\r\nwith lines\n\x00\xff"


@pytest.fixture()
def fs(tmp_path):
    return FilesystemStorage(root=tmp_path / "objects")


@pytest.fixture()
def s3():
    with mock_aws():
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="testing",  # noqa: S106 - moto's fixed dummy pair
            aws_secret_access_key="testing",  # noqa: S106 - never a credential
            config=botocore_config(signature_version="s3v4"),
        )
        client.create_bucket(Bucket="temper-test")
        yield S3Storage(bucket="temper-test", client=client)


# --- the round-trip both implementations must satisfy ------------------------


def test_filesystem_round_trip_is_byte_identical(fs):
    fs.put("datasets/ds_1.jsonl", PAYLOAD)
    assert fs.get("datasets/ds_1.jsonl") == PAYLOAD


def test_object_store_round_trip_is_byte_identical(s3):
    s3.put("datasets/ds_1.jsonl", PAYLOAD)
    assert s3.get("datasets/ds_1.jsonl") == PAYLOAD


def test_a_missing_object_is_a_typed_error_not_an_exception_leak(fs, s3):
    with pytest.raises(ObjectNotFound):
        fs.get("datasets/ds_missing.jsonl")
    with pytest.raises(ObjectNotFound):
        s3.get("datasets/ds_missing.jsonl")


def test_delete_removes_and_is_idempotent(fs, s3):
    for store in (fs, s3):
        store.put("artifacts/j_1/adapter_model.safetensors", b"w")
        store.delete("artifacts/j_1/adapter_model.safetensors")
        with pytest.raises(ObjectNotFound):
            store.get("artifacts/j_1/adapter_model.safetensors")
        # A second delete of an already-gone object is not an error: teardown
        # paths call it exactly once, but cancellation races can call twice.
        store.delete("artifacts/j_1/adapter_model.safetensors")


# --- keys are addresses, not locations ---------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "../escape.jsonl",
        "datasets/../../escape.jsonl",
        "/etc/passwd",
        r"datasets\ds_1.jsonl",
        "datasets//ds_1.jsonl",
        "",
    ],
)
def test_traversal_shaped_keys_are_refused(fs, key):
    """A key that could resolve outside the store is refused before any I/O.

    The filesystem implementation resolves keys to real paths internally, so
    this is where traversal dies; the check lives in the seam because a key
    that only the S3 backend would tolerate is a key the product does not have.
    """
    with pytest.raises(ValueError):
        fs.put(key, b"x")


# --- the scoped write grant ---------------------------------------------------


def test_a_filesystem_grant_confers_exactly_one_write_to_its_own_key(fs):
    grant = fs.mint_write_grant("artifacts/j_1/adapter_model.safetensors", 60)
    fs.redeem(grant, PAYLOAD)
    assert fs.get("artifacts/j_1/adapter_model.safetensors") == PAYLOAD

    other = WriteGrant(
        url=grant.url,
        key="datasets/ds_other.jsonl",
        expires_at=grant.expires_at,
    )
    with pytest.raises(storage.GrantInvalid):
        fs.redeem(other, b"x")


def test_an_expired_grant_confers_nothing(fs):
    """Expiry is enforced at redemption from the signed payload, not trusted
    to the holder's own claim about when it expires."""
    grant = fs.mint_write_grant("artifacts/j_1/w", 0.05)
    time.sleep(0.06)
    with pytest.raises(storage.GrantInvalid):
        fs.redeem(grant, b"x")
    with pytest.raises(ObjectNotFound):
        fs.get("artifacts/j_1/w")


def test_a_tampered_grant_is_refused(fs):
    grant = fs.mint_write_grant("artifacts/j_1/w", 60)
    payload, sig = grant.url.split(".", 1)
    forged = WriteGrant(
        url=f"{payload}.{base64.urlsafe_b64encode(b'forged').decode()}",
        key=grant.key,
        expires_at=grant.expires_at,
    )
    with pytest.raises(storage.GrantInvalid):
        fs.redeem(forged, b"x")


def test_an_object_store_grant_is_a_presigned_put_for_one_key(s3):
    ttl = 900
    grant = s3.mint_write_grant("artifacts/j_1/adapter_model.safetensors", ttl)
    parts = urlsplit(grant.url)
    assert parts.path.endswith("/artifacts/j_1/adapter_model.safetensors")
    query = parse_qs(parts.query)
    assert query["X-Amz-Expires"] == [str(ttl)]
    assert query["X-Amz-Signature"]
    assert grant.key == "artifacts/j_1/adapter_model.safetensors"
    assert abs(grant.expires_at - (time.time() + ttl)) < 5


def test_an_object_store_grant_actually_lands_bytes_when_put_over_http():
    """The grant is exercised the way a machine will use it (ADR-0009): minted
    by the store, PUT over real HTTP, then read back through the seam.

    Honest about what this proves: moto's server honours the request without
    cryptographically verifying the signature, so this pins the URL's shape,
    its scoping to one key, and the whole wiring working against a real HTTP
    endpoint -- it does not prove MinIO's enforcement. That tier is Phase B
    integration work.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = ThreadedMotoServer("127.0.0.1", port)
    server.start()
    try:
        client = boto3.client(
            "s3",
            endpoint_url=f"http://127.0.0.1:{port}",
            region_name="us-east-1",
            aws_access_key_id="testing",  # noqa: S106 - moto's fixed dummy pair
            aws_secret_access_key="testing",  # noqa: S106 - never a credential
            config=botocore_config(signature_version="s3v4"),
        )
        client.create_bucket(Bucket="temper-test")
        store = S3Storage(bucket="temper-test", client=client)

        grant = store.mint_write_grant("artifacts/j_9/w", 900)
        # The explicit content type is what any binary PUT carries anyway;
        # urllib's default (urlencoded) makes the receiving WSGI layer parse
        # the body as a form instead of storing it.
        request = urllib.request.Request(  # noqa: S310 - loopback moto server
            grant.url,
            data=PAYLOAD,
            method="PUT",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(request) as response:  # noqa: S310
            assert response.status == 200
        assert store.get("artifacts/j_9/w") == PAYLOAD
    finally:
        server.stop()


# --- key layout: defined once, read everywhere -------------------------------


def test_dataset_and_artifact_keys_have_one_definition():
    assert dataset_key("ds_abc123") == "datasets/ds_abc123.jsonl"
    assert (
        artifact_key("job_abc123", ADAPTER_WEIGHTS_NAME)
        == "artifacts/job_abc123/adapter_model.safetensors"
    )
    assert (
        artifact_key("job_abc123", ADAPTER_CONFIG_NAME)
        == "artifacts/job_abc123/adapter_config.json"
    )


# --- backend selection is configuration ---------------------------------------


def test_the_default_backend_is_the_filesystem(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "STORAGE_BACKEND", "filesystem")
    monkeypatch.setattr(config, "STORAGE_ROOT", tmp_path / "objects")
    store = storage.from_config()
    assert isinstance(store, FilesystemStorage)


def test_configuration_selects_the_object_store(monkeypatch):
    monkeypatch.setattr(config, "STORAGE_BACKEND", "s3")
    monkeypatch.setattr(config, "S3_BUCKET", "temper-prod")
    monkeypatch.setattr(config, "S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setattr(config, "S3_REGION", None)
    store = storage.from_config()
    assert isinstance(store, S3Storage)
    assert store.bucket == "temper-prod"


def test_an_unknown_backend_stops_the_process_rather_than_guessing(
    monkeypatch,
):
    """Same contract as every limit in config: a value that cannot be honoured
    refuses to boot. Falling back to a default would mean an operator who set
    TEMPER_STORAGE_BACKEND=s2 is running on a store nobody configured."""
    monkeypatch.setattr(config, "STORAGE_BACKEND", "s2")
    with pytest.raises(ValueError, match="TEMPER_STORAGE_BACKEND"):
        storage.from_config()


def test_an_s3_backend_without_a_bucket_is_refused(monkeypatch):
    monkeypatch.setattr(config, "STORAGE_BACKEND", "s3")
    monkeypatch.setattr(config, "S3_BUCKET", None)
    with pytest.raises(ValueError, match="TEMPER_S3_BUCKET"):
        storage.from_config()


def test_ensure_ready_prepares_the_backend(fs, tmp_path):
    fs.ensure_ready()
    assert (tmp_path / "objects").is_dir()


def test_the_module_singleton_follows_the_configured_backend():
    """The singleton exists so call sites read `storage.STORE` without knowing
    which backend answered -- which is the whole point of the seam."""
    assert isinstance(storage.STORE, FilesystemStorage)


# --- streamed reads and writes: the seam moves big objects too ----------------
#
# Spec 006's flat-memory clause, at the storage tier. `get` and `put` remain
# for small whole objects -- configs, result documents -- but datasets and
# artifacts cross as chunk streams, so the seam must stream in both
# directions. What is asserted is external: joined chunks equal the bytes,
# chunks arrive bounded, absence raises eagerly, and a failed streamed write
# publishes nothing.


def test_filesystem_get_stream_yields_the_bytes_in_bounded_chunks(fs):
    payload = PAYLOAD * 40000  # ~880 KiB: several chunks' worth
    fs.put("datasets/ds_1.jsonl", payload)

    stream = fs.get_stream("datasets/ds_1.jsonl")
    received = list(stream)

    assert b"".join(received) == payload
    assert len(received) > 1, "arrived as one blob, not a stream"
    assert max(len(c) for c in received) <= storage.STREAM_CHUNK_BYTES


def test_object_store_get_stream_yields_the_bytes_in_bounded_chunks(s3):
    payload = PAYLOAD * 40000
    s3.put("datasets/ds_1.jsonl", payload)

    received = list(s3.get_stream("datasets/ds_1.jsonl"))

    assert b"".join(received) == payload
    assert len(received) > 1


@pytest.mark.parametrize("store", ["fs", "s3"])
def test_get_stream_of_a_missing_object_raises_before_the_first_chunk(
    request, store
):
    """Eagerly, not at first `next`: a caller that must answer with a status
    code can only do so before it starts streaming."""
    backend = request.getfixturevalue(store)
    with pytest.raises(ObjectNotFound):
        backend.get_stream("datasets/ds_missing.jsonl")


def test_filesystem_put_stream_stores_whole_from_chunks_and_overwrites(fs):
    fs.put_stream("artifacts/j_1/w", iter([b"alpha", b"omega"]))
    assert fs.get("artifacts/j_1/w") == b"alphaomega"

    # Overwrite: an object at the key is replaced whole.
    fs.put_stream("artifacts/j_1/w", iter([b"replacement"]))
    assert fs.get("artifacts/j_1/w") == b"replacement"


def test_a_failed_filesystem_put_stream_publishes_nothing(fs):
    def broken():
        yield b"half"
        raise OSError("the source died mid-stream")

    with pytest.raises(OSError):
        fs.put_stream("artifacts/j_1/w", broken())

    with pytest.raises(ObjectNotFound):
        fs.get("artifacts/j_1/w")
    # And no staging leftover sits beside where the object would be.
    siblings = (
        [p.name for p in (fs._root / "artifacts" / "j_1").iterdir()]
        if (fs._root / "artifacts" / "j_1").is_dir()
        else []
    )
    assert not any(name.endswith(".part") for name in siblings)


def test_an_object_store_put_stream_round_trips_across_many_parts(
    s3, monkeypatch
):
    """The multipart path, exercised for real through boto3 against moto.

    Part size turned down so several parts are produced quickly; what is
    asserted is that an object spanning many parts reads back identical,
    which is exactly the property a part boundary could break."""
    monkeypatch.setattr(storage, "S3_PART_BYTES", 64 << 10)
    payload = PAYLOAD * 1024  # ~1 MiB across 16+ parts

    s3.put_stream(
        "artifacts/j_1/w",
        iter([payload[i : i + 9973] for i in range(0, len(payload), 9973)]),
    )

    assert s3.get("artifacts/j_1/w") == payload


def test_an_object_store_put_stream_below_one_part_is_a_single_put(
    s3, monkeypatch
):
    monkeypatch.setattr(storage, "S3_PART_BYTES", 64 << 10)
    s3.put_stream("artifacts/j_1/w", iter([b"small"]))
    assert s3.get("artifacts/j_1/w") == b"small"


def test_a_failed_object_store_put_stream_aborts_cleanly(s3):
    def broken():
        yield b"x" * (storage.S3_PART_BYTES + 1)
        raise OSError("the source died mid-upload")

    with pytest.raises(OSError):
        s3.put_stream("artifacts/j_1/w", broken())

    with pytest.raises(ObjectNotFound):
        s3.get("artifacts/j_1/w")
