"""The one storage seam: every stored object crosses this module or nowhere.

Spec 006 / issue #22. Datasets and artifacts used to be paths under whichever
directory the process happened to be sitting in -- unreachable by a second
process, unscopeable, and tied to one machine's disk. Now they are **objects
addressed by key**, and exactly two implementations know what a key resolves
to:

* `FilesystemStorage` -- keys under one root directory. The default when
  nothing is configured, because a fresh clone must run with no configuration
  at all.
* `S3Storage` -- any S3-compatible store behind configuration, MinIO included,
  which is the store ADR-0009's pre-signed-URL flow assumes.

Callers outside this module hold **keys** (`datasets/{id}.jsonl`,
`artifacts/{job}/{name}`) and nothing else. They never learn that the
filesystem backend exists, never construct a path, and survive the move from a
laptop directory to a bucket unchanged. That is the acceptance criterion --
"no code outside the seam refers to a filesystem path for a stored object" --
and it holds because there is nothing left outside the seam that could.

**Write grants.** `mint_write_grant` produces the credential ADR-0009 hands to
a machine: authority for *one write to one key*, expiring. The S3 backend
mints a real pre-signed PUT URL the machine can use directly. The filesystem
backend mints an HMAC-signed token honoured by presenting it back to the same
process (`redeem`) -- honest about local reality, where there is no second
storage process to talk to. Nothing can list, read, enumerate or reach any
other key by holding a grant; expiry is checked on redemption, not trusted to
the holder.

Testing note, recorded rather than glossed: the S3 tests here run against an
in-process S3 API mock (`moto`). That proves wiring, key handling and byte
fidelity through a real boto3 client; it does **not** prove MinIO itself. The
real-store tier is Phase B integration work (testcontainers), same rule as the
database tickets: the ticket whose subject is *which* store gets the real one.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from botocore.exceptions import ClientError

from . import config

# How much one streamed step reads or writes. Bounded, not tuned: the
# flat-memory tests assert peak held memory does not scale with object size,
# not that a particular chunk size was used.
STREAM_CHUNK_BYTES = 256 * 1024

# The multipart part size for streamed uploads to the object store. S3's own
# floor is 5 MiB per non-final part; 8 MiB sits above it with headroom, and a
# test turns it down to keep the multipart path fast.
S3_PART_BYTES = 8 << 20

# The two names an artifact consists of. Defined once here because the
# orchestrator writes them and the download endpoint reads them: a value two
# components agree on is defined once and read, never retyped.
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
ARTIFACT_MEMBERS = (ADAPTER_WEIGHTS_NAME, ADAPTER_CONFIG_NAME)


def dataset_key(ds_id: str) -> str:
    """The key a dataset is stored under, derived from its id."""
    return f"datasets/{ds_id}.jsonl"


def artifact_key(job_id: str, name: str) -> str:
    """The key one file of one job's artifact is stored under."""
    return f"artifacts/{job_id}/{name}"


def artifact_config_key(weights_key: str) -> str:
    """The config object that travels beside one artifact weights object.

    An artifact is weights plus the config that makes them loadable; the
    download endpoint holds the weights' key (from the job row) and derives
    the config's address from it, so the pair cannot drift apart.
    """
    parent, _, _ = _checked(weights_key).rpartition("/")
    return f"{parent}/{ADAPTER_CONFIG_NAME}"


def checkpoint_key(job_id: str, slot: int) -> str:
    """The key one checkpoint slot of one job is stored under.

    Checkpoints are stored as **slots**, not as one object per step: retention
    is a bounded, configurable number of checkpoints per job (issue #37), so
    the machine overwrites the oldest slot with each new checkpoint and
    storage never holds more than `CHECKPOINT_RETENTION` objects per job. The
    slot number, not the step, is part of the key, because the step is data
    the trainer reports and the control plane verifies -- the key must be
    knowable at mint time, before the run has produced a single step.
    """
    return f"checkpoints/{job_id}/slot-{slot}"


def checkpoint_keys(job_id: str, count: int) -> list[str]:
    """The `count` slot keys a job's checkpoints may be written to, in order.

    Defined once here because the orchestrator mints a write grant per slot
    and the cancellation path deletes every slot: the two must agree on how
    many slots a job has and what they are called.
    """
    if count < 0:
        raise ValueError("checkpoint slot count cannot be negative")
    return [checkpoint_key(job_id, slot) for slot in range(count)]


def delivery_key(job_id: str, format_id: str, name: str) -> str:
    """The key one produced delivery format's file is stored under (issue #74).

    Delivery formats are distinct objects from the canonical artifact, stored
    under `artifacts/{job}/{format}/{name}` so the machine writes each through
    its own one-key grant and the download path reads each by the job row's own
    record -- never by re-deriving an address.
    """
    return artifact_key(job_id, f"{format_id}/{name}")


class ObjectNotFound(KeyError):
    """No object lives at this key. Raised by `get`, never by `delete`."""

    def __init__(self, key: str):
        super().__init__(key)
        self.key = key


@dataclass(frozen=True)
class WriteGrant:
    """Authority for one write to one key, expiring at `expires_at`.

    `url` is what crosses to the machine, opaque to its holder: a pre-signed
    HTTPS URL on the object-store backend, a signed local token on the
    filesystem backend. Holding it confers nothing but its one write.
    """

    url: str
    key: str
    expires_at: float


class GrantInvalid(Exception):
    """A presented write grant does not confer what it claims to."""


class Storage(Protocol):
    """What every backend promises. This protocol *is* the seam."""

    def put(self, key: str, data: bytes) -> None:
        """Store one object whole. Overwrites an existing object at the key."""
        ...

    def get(self, key: str) -> bytes:
        """Read one object back, unchanged. Raises ObjectNotFound."""
        ...

    def get_stream(self, key: str) -> Iterator[bytes]:
        """Yield the object's bytes in bounded chunks.

        For payloads that grow without bound -- a dataset travelling to a
        machine, an artifact travelling to a browser -- so that moving an
        object never requires holding it. Raises ObjectNotFound like `get`,
        eagerly: before the first chunk is due, so a caller can still turn
        absence into a status code rather than a mid-stream failure.
        """
        ...

    def put_stream(self, key: str, chunks: Iterable[bytes]) -> None:
        """Store one object from chunks, holding at most one at a time.

        Overwrites an existing object at the key, and publishes whole: no
        reader ever observes a half-written object under `key`. If the chunk
        source fails part-way, nothing is published and what was staged is
        cleaned up.
        """
        ...

    def delete(self, key: str) -> None:
        """Remove one object if present; absence is not an error."""
        ...

    def mint_write_grant(self, key: str, expires_in_s: float) -> WriteGrant:
        """Mint one-write-to-one-key authority, expiring in `expires_in_s`."""
        ...

    def ensure_ready(self) -> None:
        """Prepare the backend, failing loudly at boot rather than mid-run."""
        ...


def _checked(key: str) -> str:
    """Refuse any key that is not a plain slash-separated address.

    Backslashes, empty segments, dot segments and absolute forms are refused
    before any I/O: they are how a key stops being an address and becomes a
    place somewhere else. Both implementations run every key through this.
    """
    if (
        not key
        or key.startswith("/")
        or "\\" in key
        or ".." in key.split("/")
        or any(not segment for segment in key.split("/"))
    ):
        raise ValueError(
            f"{key!r} is not a valid object key. Keys are relative, "
            f"slash-separated, and contain no empty or dot segments."
        )
    return key


def _grant_ttl(expires_in_s: float) -> int:
    if expires_in_s <= 0:
        raise ValueError(
            "expires_in_s must be greater than zero. A grant that cannot "
            "expire is not a scoped grant."
        )
    return int(expires_in_s)


class FilesystemStorage:
    """Keys under one root directory. The default backend."""

    scheme = "temper-local"

    def __init__(self, root: Path, secret: bytes | None = None):
        self._root = root
        self._secret = secret or secrets.token_bytes(32)

    def _resolve(self, key: str) -> Path:
        resolved = (self._root / _checked(key)).resolve()
        root_resolved = self._root.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ValueError(f"{key!r} escapes the storage root")
        return resolved

    def put(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        try:
            return self._resolve(key).read_bytes()
        except FileNotFoundError:
            raise ObjectNotFound(key) from None

    def get_stream(self, key: str) -> Iterator[bytes]:
        # Opened here rather than inside the generator so that a missing
        # object raises now, while the caller can still answer with a status
        # code, and not at the first `next`.
        try:
            source = open(self._resolve(key), "rb")
        except FileNotFoundError:
            raise ObjectNotFound(key) from None
        return self._chunks(source)

    @staticmethod
    def _chunks(source) -> Iterator[bytes]:
        try:
            while chunk := source.read(STREAM_CHUNK_BYTES):
                yield chunk
        finally:
            source.close()

    def put_stream(self, key: str, chunks: Iterable[bytes]) -> None:
        final = self._resolve(key)
        final.parent.mkdir(parents=True, exist_ok=True)
        # Staged under a sibling name and moved into place only once every
        # chunk has arrived, which is what makes the publish whole: a reader
        # at `key` sees either the previous object or the complete new one,
        # never a partial write. The same rule cleans up after a chunk
        # source that dies part-way.
        pending = final.with_name(final.name + ".part")
        try:
            with open(pending, "wb") as out:
                for chunk in chunks:
                    out.write(chunk)
            os.replace(pending, final)
        except BaseException:
            pending.unlink(missing_ok=True)
            raise

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)

    def mint_write_grant(self, key: str, expires_in_s: float) -> WriteGrant:
        """Sign key-and-expiry with HMAC-SHA256 under this store's secret.

        Unset TEMPER_STORAGE_SECRET means a per-process random secret, so
        local grants do not survive a restart. That is correct rather than
        convenient: a grant's whole lifetime is one job inside one process.
        """
        expires_at = time.time() + _grant_ttl(expires_in_s)
        payload = json.dumps({"key": key, "exp": expires_at}).encode()
        sig = hmac.new(self._secret, payload, hashlib.sha256).digest()
        token = (
            base64.urlsafe_b64encode(payload).decode().rstrip("=")
            + "."
            + base64.urlsafe_b64encode(sig).decode().rstrip("=")
        )
        return WriteGrant(
            url=f"{self.scheme}:{token}", key=key, expires_at=expires_at
        )

    def redeem(self, grant: WriteGrant, data: bytes) -> None:
        """Honour a minted grant, or refuse it loudly.

        Every clause is checked against the signature, not the envelope: the
        token matches this store's secret, it has not expired, and the key it
        signed is the key being claimed. Signature comparison is constant-time.
        """
        prefix = f"{self.scheme}:"
        if not grant.url.startswith(prefix):
            raise GrantInvalid("not a filesystem-backend grant")

        def unpadded(part: str) -> bytes:
            return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))

        payload_b64, _, sig_b64 = grant.url[len(prefix) :].partition(".")
        payload = unpadded(payload_b64)
        expected = hmac.new(self._secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, unpadded(sig_b64)):
            raise GrantInvalid("signature does not match")
        try:
            signed = json.loads(payload)
        except ValueError as e:
            raise GrantInvalid("malformed grant") from e
        if time.time() > signed["exp"]:
            raise GrantInvalid("grant has expired")
        if signed["key"] != grant.key:
            raise GrantInvalid("grant was minted for a different key")
        self.put(grant.key, data)

    def ensure_ready(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)


class S3Storage:
    """Keys in one S3-compatible bucket. Selected by configuration.

    Speaks the S3 API through boto3, so it faces AWS S3, MinIO, or anything
    else compatible. Credentials come from boto3's standard chain; the client
    is injectable because the tests build one against an in-process mock and
    constructing a real one here would couple the seam to the network.
    """

    def __init__(
        self,
        bucket: str,
        client=None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ):
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        # s3v4 pinned: boto3's default presigning emits a Signature-V2 query
        # string, which MinIO -- the store this backend exists for -- refuses.
        self._client = client or boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            config=Config(signature_version="s3v4"),
        )

    def put(self, key: str, data: bytes) -> None:
        self._client.put_object(
            Bucket=self.bucket, Key=_checked(key), Body=data
        )

    def get(self, key: str) -> bytes:
        try:
            response = self._client.get_object(
                Bucket=self.bucket, Key=_checked(key)
            )
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise ObjectNotFound(key) from None
            raise
        return response["Body"].read()

    def get_stream(self, key: str) -> Iterator[bytes]:
        try:
            response = self._client.get_object(
                Bucket=self.bucket, Key=_checked(key)
            )
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise ObjectNotFound(key) from None
            raise
        # The SDK's streaming body yields chunks as they arrive over the
        # wire; nothing here reads the object whole and slices it.
        return response["Body"].iter_chunks(STREAM_CHUNK_BYTES)

    def put_stream(self, key: str, chunks: Iterable[bytes]) -> None:
        _checked(key)
        # A manual multipart upload: parts are accumulated to S3_PART_BYTES
        # -- the last may be smaller, S3 allows exactly one such -- and each
        # is handed to the store as it fills, so peak memory is one part
        # however large the object is. boto3's transfer manager would also
        # do this, but it demands a seekable file-like; an iterator of
        # chunks is not one, and faking seekability would mean buffering.
        part_number = 0
        parts: list[dict] = []
        buffer = bytearray()
        upload_id: str | None = None
        try:
            for chunk in chunks:
                buffer.extend(chunk)
                while len(buffer) >= S3_PART_BYTES:
                    part_number += 1
                    if upload_id is None:
                        upload_id = self._begin_upload(key)
                    parts.append(
                        self._upload_part(
                            key,
                            upload_id,
                            part_number,
                            bytes(buffer[:S3_PART_BYTES]),
                        )
                    )
                    del buffer[:S3_PART_BYTES]
            if upload_id is None:
                # Never crossed a part boundary: a small object is one PUT,
                # the same shape `put` takes, rather than a one-part
                # multipart ceremony.
                self._client.put_object(
                    Bucket=self.bucket, Key=key, Body=bytes(buffer)
                )
                return
            part_number += 1
            parts.append(
                self._upload_part(key, upload_id, part_number, bytes(buffer))
            )
            self._client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={
                    "Parts": [
                        {"ETag": p["ETag"], "PartNumber": p["PartNumber"]}
                        for p in parts
                    ]
                },
            )
        except BaseException:
            # An abandoned multipart upload leaves billed, invisible parts
            # behind unless it is aborted; that cleanup must happen even
            # while the original failure propagates.
            if upload_id is not None:
                self._client.abort_multipart_upload(
                    Bucket=self.bucket, Key=key, UploadId=upload_id
                )
            raise

    def _begin_upload(self, key: str) -> str:
        response = self._client.create_multipart_upload(
            Bucket=self.bucket, Key=key
        )
        return response["UploadId"]

    def _upload_part(
        self, key: str, upload_id: str, part_number: int, body: bytes
    ) -> dict:
        response = self._client.upload_part(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
            PartNumber=part_number,
            Body=body,
        )
        return {"ETag": response["ETag"], "PartNumber": part_number}

    def delete(self, key: str) -> None:
        # S3 delete of an absent key succeeds by design; idempotency is free.
        self._client.delete_object(Bucket=self.bucket, Key=_checked(key))

    def mint_write_grant(self, key: str, expires_in_s: float) -> WriteGrant:
        _checked(key)
        ttl = _grant_ttl(expires_in_s)
        url = self._client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=ttl,
        )
        return WriteGrant(url=url, key=key, expires_at=time.time() + ttl)

    def ensure_ready(self) -> None:
        # Fail at boot, where it is one legible line, rather than four seconds
        # into someone's first job. Same reasoning as the credential check.
        self._client.head_bucket(Bucket=self.bucket)


def from_config() -> Storage:
    """Build whichever backend configuration selects.

    An unknown backend or an unusable combination refuses here rather than
    falling back: an operator who set TEMPER_STORAGE_BACKEND=s2 must not find
    themselves on a store nobody configured.
    """
    backend = config.STORAGE_BACKEND
    if backend == "s3":
        if not config.S3_BUCKET:
            raise ValueError(
                "TEMPER_STORAGE_BACKEND=s3 requires TEMPER_S3_BUCKET. "
                "Refusing to guess a bucket for other people's data."
            )
        return S3Storage(
            bucket=config.S3_BUCKET,
            endpoint_url=config.S3_ENDPOINT_URL,
            region_name=config.S3_REGION,
        )
    if backend == "filesystem":
        return FilesystemStorage(
            root=config.STORAGE_ROOT,
            secret=(
                config.STORAGE_SECRET.encode()
                if config.STORAGE_SECRET
                else None
            ),
        )
    raise ValueError(
        f"TEMPER_STORAGE_BACKEND={backend!r} is not a known backend. "
        f"It selects where stored objects live, so it is refused rather "
        f"than ignored. Known backends: filesystem, s3."
    )


# Read once at import, like every other piece of process configuration. Call
# sites read `storage.STORE` and cannot know -- or care -- which class answers.
STORE: Storage = from_config()
