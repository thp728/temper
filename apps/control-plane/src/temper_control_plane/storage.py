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
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from botocore.exceptions import ClientError

from . import config

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
