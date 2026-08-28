"""Job orchestration: provision, bootstrap, train, collect, destroy.

This is `spike/spike4.py` made durable. The sequence is unchanged because it is
proven; what is added is a job row, state transitions, and events.

Everything that touches the compute provider goes through the injected
`Provider` protocol — one seam, so the whole money-spending path can be
exercised with a fake and no GPU. The default is the real one, so callers that
do not care about testing pass nothing.

Five properties of this path, three of them carried over from the spikes and
each learned the expensive way:

* **Teardown runs in `finally`, then is independently confirmed** by listing
  machines. Trusting a destroy call's return value is exactly the assumption
  that leaves a GPU billing overnight. It now runs **before** the terminal
  state transition, so a client that stops polling once the job says it is
  finished still sees the confirmation.
* **Readiness distinguishes *unreachable* from *authentication failed*.** They
  have opposite remedies, and collapsing both into "no answer" cost an evening
  and produced a wrongly-filed platform bug. That logic lives in the default
  provider now, with the two codes intact.
* **A job that stops making progress stops itself.** Two limits, in
  `limits.py`: silence beyond the stall timeout and elapsed time beyond the
  duration ceiling. They are circuit breakers against a wedged job on a billing
  machine, not a cap on what a user may legitimately train, and they replace a
  single wall-clock constant that no code path ever read — a control that looks
  implemented and is not is worse than none at all.
* **Cancellation is destructive, and is checked where it can be honoured.**
  The user's request sets a flag on the job row; this path reads it at every
  boundary between stages and on every trip round the streaming loop, then
  destroys the machine and produces no adapter. Handing back a half-trained
  adapter would invite the user to mistake it for a finished model. The
  outcome is `cancelled`, never `failed` -- a decision is not a defect.
* **The trainer publishes no ports.** `ufw` does not filter Docker-published
  ports and a `DOCKER-USER` rule on the published port never matches, because
  the packet is already DNAT'd. Not publishing is the mitigation that works.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from typing import BinaryIO, NamedTuple

from temper_core import (
    actuals,
    artifacts,
    catalog,
    checkpoint,
    disk,
    events,
    hyperparams,
    overrides,
    selection,
)
from temper_core import faults as fault_surface
from temper_core.errors import Cancelled, OrchestratorError
from temper_core.models import Models

from . import config, db, storage
from .chunks import ChunkReader, piped_chunks
from .limits import RunLimits, guard
from .models import new_models
from .provider import Provider, new_provider
from .trainer_build import published_reference

DATASET_TARBALL = "/tmp/dataset.tar.gz"
# The reference the script carries when the simulated provider is in use
# (TEMPER_FAKE_PROVIDER) and no image has been published. The simulated
# machine executes no container, so this is only ever a well-formed stand-in
# to keep the script shaped like the real one; it is deliberately not a
# registry reference so it cannot be mistaken for one.
_SIMULATED_REFERENCE = "temper-simulated:local"
# The machine emits this when the image cannot be pulled, so that a pre-training
# failure still arrives as a result document naming its own code rather than as
# "the trainer produced nothing". An image that could not be pulled is not a
# training failure, and telling a user otherwise sends them to read the wrong
# output. Single quotes are forbidden in these strings -- the script echoes
# them inside a single-quoted shell literal.
PULL_FAILED = json.dumps(
    {
        "stage": "pull",
        "ok": False,
        "error_code": "image_pull_failed",
        "error": "The trainer image could not be pulled; the pull output is "
        "in the job events.",
    }
)
NO_RESULT = json.dumps(
    {
        "stage": "train",
        "ok": False,
        "error_code": "trainer_no_result",
        "error": "The trainer produced no result.json; its output is in the job "
        "events.",
    }
)
RESULT_MARKER = "---RESULT---"
DESTROY_ATTEMPTS = 3
DESTROY_RETRY_DELAY_S = 5

# The environment variable the trainer reads its fault from (issue #24).
# Trainer-side faults are an environment switch, by the spec's constraint; the
# name is defined once in `temper_core.faults` and pinned equal to the
# trainer's own constant by a trainer test, so the two sides cannot drift.
FAULT_ENV = fault_surface.FAULT_ENV

# Cancellation says the same thing twice, at the two moments a user is
# listening. The acknowledgement is one string used both as the API's answer
# and as the event recorded against the job, so what the button said and what
# the run's own history says cannot drift apart.
#
# It promises nothing it has not done: the machine is *being* destroyed, not
# destroyed, because the destroy call can fail and the stray-machine path is
# real. The confirmation is a separate event, and it arrives before the job
# reports that it is over.
CANCEL_ACK = (
    "Cancellation requested. The machine is being destroyed and no artifact "
    "will be produced."
)
CANCEL_MESSAGE = "Cancelled at your request. No artifact was produced."


class _TarMember(NamedTuple):
    """One archive entry, with a lazy byte source.

    A chunk reader (a storage object mid-stream) is pulled through block by
    block at pour time, so nothing is held between describing the members
    and the transfer actually running; raw `bytes` -- the normalised
    sources, small and bounded by the repo itself -- travel in their own
    buffer.
    """

    name: str
    size: int
    source: bytes | ChunkReader


def _pour_tar(members: list[_TarMember], sink: BinaryIO) -> None:
    """Write the gzipped tar of `members` to `sink`, streaming throughout.

    Stream mode (`w|`) writes without seeking, which is what makes a pipe
    possible; addfile copies each member through in blocks, and the one at a
    time rule means peak cost is a block of the largest member plus the gzip
    window -- never the archive, never the dataset.
    """
    with tarfile.open(fileobj=sink, mode="w|gz") as tar:
        for member in members:
            info = tarfile.TarInfo(name=member.name)
            info.size, info.mode = member.size, 0o644
            if isinstance(member.source, bytes):
                tar.addfile(info, io.BytesIO(member.source))
            else:
                # A reader over a streamed object: addfile pulls exactly
                # `size` bytes through it, in blocks, and never holds the
                # member whole.
                tar.addfile(info, member.source)


def _trainer_reference() -> str:
    """The published trainer image the machine pulls, image@digest.

    The pipeline builds and publishes the image from the same named sources
    `just image` builds from (issue #44), and this side reads the digest out
    of the checked-in contract. The machine never builds: it pulls exactly the
    image the pipeline built and verified, so what runs is exactly what was
    built.

    A digest that has not been published is refused -- before any machine is
    provisioned -- unless the simulated provider is in use, which pulls no
    image and therefore needs no published one. That exception exists for the
    browser journeys (ADR-0024), which boot this process with
    TEMPER_FAKE_PROVIDER and drive a launch to completion on a machine that
    executes no container.
    """
    reference = published_reference()
    if reference is not None:
        return reference
    if config.FAKE_PROVIDER:
        return _SIMULATED_REFERENCE
    raise OrchestratorError(
        "image_not_published",
        "The trainer image has not been published yet, so a real job "
        "cannot launch. Run the pipeline's publish workflow "
        "(`.github/workflows/image.yml`) and merge the pull request it "
        "opens, which lands the digest this launch would pull.",
    )


def _dataset_chunks(dataset_object_key: str) -> Iterator[bytes]:
    """The dataset as one streamed tar.gz of its raw bytes.

    The same transport as the sources above, with one deliberate difference:
    **no line-ending normalisation.** Rewriting CRLF is correct for a shell
    script and a Dockerfile and wrong for user data, which must arrive
    byte-identical -- a dataset silently edited in transit is a bug that
    looks like anything else.

    The member streams out of the storage seam by key: no code on this side
    ever holds the whole payload. One honest cost is paid for that -- a tar
    header states the member's size up front and the seam exposes streams,
    not sizes, so the object is read twice: once to count, once to travel.
    Both passes hold one chunk, and on the default backend the second read
    lands in the page cache.
    """
    total = sum(
        len(chunk) for chunk in storage.STORE.get_stream(dataset_object_key)
    )
    member = _TarMember(
        "dataset.jsonl",
        total,
        ChunkReader(storage.STORE.get_stream(dataset_object_key)),
    )
    return piped_chunks(lambda sink: _pour_tar([member], sink))


def _remote_script(
    job: dict,
    model: catalog.BaseModel,
    enable_thinking: bool,
    reference: str,
    artifact_grant: storage.WriteGrant | None = None,
    checkpoint_grants: list[storage.WriteGrant] | None = None,
) -> bytes:
    """The on-machine script: pull the published image, run the job, report.

    The image is not built here and never was expected to be: the pipeline
    builds it from the same named sources `just image` builds from (issue #44),
    and the machine pulls the published image by digest (`reference`), so what
    runs is exactly what the pipeline built and verified. Nothing installs
    into the image at run time.

    Nothing here carries the dataset. It arrived ahead of this script as its
    own archive on standard input, so the script stays a fixed few kilobytes
    however large the dataset is -- it is never doubled into hex text, held in
    memory several times over, or fed through a pipe sized for commands.

    When `artifact_grant` is supplied, the scoped write URL (ADR-0009) rides
    into the job spec: the machine writes its own artifact to it and holds no
    credential that outlives the job. The URL is opaque to the machine -- it is
    already an authorisation, scoped to one key, expiring with the job.

    When `checkpoint_grants` is supplied, the same scoped write URLs (issue
    #37) ride into the job spec: one per checkpoint slot, each scoped to
    exactly one `checkpoints/...` key, each expiring with the job. The machine
    writes each checkpoint as it is produced to the next slot, so checkpoints
    leave the machine during training rather than only at its end.

    Nothing here is redirected to a file. That was the outermost of three
    redirections between the training framework and the user, and while any one
    of them stood the others bought nothing: output written to `/tmp/run.log`
    reaches the control plane when the run ends, which is the silence this
    channel exists to remove — and the machine is destroyed immediately after,
    taking the file with it.

    The container's output goes to **stderr**, which the provider folds into the
    same ordered stream. Two reasons: stdout carries the result protocol, so a
    training line that happened to equal the result marker could otherwise
    corrupt the document the orchestrator parses; and the script's own narration
    already goes there, so phase markers and the output they bracket stay in
    order.
    """
    revision = job.get("base_revision") or model.revision
    # Resolved here, before launch, and written whole (#83): the trainer
    # resolves nothing, so this is the one place a value is chosen and the
    # only copy the machine ever sees. What lands in the job record's
    # `hyperparameters` field stays the user's request; what reaches the
    # trainer is the resolver's answer to it.
    job_spec = {
        "job_id": job["id"],
        "base_model": model.repo,
        "base_revision": revision,
        "hyperparameters": hyperparams.effective(job["hyperparameters"] or {}),
    }
    if artifact_grant is not None:
        job_spec["artifact_upload"] = {
            "url": artifact_grant.url,
            "key": artifact_grant.key,
            "expires_at": artifact_grant.expires_at,
        }
    if checkpoint_grants:
        job_spec["checkpoint_grants"] = [
            {
                "url": grant.url,
                "key": grant.key,
                "expires_at": grant.expires_at,
            }
            for grant in checkpoint_grants
        ]
    # Issue #24: a fault spec only reaches the trainer's environment when the
    # surface is switched on -- the fake provider (TEMPER_FAKE_PROVIDER) or
    # the deliberate operator tier (TEMPER_FAULT_SURFACE). Otherwise the spec
    # travels in the job spec where the simulated machine honours it, and no
    # fault can ever fire against a real machine by accident. Single quotes
    # are escaped for the shell literal the spec rides in.
    fault_env = ""
    fault_spec = fault_surface.from_hyperparameters(job.get("hyperparameters"))
    if fault_spec is not None and (
        config.FAKE_PROVIDER or config.FAULT_SURFACE
    ):
        payload = json.dumps(fault_spec).replace("'", "'\\''")
        fault_env = f"  -e {FAULT_ENV}='{payload}'"
    script = f"""
set -u
say() {{ echo "[$(date +%H:%M:%S)] $*" >&2; }}

# The trainer needs no inbound port, so it publishes none. That is the only
# firewall mitigation that actually holds for Docker on this platform.
sudo ufw allow 22/tcp >/dev/null 2>&1
sudo ufw default deny incoming >/dev/null 2>&1
sudo ufw --force enable >/dev/null 2>&1

mkdir -p /tmp/job /tmp/out
tar xzf {DATASET_TARBALL} -C /tmp/job
cat > /tmp/job/job.json <<'JOBSPEC'
{json.dumps(job_spec, indent=2)}
JOBSPEC

say "pulling trainer image"
t0=$(date +%s)
# The digest is the contract and the tag is a comment (issue #44): the
# pipeline published exactly this image and verified it is pullable, so the
# machine pulls by digest and never builds. The pull's own output is the
# record of what reached the machine.
sudo docker pull {reference} 1>&2 || {{
  say "PULL FAILED"
  echo "{RESULT_MARKER}"
  echo '{PULL_FAILED}'
  exit 0
}}
say "image pulled in $(( $(date +%s) - t0 ))s"

say "running training"
# PYTHONUNBUFFERED is the innermost of the three redirections. The trainer and
# the framework beneath it are both Python, and Python buffers its stdout
# whenever it is a pipe rather than a terminal -- so without this the container
# holds minutes of output and the two layers outside it relay nothing.
# `fault_env` (issue #24) carries the job's fault spec into the trainer's
# environment when the surface is switched on, and nothing otherwise: a fault
# the surface could not have switched on never reaches the trainer.
sudo docker run --rm --gpus all \\
  -v /tmp/job:/job:ro -v /tmp/out:/out -e HF_HOME=/out/hf \\
  -e PYTHONUNBUFFERED=1 \\
{fault_env}  {reference} 1>&2 || say "TRAINER EXITED NONZERO"

if [ -f /tmp/out/result.json ]; then
  echo "{RESULT_MARKER}"
  sudo cat /tmp/out/result.json
else
  echo "{RESULT_MARKER}"
  echo '{NO_RESULT}'
fi
"""
    return script.encode("utf-8")


def _consume(job_id: str, lines) -> dict:
    """Turn the machine's output into events, and return the trainer's result.

    Everything before the marker is the job's output and is classified: a line
    carrying step, loss or epoch becomes a `metric` event with those numbers in
    structured fields, and everything else becomes a `log` event. Classifying
    here rather than when the log is read is what makes a chart possible later
    without re-parsing prose — by then the job is over and the format the line
    was written in is whatever the framework happened to use that day.

    Everything after the marker is the trainer's result document, which is
    machinery rather than output and is not logged as such.

    `lines` is consumed lazily and each event is written the moment its line is
    read, which is what stamps it with the time it actually happened. Reading
    the stream into a list first would put every event back on the same
    timestamp -- which is exactly what the log looked like before.
    """
    result_lines: list[str] = []
    seen_marker = False
    for line in lines:
        text = line.strip()
        if not seen_marker:
            if text == RESULT_MARKER:
                seen_marker = True
            elif text:
                # Classify the whole line, then truncate what is stored.
                # Truncating first would cut a long log dict short of its
                # closing brace, and a metric would be dropped for being a
                # partial line that the transport, not the framework, cut.
                event = events.classify(text)
                db.add_event(
                    job_id, event.kind, event.message[:500], event.data
                )
        else:
            result_lines.append(line)

    if not seen_marker:
        raise OrchestratorError(
            "training_failed",
            "Trainer produced no result.json. See job events.",
        )
    try:
        return json.loads("\n".join(result_lines))
    except json.JSONDecodeError as e:
        raise OrchestratorError(
            "training_failed", f"Trainer's result document did not parse: {e}"
        ) from e


def _delete_stored_artifact(job_id: str) -> None:
    """Remove the objects one job's artifact consists of, if any.

    Which objects those are comes from the same resolution the download path
    reads (`db.artifact_members`): the record's member keys, whatever kind
    they are -- so cancellation discards a full-model artifact's objects
    exactly as it discards an adapter's. When the row records nothing yet
    (the machine writes through the grant before `artifact_key` is set, so a
    cancellation mid-write or a verification refusal can land first), the
    canonical adapter pair is still deleted: it is the only set a QLoRA
    machine could have written, and deleting an absent key is a no-op.
    Deletion failures are suppressed deliberately: teardown must not mask the
    cancellation that caused them, and an orphaned object is cheaper than a
    half-reported state. (An object written by a machine whose run has since
    ended is an orphan in the same sense a stray machine is; ADR-0009 records
    that nothing reconciles them yet.)
    """
    job = db.get_job(job_id) or {}
    members = db.artifact_members(job) or [
        (name, storage.artifact_key(job_id, name))
        for name in artifacts.ADAPTER_MEMBER_NAMES
    ]
    for _, key in members:
        with suppress(Exception):
            storage.STORE.delete(key)


def _delete_stored_checkpoints(job_id: str) -> None:
    """Remove every checkpoint slot one job may have written to, if any.

    Called on the same cancellation paths as `_delete_stored_artifact`: a job
    the user stopped keeps no recovery material either, for the same reason it
    keeps no adapter. The slot count and names are read from the same
    definition the grants were minted from, so what this deletes is exactly
    what the machine could have written.
    """
    for key in storage.checkpoint_keys(job_id, config.CHECKPOINT_RETENTION):
        with suppress(Exception):
            storage.STORE.delete(key)


def _collect_artifact(
    job_id: str, result: dict, method: str | None
) -> dict | None:
    """Verify the artifact the machine wrote, and publish its config.

    Returns the artifact record -- the kind it declares, the stored members
    that make it up, and the verification's outcome -- or None when the
    machine produced nothing to store. This is where the artifact record is
    assembled (ADR-0035): the machine wrote the weights itself to the scoped
    grant (ADR-0009); this side verifies what landed against the checksum the
    machine reported, **before the job may report success**, and never holds
    the payload: the stored object is streamed through the hash one chunk at a
    time.

    The kind is derived from the job's method (issue #32), never stored: a
    QLoRA or LoRA run produces an adapter, a full fine-tune produces a fully
    trained model, and the derivation reads existing rows correctly with no
    migration. The members are recorded because they are what the download
    path serves -- any kind, without special-casing.

    The config that makes the weights loadable still travels inside result.json
    -- it is small by construction -- and is stored here rather than written by
    the machine: the control plane owns the artifact's identity (ADR-0009 point
    5), so it decides both the key and what sits beside it.
    """
    if not result.get("adapter_path"):
        return None

    weights_key = storage.artifact_key(job_id, storage.ADAPTER_WEIGHTS_NAME)
    want = result.get("adapter_sha256")
    if not want:
        # A result that names an artifact but carries no checksum cannot be
        # verified at all -- refused rather than delivered uncheckable.
        _delete_stored_artifact(job_id)
        raise OrchestratorError(
            "artifact_unverified",
            "The run result carried no artifact checksum, so the stored "
            "artifact cannot be verified; it was refused rather than "
            "delivered uncheckable.",
        )

    # The machine's own half of verification is the URL; this half is the
    # bytes that landed. Streamed so that verifying an artifact that grew
    # without bound holds one chunk, not the object.
    digest = hashlib.sha256()
    received = 0
    try:
        for chunk in storage.STORE.get_stream(weights_key):
            digest.update(chunk)
            received += len(chunk)
    except storage.ObjectNotFound:
        # The machine claimed an artifact but nothing landed at the key. If it
        # recorded its own upload failure, say so rather than guessing -- the
        # specific cause is already in result.json.
        recorded = result.get("artifact_upload")
        cause = ""
        if isinstance(recorded, dict) and recorded.get("error"):
            cause = f" The machine recorded: {recorded['error']}"
        raise OrchestratorError(
            "artifact_unverified",
            "The run result names an artifact, but no object landed at its "
            "storage key. The machine's upload must have failed." + cause,
        ) from None

    got = digest.hexdigest()
    if got != want:
        # A silently truncated or corrupted upload produces an object that
        # looks fine and is not; the machine's own checksum is what catches
        # it. The object is removed rather than left to read as the
        # deliverable.
        _delete_stored_artifact(job_id)
        raise OrchestratorError(
            "artifact_corrupt",
            f"Artifact SHA mismatch: machine reported {want[:16]}…, stored "
            f"object is {got[:16]}…",
        )
    db.add_event(job_id, "log", f"Artifact verified, {received / 1e6:.1f} MB")

    # A bare .safetensors is not a loadable adapter: PEFT needs
    # adapter_config.json beside it to know the rank, alpha and target modules.
    # Shipping only the weights would have handed the user a file that looks
    # like the deliverable and cannot be used. The config is already inside
    # result.json, so this costs no extra transfer.
    members: list[dict] = [
        {"name": storage.ADAPTER_WEIGHTS_NAME, "key": weights_key}
    ]
    adapter_config = result.get("adapter_config")
    if adapter_config:
        config_key = storage.artifact_key(job_id, storage.ADAPTER_CONFIG_NAME)
        storage.STORE.put(
            config_key,
            json.dumps(adapter_config, indent=2).encode("utf-8"),
        )
        members.append(
            {"name": storage.ADAPTER_CONFIG_NAME, "key": config_key}
        )
    else:
        db.add_event(
            job_id,
            "error",
            "No adapter_config.json in the run result; the downloaded "
            "artifact will not load without one.",
        )
    return {
        "kind": artifacts.kind_for(method),
        "members": members,
        "weights_key": weights_key,
        "bytes": received,
        "sha256": want,
    }


def _base_checkpoint_record(ckpt: dict) -> dict:
    """The fields a reported checkpoint carries regardless of its verdict."""
    base: dict = {}
    for field in ("step", "slot"):
        if ckpt.get(field) is not None:
            base[field] = ckpt[field]
    for loss_key in ("loss", "held_out_loss"):
        if ckpt.get(loss_key) is not None:
            base[loss_key] = ckpt[loss_key]
    return base


def _failed_checkpoint(job_id: str, ckpt: dict) -> dict:
    """A checkpoint whose upload the machine itself recorded as failed."""
    base = _base_checkpoint_record(ckpt)
    error = (
        ckpt.get("error") or "the machine did not report a successful upload"
    )
    db.add_event(
        job_id,
        "error",
        f"Checkpoint at step {base.get('step')} was not uploaded: {error}",
    )
    return {**base, "verified": False, "error": error}


def _superseded_checkpoint(job_id: str, ckpt: dict) -> dict:
    """A checkpoint whose slot a newer checkpoint overwrote (retention).

    Not a failure: the ring holds the newest `CHECKPOINT_RETENTION`
    checkpoints by construction, so an older one is *evicted*, and recording
    it as a checksum mismatch would misdescribe a design decision as
    corruption. It is recorded as superseded -- present, verifiable at the
    time it was written, no longer retained.
    """
    base = _base_checkpoint_record(ckpt)
    db.add_event(
        job_id,
        "log",
        f"Checkpoint at step {base.get('step')} superseded by a newer "
        f"checkpoint (retention {config.CHECKPOINT_RETENTION})",
    )
    return {**base, "verified": False, "superseded": True}


def _verify_checkpoint(job_id: str, ckpt: dict) -> dict:
    """Verify one retained checkpoint against what landed, or record its fall.

    Returns the record that becomes the job's answer about this checkpoint:
    `verified: True` only when the stored slot object streams back to the
    SHA-256 the machine reported for it, byte for byte. Anything short of that
    -- an object that never landed, or one whose bytes do not hash -- is
    recorded as `verified: False` with the reason, so a partially written
    checkpoint is never presented as complete (issue #37).

    Deliberately not terminal: the adapter is the deliverable, and a failed
    checkpoint upload does not make a trained adapter untrained. What it does
    is make that checkpoint unavailable for resumption, which the record and
    the event say plainly.
    """
    base = _base_checkpoint_record(ckpt)
    step = base.get("step")
    slot = base.get("slot")

    if not isinstance(slot, int):
        db.add_event(
            job_id,
            "error",
            f"Checkpoint at step {step} carried no usable slot; not recorded",
        )
        return {**base, "verified": False, "error": "no usable slot reported"}

    want = ckpt.get("sha256")
    if not want:
        db.add_event(
            job_id,
            "error",
            f"Checkpoint at step {step} carried no checksum; not recorded",
        )
        return {**base, "verified": False, "error": "no checksum reported"}

    key = storage.checkpoint_key(job_id, slot)
    digest = hashlib.sha256()
    received = 0
    try:
        for chunk in storage.STORE.get_stream(key):
            digest.update(chunk)
            received += len(chunk)
    except storage.ObjectNotFound:
        db.add_event(
            job_id,
            "error",
            f"Checkpoint at step {step} reported a slot, but no object landed "
            f"at its storage key",
        )
        return {
            **base,
            "verified": False,
            "error": "no object landed at the slot",
        }

    if digest.hexdigest() != want:
        db.add_event(
            job_id,
            "error",
            f"Checkpoint at step {step} did not match its reported checksum; "
            f"it was not recorded as complete",
        )
        return {**base, "verified": False, "error": "checksum mismatch"}

    db.add_event(
        job_id,
        "log",
        f"Checkpoint at step {step} verified, {received / 1e6:.1f} MB",
    )
    return {
        **base,
        "key": key,
        "sha256": want,
        "bytes": received,
        "verified": True,
    }


def _collect_checkpoints(job_id: str, result: dict) -> list[dict]:
    """Verify the checkpoints the machine wrote, and record the verdicts.

    Runs beside `_collect_artifact`, on whichever outcome the run reached --
    a failed run's checkpoints are still recovery material, so they are
    verified and recorded before the failure is reported, never presented as
    complete without that verification. Streaming, never whole: checkpoints
    are larger than adapters, so each stored object is pulled through the hash
    one chunk at a time and held nowhere.

    The verdicts distinguish the three things that can be true of a reported
    checkpoint: verified (its slot still holds its bytes), superseded (its
    slot was overwritten by a newer checkpoint -- retention, not corruption),
    or failed (its upload never succeeded). Only the first is complete.
    """
    reported = result.get("checkpoints")
    if not isinstance(reported, list) or not reported:
        return []

    successes = [
        c for c in reported if isinstance(c, dict) and c.get("ok") is True
    ]
    successes.sort(key=lambda c: c.get("step", 0))
    failures = [
        c
        for c in reported
        if not (isinstance(c, dict) and c.get("ok") is True)
    ]

    retained = config.CHECKPOINT_RETENTION
    retained_steps = (
        {c.get("step") for c in successes[-retained:]}
        if retained > 0
        else set()
    )

    records = [_failed_checkpoint(job_id, ckpt) for ckpt in failures]
    for ckpt in successes:
        if ckpt.get("step") in retained_steps:
            records.append(_verify_checkpoint(job_id, ckpt))
        else:
            records.append(_superseded_checkpoint(job_id, ckpt))
    return sorted(records, key=lambda r: r.get("step", -1))


def _record_best_checkpoint(job_id: str, records: list[dict]) -> None:
    """Choose the run's result checkpoint by held-out loss, and store the choice.

    Issue #62. The choice is recorded once, at the moment the verified
    checkpoints are recorded, and never recomputed: the whole point is that a
    run's answer cannot change later -- retention evicting a checkpoint or the
    selection rule being edited must not move a choice that was already made.
    Selection runs over the retained, verified checkpoints only: a checkpoint
    whose bytes are not in storage is not something a run can stand behind.
    Runs on both terminal outcomes record one -- a failed run's checkpoints are
    still its recovery material, and naming which one is best is as useful to a
    resumption as to a delivered result.
    """
    verified = [c for c in records if c.get("verified") is True]
    chosen = checkpoint.select_best_checkpoint(verified)
    db.set_best_checkpoint(job_id, chosen.to_dict())
    if chosen.step is not None:
        db.add_event(
            job_id,
            "log",
            f"Best checkpoint by held-out loss: step {chosen.step}"
            f"{f' ({chosen.held_out_loss})' if chosen.held_out_loss is not None else ''}."
            f" {chosen.reason}",
        )


def _discard_if_cancelled(job_id: str, check) -> None:
    """Throw away an artifact that arrived after the user asked to stop.

    The alternative -- keeping it, since it is trained and paid for -- would
    mean the answer a user got when they clicked depended on how many seconds
    the download took, which is the one thing about their own decision they
    cannot see. Cancellation is destructive by decision (ADR-0003), and a
    decision that holds only outside a race is not one.

    The machine writes its artifact directly now, so by the time this runs the
    object may already be in storage: the delete is what keeps a request
    answered with "no adapter will be produced" true even though an upload
    already landed. Checkpoints are discarded the same way -- a job the user
    stopped keeps no recovery material either (issue #37).
    """
    try:
        check()
    except Cancelled:
        _delete_stored_artifact(job_id)
        _delete_stored_checkpoints(job_id)
        raise


def _teardown(provider: Provider, job_id: str, machine) -> None:
    """Destroy the machine, then confirm it independently.

    A destroy call that returns cleanly is a claim. The evidence is the machine
    no longer being listed, and a machine that is still listed is billing right
    now — so it is reported as an error an operator cannot miss.
    """
    for attempt in range(DESTROY_ATTEMPTS):
        try:
            provider.destroy(machine.machine_id)
            db.add_event(
                job_id, "log", f"Machine {machine.machine_id} destroyed"
            )
            break
        except Exception as e:
            db.add_event(job_id, "error", f"Destroy attempt failed: {e}")
            if attempt < DESTROY_ATTEMPTS - 1:
                time.sleep(DESTROY_RETRY_DELAY_S)
    try:
        if machine.machine_id in provider.list_machine_ids():
            db.add_event(
                job_id,
                "error",
                f"STRAY MACHINE {machine.machine_id} still listed — "
                f"destroy it manually, it is billing",
            )
    except Exception as e:
        db.add_event(
            job_id,
            "error",
            f"Could not confirm teardown of machine {machine.machine_id}: {e}",
        )


def _stall_reporter(job_id: str, limits: RunLimits):
    """Say so when a long silence ends, so the detector is visible working.

    Deliberately not one event per line: the event log is the user's view of
    their own run, and a bookkeeping entry per training step would bury the
    output it exists to make legible. What is worth an event is a gap long
    enough that the next one might not have come back at all.
    """

    def reset(gap: float) -> None:
        db.add_event(
            job_id,
            "log",
            f"Output resumed after {gap:.0f}s of silence "
            f"(stall limit {limits.stall_timeout_s:.0f}s)",
            {
                "silence_s": round(gap, 1),
                "stall_timeout_s": limits.stall_timeout_s,
            },
        )

    return reset


def _cancellation_check(job_id: str):
    """A callable that raises the moment the user has asked this job to stop.

    Shaped as a check rather than a signal because the request and the run are
    on different threads and nothing connects them but the row: the job asks,
    it is not told. Cheap enough to ask often — one indexed read by primary key
    against a local file, once per stage boundary and once per trip round the
    streaming loop.
    """

    def check() -> None:
        if db.cancel_requested(job_id):
            raise Cancelled(CANCEL_MESSAGE)

    return check


def _attempt(
    provider: Provider,
    job_id: str,
    machines: list,
    limits: RunLimits,
    models: Models,
) -> tuple[str, str, dict]:
    """Do the work. Returns the terminal state to record, but never records it.

    Recording the outcome is the caller's job precisely so that teardown can
    happen in between: the confirmation that the machine is gone must reach the
    event log before the job reports that it is finished.

    `machines` is the caller's handle on anything created, appended to the
    moment it exists — a machine that exists but was never recorded is a
    machine nobody destroys.
    """
    job = db.require_job(job_id)

    # Issue #24: the fault surface, off by default, and checked before
    # anything else -- before model facts are resolved and long before
    # anything is provisioned. A job whose spec carries a fault is refused
    # here too (the create path already refused it; this is the belt for a
    # row that slipped past, and a guard that only lives on one side of a
    # money path is a hope). When the surface is on, the injected fault is
    # named in the run's own history, so a deliberately broken run can never
    # be mistaken for a real one.
    fault_spec = fault_surface.from_hyperparameters(job.get("hyperparameters"))
    if fault_spec is not None:
        if not (config.FAKE_PROVIDER or config.FAULT_SURFACE):
            raise OrchestratorError(
                "fault_surface_refused",
                "This job carries a fault spec, but the fault surface is off "
                "by default. It can only be switched on deliberately, with "
                "TEMPER_FAKE_PROVIDER (the zero-cost tier) or "
                "TEMPER_FAULT_SURFACE (the deliberate real-hardware tier); "
                "nothing was provisioned.",
            )
        name = fault_spec.get("name")
        if not isinstance(name, str) or not fault_surface.is_known(name):
            raise OrchestratorError(
                "fault_unknown",
                f"'{name}' is not a fault the surface knows; the job was "
                "refused rather than run under a fault nobody can explain.",
            )
        db.add_event(
            job_id,
            "log",
            f"Simulated fault injected: {name} - "
            f"{fault_surface.describe(name)}. This run is deliberately "
            "broken and cannot be mistaken for a real one.",
        )

    dataset = db.require_dataset(job["dataset_id"])
    model = catalog.get(job["base_model"]) or catalog.get(
        catalog.DEFAULT_MODEL
    )
    if model is None:
        # Unreachable while creation validates against this catalog, but a
        # re-read that outlives its guard fails by name on a money path.
        raise OrchestratorError(
            "unknown_model",
            f"Model '{job['base_model']}' is not in the catalog.",
        )
    enable_thinking = bool(dataset.get("enable_thinking"))
    revision = job.get("base_revision") or model.revision
    facts = models.resolve(model.repo, revision)
    hp = hyperparams.effective(job["hyperparameters"] or {})

    # The decisions the launch committed to (issue #79), re-applied here so
    # provisioning honours them rather than silently re-picking the
    # predictor's cheapest configuration: a run that provisioned something
    # other than what the plan froze would be lying about what it did. An
    # override that is no longer available at provisioning fails here, by
    # name, rather than being silently dropped.
    frozen_overrides = job.get("overrides") or []
    resolved_overrides = overrides.resolve(
        hp,
        [overrides.from_dict(d) for d in frozen_overrides],
    )

    # The image the machine will pull, read before anything is provisioned.
    # A job that cannot know which image it would run must not spend money
    # finding that out: "nothing may be provisioned without it" holds for the
    # published image exactly as it does for disk.
    reference = _trainer_reference()

    # Checked at every boundary between stages, for the same reason the
    # duration ceiling is: inside a provider call nothing is interruptible, so
    # the boundaries are where a request to stop can actually be honoured. The
    # cheapest place to notice one is before the machine exists at all.
    cancelled = _cancellation_check(job_id)

    cancelled()
    limits.check_duration()
    db.set_state(job_id, "provisioning", "Selecting hardware")
    try:
        plan = selection.select_hardware(
            facts,
            lora_r=hp["lora_r"],
            sequence_len=resolved_overrides.hyperparameters["sequence_len"],
            micro_batch_size=hp["micro_batch_size"],
            availability=provider.gpu_availability(),
            currency=provider.currency(),
            method=resolved_overrides.method,
            gpu_type=resolved_overrides.gpu_type,
            device_count=resolved_overrides.device_count,
        )
    except selection.NoFittingHardwareError as e:
        raise OrchestratorError("provider_capacity_unavailable", str(e)) from e
    try:
        disk_plan = disk.required_disk(
            facts,
            method=plan.method,
            lora_r=hp["lora_r"],
            retained_checkpoints=hp["save_total_limit"],
            provisioned_gb=resolved_overrides.disk_gb,
        )
    except disk.DiskExceedsCeilingError as e:
        raise OrchestratorError("disk_exceeds_ceiling", str(e)) from e
    except (disk.DiskBelowNeedError, disk.DiskBelowMinimumError) as e:
        raise OrchestratorError("disk_override_refused", str(e)) from e
    cancelled()
    db.set_state(
        job_id,
        "provisioning",
        f"Provisioning {plan.device_count}x {plan.gpu_type} ({plan.method}) "
        f"at {plan.price_per_hour}{plan.currency}/hr, "
        f"{disk_plan.provisioned_gb} GB disk",
        gpu_type=plan.gpu_type,
        price_per_hour=plan.price_per_hour,
        currency=plan.currency,
        device_count=plan.device_count,
        method=plan.method,
        disk_gb=disk_plan.provisioned_gb,
        storage_cost_usd_per_hour=disk_plan.storage_cost_usd_per_hour,
    )

    machine = provider.create(
        plan.gpu_type,
        plan.device_count,
        disk_plan.provisioned_gb,
        f"temper-{job_id[:12]}",
    )
    machines.append(machine)
    db.set_state(
        job_id,
        "preparing",
        f"Machine {machine.machine_id} running; waiting for SSH",
        machine_id=machine.machine_id,
    )

    cancelled()
    db.add_event(job_id, "log", provider.await_ready(machine))
    cancelled()
    # The dataset streams: push_stream holds at most one chunk, and the
    # dataset member is pulled through the storage seam in chunks, so the
    # control plane's memory has nothing to do with the size of the dataset.
    # The trainer image travels as its published digest inside the script
    # (issue #44) -- nothing to push, because the machine pulls it itself.
    provider.push_stream(
        machine,
        _dataset_chunks(dataset["object_key"]),
        DATASET_TARBALL,
    )

    # Checked between stages as well as inside the stream: the guard below can
    # only notice the ceiling while lines are arriving, and everything above
    # this line happened before any line existed.
    limits.check_duration()
    cancelled()

    db.set_state(job_id, "training", "Pulling image and training")
    # The machine writes its own artifact to a scoped grant (ADR-0009), so the
    # control plane hands it a URL rather than later pulling bytes through
    # itself. The grant covers one key -- the weights this job will produce --
    # and expires no later than the job can legitimately end: its lifetime is
    # what remains of the duration ceiling, counted from the job's own
    # creation timestamp (the ceiling counts from the same origin). An
    # abandoned URL is not a standing grant.
    weights_key = storage.artifact_key(job_id, storage.ADAPTER_WEIGHTS_NAME)
    remaining = max(
        1.0, config.MAX_JOB_DURATION_S - (time.time() - job["created_at"])
    )
    grant = storage.STORE.mint_write_grant(weights_key, remaining)
    # Issue #37: the machine also writes its own checkpoints as it produces
    # them, one scoped grant per retention slot. The slot count is bounded
    # configuration (storage never holds more checkpoint objects per job than
    # this), and every grant carries the same remaining-ceiling lifetime as
    # the artifact's -- no grant outlives the job it was minted for.
    checkpoint_grants = [
        storage.STORE.mint_write_grant(key, remaining)
        for key in storage.checkpoint_keys(job_id, config.CHECKPOINT_RETENTION)
    ]
    script = _remote_script(
        job,
        model,
        enable_thinking,
        reference,
        artifact_grant=grant,
        checkpoint_grants=checkpoint_grants,
    )
    # The guard sits between the transport and the reader, so both limits hold
    # for any provider rather than for the SSH one only.
    lines = guard(
        provider.stream(machine, script),
        limits,
        on_reset=_stall_reporter(job_id, limits),
        check=cancelled,
    )
    result = _consume(job_id, lines)
    if not result.get("ok"):
        # A failed run's checkpoints are still recovery material: what the
        # machine wrote and reported is verified and recorded before the job
        # reports failure, so a later resumption can find what survived the
        # machine. The outcome is unchanged -- this run produced no adapter --
        # but its checkpoints are no longer orphaned bytes in the store.
        failed_checkpoints = _collect_checkpoints(job_id, result)
        db.set_checkpoints(job_id, failed_checkpoints)
        _record_best_checkpoint(job_id, failed_checkpoints)
        # The result document names its own failure where it can. A stage that
        # failed before training started is not a training failure, and telling
        # a user otherwise sends them to read the wrong logs.
        raise OrchestratorError(
            result.get("error_code") or "training_failed",
            result.get("error") or "Training did not complete.",
        )

    # Cancellation is honoured to the last moment an adapter could appear, not
    # only while the stream is open. The machine writes its artifact directly
    # now, so by the time the result is parsed the object may already be in
    # storage: a request answered with "no adapter will be produced" that then
    # produced one would be the single worst thing this path could tell a user,
    # so what already landed is discarded rather than kept.
    _discard_if_cancelled(job_id, cancelled)
    db.set_state(job_id, "packaging", "Verifying artifact")
    artifact = _collect_artifact(job_id, result, plan.method)
    checkpoints = _collect_checkpoints(job_id, result)
    db.set_checkpoints(job_id, checkpoints)
    # The choice is recorded beside the verified checkpoints, not derived at
    # download time (issue #62): a run's answer must not move under retention
    # or a rule edit, so the chosen step and its reason are frozen here.
    _record_best_checkpoint(job_id, checkpoints)
    _discard_if_cancelled(job_id, cancelled)
    fields: dict = {
        "result_json": result,
        "artifact_key": (
            artifact["weights_key"] if artifact is not None else None
        ),
    }
    if artifact is not None:
        # Only a produced artifact records members; a run that produced none
        # leaves the column NULL rather than a JSON "null".
        fields["artifact_json"] = artifact
    return ("complete", "Training complete", fields)


def _record_actuals(
    job_id: str, result: dict | None, terminal_state: str, terminal_ts: float
) -> None:
    """Freeze the measured figures onto a terminal job (issue #77).

    Runs *before* the terminal state transition, so that by the moment the
    job's status reads terminal the actuals are already beside it: the
    orchestrator thread's database work ends when the status becomes visible,
    and a reader can never observe a terminal job without its actuals -- nor
    a test tear down a database while the thread still writes to it. The
    `packaging` stage ends at the terminal transition, which is happening now
    (`terminal_ts`, the same instant set_state stamps), so the measurement
    includes it rather than losing the run's last stage.

    The result document carries the trainer's measured peak VRAM when it
    exists; a run that produced no result records duration and cost and no
    peak -- the honest absence, never a guessed number.
    """
    job = db.get_job(job_id)
    if job is None:
        return
    row = dict(job)
    if row.get("finished_at") is None:
        row["finished_at"] = terminal_ts
    events = db.get_events(job_id)
    events = [
        *events,
        {
            "id": 0,
            "job_id": job_id,
            "ts": terminal_ts,
            "kind": "state",
            "message": terminal_state,
            "data": {"state": terminal_state},
        },
    ]
    measured = actuals.measure(row, events, result)
    db.record_actuals(job_id, actuals.to_dict(measured))


def run_job(
    job_id: str,
    provider: Provider | None = None,
    limits: RunLimits | None = None,
    models: Models | None = None,
) -> None:
    """Drive one job to a terminal state. Always tears down.

    The provider is injected so the whole path is testable; omit it and the
    real one is built, which is where a missing credential surfaces.

    The limits are injected for the same reason, and they carry their own
    clock: a fifteen-minute silence and a twenty-four-hour run are both things
    the suite has to be able to reach, and it cannot reach them by waiting.

    `models` resolves the base model to the facts hardware selection prices
    against; omit it and the real, network-backed resolver is built, mirroring
    the provider.
    """
    models = models or new_models()
    # Stamped here, so the ceiling counts from the moment the job began rather
    # than from the moment output started.
    limits = (limits or RunLimits.from_config()).start()

    # Before the provider is even built. A job cancelled while it sat in
    # `queued` -- which is where it is between the request that created it and
    # the thread that picks it up -- should cost nothing at all, and building a
    # client is the first thing on this path that can talk to the account.
    if db.cancel_requested(job_id):
        # Still recorded (issue #77): the attempt consumed the wall time from
        # creation to the cancellation, even though no machine was provisioned.
        _record_actuals(job_id, None, "cancelled", time.time())
        db.set_state(job_id, "cancelled", CANCEL_MESSAGE)
        return

    owns_provider = provider is None
    # The None check, not `if owns_provider:`: only a condition mypy can see
    # through narrows the parameter here, and everything below this block
    # passes the provider on as non-optional.
    if provider is None:
        try:
            provider = new_provider()
        except OrchestratorError as e:
            _record_actuals(job_id, None, "failed", time.time())
            db.set_state(
                job_id,
                "failed",
                str(e),
                error_code=e.code,
                error_message=str(e),
            )
            return
        except Exception as e:
            # Nothing was provisioned, so there is nothing to tear down -- but
            # the job still has to reach a terminal state rather than sit in
            # `queued` forever because the SDK failed to import.
            _record_actuals(job_id, None, "failed", time.time())
            db.set_state(
                job_id,
                "failed",
                f"{type(e).__name__}: {e}",
                error_code="internal_error",
                error_message=str(e),
            )
            return

    machines: list = []
    wall_started = time.time()
    try:
        try:
            outcome = _attempt(provider, job_id, machines, limits, models)
        except Cancelled as e:
            # No error code and no error message: the user's own decision is
            # not a defect, and a `cancelled` job carrying an error code would
            # be read as one by every client that branches on codes.
            #
            # The machine writes its artifact directly, so an upload that
            # landed before the cancellation was seen is discarded here rather
            # than left readable as the deliverable. Checkpoints are discarded
            # with it (issue #37): a job the user stopped keeps no recovery
            # material.
            _delete_stored_artifact(job_id)
            _delete_stored_checkpoints(job_id)
            outcome = ("cancelled", str(e), {})
        except OrchestratorError as e:
            outcome = (
                "failed",
                str(e),
                {"error_code": e.code, "error_message": str(e)},
            )
        except Exception as e:
            outcome = (
                "failed",
                f"{type(e).__name__}: {e}",
                {"error_code": "internal_error", "error_message": str(e)},
            )
        finally:
            # Before the terminal transition, and on every path including one
            # nobody anticipated.
            for machine in machines:
                _teardown(provider, job_id, machine)
            db.add_event(
                job_id,
                "log",
                f"Job finished in {time.time() - wall_started:.0f}s",
            )

        state, message, fields = outcome
        # The measured half of issue #77's comparison, frozen the moment the
        # run ends -- the mirror of the quote frozen at launch -- so no run is
        # wasted even before anything consumes the record. Frozen *before*
        # the terminal status is written, so a reader can never observe a
        # terminal job without its actuals beside it.
        _record_actuals(job_id, fields.get("result_json"), state, time.time())
        db.set_state(job_id, state, message, **fields)
    finally:
        if owns_provider:
            provider.close()


def launch(
    job_id: str,
    provider: Provider | None = None,
    limits: RunLimits | None = None,
    models: Models | None = None,
) -> None:
    """Start a job on a background thread.

    A thread rather than Celery: one process is the whole deployment, and a
    queue with one worker and no retries would be ceremony. The cost is honest
    -- a process restart orphans in-flight jobs, which `db.active_jobs()`
    surfaces at startup rather than hiding.
    """
    threading.Thread(
        target=run_job,
        args=(job_id, provider, limits, models),
        daemon=True,
        name=f"job-{job_id[:8]}",
    ).start()
