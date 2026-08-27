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

from temper_core import catalog, disk, events, hyperparams, selection
from temper_core.errors import Cancelled, OrchestratorError
from temper_core.models import Models

from . import db, storage
from .chunks import ChunkReader, piped_chunks
from .limits import RunLimits, guard
from .models import new_models
from .provider import Provider, new_provider
from .trainer_build import TRAINER_SOURCES, normalised

TRAINER_TARBALL = "/tmp/trainer.tar.gz"
DATASET_TARBALL = "/tmp/dataset.tar.gz"
# The machine emits this when it fails before the trainer ever runs, so that a
# pre-training failure still arrives as a result document naming its own code
# rather than as "the trainer produced nothing".
SOURCE_UNPACK_FAILED = json.dumps(
    {
        "stage": "source",
        "ok": False,
        "error_code": "source_upload_failed",
        "error": "The trainer sources reached the machine but did not unpack.",
    }
)
# Each of these is echoed *after* the result marker, so a failure that never
# reached the trainer still arrives as a result document naming its own cause
# rather than as "the trainer produced nothing". Each names its own code for the
# same reason the unpack failure does: a build that never produced an image is
# not a training failure, and telling a user otherwise sends them to read the
# wrong output. Single quotes are forbidden in these strings -- the script
# echoes them inside a single-quoted shell literal.
BUILD_FAILED = json.dumps(
    {
        "stage": "build",
        "ok": False,
        "error_code": "image_build_failed",
        "error": "The trainer image failed to build; the build output is in the "
        "job events.",
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
    "Cancellation requested. The machine is being destroyed and no adapter "
    "will be produced."
)
CANCEL_MESSAGE = "Cancelled at your request. No adapter was produced."


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


def _trainer_chunks() -> Iterator[bytes]:
    """The trainer image's sources as one streamed tar.gz.

    Members are flat: the machine untars into a single directory and builds
    there, so a member name is a build-context file name and the Dockerfile's
    COPY reads the same whichever directory a source came from.
    """
    members = []
    for source in TRAINER_SOURCES:
        data = normalised(source)
        members.append(_TarMember(source.name, len(data), data))
    return piped_chunks(lambda sink: _pour_tar(members, sink))


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
    job: dict, model: catalog.BaseModel, enable_thinking: bool
) -> bytes:
    """The on-machine script: build the image, run the job, print the result.

    Nothing here carries the dataset. It arrived ahead of this script as its
    own archive on standard input, so the script stays a fixed few kilobytes
    however large the dataset is -- it is never doubled into hex text, held in
    memory several times over, or fed through a pipe sized for commands.

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
    script = f"""
set -u
say() {{ echo "[$(date +%H:%M:%S)] $*" >&2; }}

mkdir -p /tmp/trainer
tar xzf {TRAINER_TARBALL} -C /tmp/trainer || {{
  say "SOURCE UNPACK FAILED"
  echo "{RESULT_MARKER}"
  echo '{SOURCE_UNPACK_FAILED}'
  exit 0
}}

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

say "building trainer image"
t0=$(date +%s)
# --progress=plain: the default renderer redraws a live display, which is
# unreadable once it is a line-oriented event log rather than a terminal.
sudo docker build --progress=plain -t temper-trainer:job /tmp/trainer 1>&2 || {{
  say "BUILD FAILED"
  echo "{RESULT_MARKER}"
  echo '{BUILD_FAILED}'
  exit 0
}}
say "image built in $(( $(date +%s) - t0 ))s"

say "running training"
# PYTHONUNBUFFERED is the innermost of the three redirections. The trainer and
# the framework beneath it are both Python, and Python buffers its stdout
# whenever it is a pipe rather than a terminal -- so without this the container
# holds minutes of output and the two layers outside it relay nothing.
sudo docker run --rm --gpus all \\
  -v /tmp/job:/job:ro -v /tmp/out:/out -e HF_HOME=/out/hf \\
  -e PYTHONUNBUFFERED=1 \\
  temper-trainer:job 1>&2 || say "TRAINER EXITED NONZERO"

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


def _fetch_adapter(
    provider: Provider, machine, job_id: str, result: dict
) -> str | None:
    """Pull the artifact off the machine and put it behind the storage seam.

    Returns the artifact weights' key in storage, or None when the machine
    produced nothing to store. The key -- not a path -- is what the job row
    records; only `storage` knows what it resolves to.

    Both halves stream: fetch_stream yields the machine's bytes in chunks,
    and put_stream stores them one chunk at a time while this function
    hashes the same chunks on the way past. Verification therefore completes
    only once every chunk has been stored, so the order claimed elsewhere --
    *verified before stored* -- holds here as *deleted unless verified*: a
    mismatched, unverifiable or empty transfer has its object removed again
    before this function returns. Nothing records the key against the job
    until then, so no reader can reach even the transient object.
    """
    rel = result.get("adapter_path")
    if not rel:
        return None

    weights_key = storage.artifact_key(job_id, storage.ADAPTER_WEIGHTS_NAME)
    digest = hashlib.sha256()
    received = 0

    def hashed(chunks: Iterator[bytes]) -> Iterator[bytes]:
        # The same iteration feeds the store and the hash: the bytes are
        # never held twice, because they are never held at all.
        nonlocal received
        for chunk in chunks:
            digest.update(chunk)
            received += len(chunk)
            yield chunk

    storage.STORE.put_stream(
        weights_key,
        hashed(provider.fetch_stream(machine, f"/tmp/out/{rel}")),
    )

    if received == 0:
        # The mirror of an empty read: nothing came off the machine.
        storage.STORE.delete(weights_key)
        db.add_event(
            job_id,
            "error",
            "Adapter fetch failed; artifact left on the machine",
        )
        return None

    # Verify against the hash the container computed. A silently truncated
    # transfer produces an object that looks fine and is not; hashing as
    # chunks arrive catches it without ever holding the payload. The hash is
    # mandatory: the trainer records it whenever it records a path, so a
    # result without one cannot be verified at all -- refused rather than
    # delivered uncheckable.
    got = digest.hexdigest()
    want = result.get("adapter_sha256")
    if not want:
        storage.STORE.delete(weights_key)
        raise OrchestratorError(
            "artifact_unverified",
            "The run result carried no adapter checksum, so the download "
            "cannot be verified; it was refused rather than delivered "
            "uncheckable.",
        )
    if got != want:
        storage.STORE.delete(weights_key)
        raise OrchestratorError(
            "artifact_corrupt",
            f"Adapter SHA mismatch: container reported {want[:16]}…, "
            f"downloaded file is {got[:16]}…",
        )
    db.add_event(job_id, "log", f"Adapter verified, {received / 1e6:.1f} MB")

    # A bare .safetensors is not a loadable adapter: PEFT needs
    # adapter_config.json beside it to know the rank, alpha and target modules.
    # Shipping only the weights would have handed the user a file that looks
    # like the deliverable and cannot be used. The config is already inside
    # result.json, so this costs no extra transfer.
    adapter_config = result.get("adapter_config")
    if adapter_config:
        storage.STORE.put(
            storage.artifact_key(job_id, storage.ADAPTER_CONFIG_NAME),
            json.dumps(adapter_config, indent=2).encode("utf-8"),
        )
    else:
        db.add_event(
            job_id,
            "error",
            "No adapter_config.json in the run result; the downloaded "
            "adapter will not load without one.",
        )
    return weights_key


def _discard_if_cancelled(job_id: str, check) -> None:
    """Throw away an artifact that arrived after the user asked to stop.

    The alternative -- keeping it, since it is trained and paid for -- would
    mean the answer a user got when they clicked depended on how many seconds
    the download took, which is the one thing about their own decision they
    cannot see. Cancellation is destructive by decision (ADR-0003), and a
    decision that holds only outside a race is not one.
    """
    try:
        check()
    except Cancelled:
        for name in storage.ARTIFACT_MEMBERS:
            # Deletion failures are suppressed deliberately: teardown must
            # not mask the cancellation that caused it, and an orphaned
            # object is cheaper than a half-reported state.
            with suppress(Exception):
                storage.STORE.delete(storage.artifact_key(job_id, name))
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
            sequence_len=hp["sequence_len"],
            micro_batch_size=hp["micro_batch_size"],
            availability=provider.gpu_availability(),
            currency=provider.currency(),
        )
    except selection.NoFittingHardwareError as e:
        raise OrchestratorError("provider_capacity_unavailable", str(e)) from e
    try:
        disk_plan = disk.required_disk(
            facts,
            method=plan.method,
            lora_r=hp["lora_r"],
            retained_checkpoints=hp["save_total_limit"],
        )
    except disk.DiskExceedsCeilingError as e:
        raise OrchestratorError("disk_exceeds_ceiling", str(e)) from e
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
    # Both archives stream: push_stream holds at most one chunk, and the
    # dataset member is pulled through the storage seam in chunks, so the
    # control plane's memory has nothing to do with the size of the dataset
    # or the repo.
    provider.push_stream(machine, _trainer_chunks(), TRAINER_TARBALL)
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

    db.set_state(job_id, "training", "Building image and training")
    script = _remote_script(job, model, enable_thinking)
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
        # The result document names its own failure where it can. A stage that
        # failed before training started is not a training failure, and telling
        # a user otherwise sends them to read the wrong logs.
        raise OrchestratorError(
            result.get("error_code") or "training_failed",
            result.get("error") or "Training did not complete.",
        )

    # Cancellation is honoured to the last moment an adapter could appear, not
    # only while the stream is open. Packaging is not instantaneous -- it is a
    # download from a machine that is still billing -- and a request answered
    # with "no adapter will be produced" that then produced one would be the
    # single worst thing this path could tell a user.
    cancelled()
    db.set_state(job_id, "packaging", "Retrieving adapter")
    artifact_key = _fetch_adapter(provider, machine, job_id, result)
    _discard_if_cancelled(job_id, cancelled)
    return (
        "complete",
        "Training complete",
        {"result_json": result, "artifact_key": artifact_key},
    )


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
