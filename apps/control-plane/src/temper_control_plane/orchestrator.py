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
import time
from collections.abc import Iterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO, NamedTuple, Protocol

from temper_core import (
    actuals,
    artifacts,
    catalog,
    checkpoint,
    delivery,
    disk,
    divergence,
    events,
    gpus,
    hyperparams,
    memory_retry,
    overrides,
    progress,
    selection,
)
from temper_core import faults as fault_surface
from temper_core import (
    resume as resume_logic,
)
from temper_core.errors import Cancelled, OrchestratorError
from temper_core.models import Models

from . import config, db, storage
from .chunks import ChunkReader, piped_chunks
from .correlation import get_correlation_id, set_correlation_id
from .limits import BUDGET_EXHAUSTED_CODE, RunLimits, guard
from .logging import get_logger
from .models import new_models
from .provider import (
    Machine,
    Provider,
    container_name,
    new_provider,
    normalize_status,
)
from .trainer_build import published_reference

logger = get_logger(__name__)


class _Record(Protocol):
    """How a teardown records one line of what happened.

    `data` is optional: every caller writes a plain message for most lines
    and attaches a structured payload for a few. Declared as a Protocol so
    mypy sees the optional third argument the way the closures actually
    shape it, instead of requiring every call to pass three arguments.
    """

    def __call__(
        self, kind: str, message: str, data: dict | None = None
    ) -> None: ...


def _bind_correlation_from_job(job_id: str) -> str | None:
    """Bind the correlation identifier from the job row onto this context.

    The worker re-hydrates the request's identifier from the row before
    driving the job, so every structured log line the job emits carries the
    same ``correlation_id`` the request's error response did. Returns the
    identifier when present, ``None`` when the row predates the migration.
    """
    try:
        job = db.get_job(job_id)
    except Exception:
        set_correlation_id(None)
        return None
    if job is None:
        set_correlation_id(None)
        return None
    cid = job.get("correlation_id")
    if isinstance(cid, str) and cid:
        set_correlation_id(cid)
        return cid
    set_correlation_id(None)
    return None


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
# The code the machine reports when training ended without leaving a
# result.json -- the honest shape of an interruption (issue #60), whether the
# trainer was killed or the container died: no result document means the run
# was cut off, not that it failed on its own terms. Defined once so the
# result document and the interruption classification read the same name.
NO_RESULT_CODE = "trainer_no_result"
NO_RESULT = json.dumps(
    {
        "stage": "train",
        "ok": False,
        "error_code": NO_RESULT_CODE,
        "error": "The trainer produced no result.json; its output is in the job "
        "events.",
    }
)
RESULT_MARKER = "---RESULT---"
DESTROY_ATTEMPTS = 3
DESTROY_RETRY_DELAY_S = 5

# Teardown confirmation (spec 010, spike/teardown.py C17). The provider's
# listing is eventually consistent: after a destroy the machine can read
# absent, then reappear as `destroying`, then go absent for good. A single
# absent observation is therefore not evidence; confirmation requires
# consecutive absent listings, and a machine reported as `destroying` is
# treated as not yet confirmed rather than as a stray that will be
# retried or counted as leaked.
TEARDOWN_CONFIRM_SAMPLES = 3
TEARDOWN_CONFIRM_INTERVAL_S = 1.0
TEARDOWN_CONFIRM_TIMEOUT_S = 30.0

# The spend ceiling: the most a single job may cost, in the account currency's
# minor unit (paise for INR). Enforced by the control plane from elapsed time
# and the job's frozen price (issue #46), never by the training process itself
# -- a process that has stopped responding cannot enforce its own limit.
#
# The derivation is on record, the way ADR-0036 records 21.6 MB/s x 60 s, and
# each half of it is labelled measured or judgment:
#
# * The most a *legitimate* job can cost is bounded by the duration ceiling
#   (ADR-0002, 24h) and the most expensive card the platform has measured:
#   H200 at Rs 378.27/hr (reference-technical-architecture.md, measured
#   2026-08-17). 24h x Rs 378.27/hr is about Rs 9,078.
# * The ceiling is Rs 10,000 -- roughly 10% above that worst legitimate cost,
#   so it cannot fire on a legitimate run. A spend ceiling that fires on a
#   legitimate run is a bug, not a safety net.
# * Rs 10,000 is 20% of the account's Rs 50,000 grant (AGENTS.md): a runaway
#   is stopped before it can consume a fifth of the account.
#
# The figure is therefore derived from two inherited facts (the adopted 24h
# duration ceiling and the measured H200 rate) and one judgment (the 20% of
# grant). If the catalog or the rates move, this is the number to revisit.
SPEND_CEILING_MINOR = 10_000 * 100  # Rs 10,000 in paise

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


def _resume_extract(resume_from_checkpoint: str | None) -> str:
    """The shell that lands a resumed run's checkpoint on the machine.

    The archive was streamed to `/tmp/checkpoint.tar` ahead of the script
    (see `_attempt`); this extracts it under `/tmp/out/run` -- the host half
    of the container's `/out/run` -- so the checkpoint directory axolotl
    resumes from is exactly where the trainer's own `output_dir` lives, and
    `resume_from_checkpoint` (a container path) resolves to it. Empty when
    the run is not a resumption, so an ordinary script carries no trace of
    one. The archive is a plain (uncompressed) tar, the shape the trainer's
    uploader writes and this path re-ships unchanged.
    """
    if not resume_from_checkpoint:
        return ""
    return (
        "mkdir -p /tmp/out/run\n"
        "tar xf /tmp/checkpoint.tar -C /tmp/out/run\n"
        f'say "resuming from {resume_from_checkpoint}"\n'
    )


def _remote_script(
    job: dict,
    model: catalog.BaseModel,
    enable_thinking: bool,
    reference: str,
    method: str,
    artifact_grant: storage.WriteGrant | None = None,
    checkpoint_grants: list[storage.WriteGrant] | None = None,
    delivery_grants: list[tuple[str, storage.WriteGrant]] | None = None,
    resolved_hp: dict | None = None,
    resume_from_checkpoint: str | None = None,
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

    When `delivery_grants` is supplied (issue #74), one scoped write URL per
    requested delivery format rides into the job spec under `delivery_grants`,
    beside the frozen `delivery` request. The machine produces each format and
    writes it to its own grant -- the same ADR-0009 machinery the artifact and
    checkpoint grants use -- so a merged or quantised format leaves the machine
    the moment it is verified, and each is a distinct object the control plane
    verifies against its own checksum.

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
    # trainer is the resolver's answer to it -- resolved against the method
    # the plan chose (issue #66), so a full fine-tune's learning rate is the
    # full fine-tune's, not the adapter's. The method itself rides in the
    # spec so the trainer builds the matching config and collects the matching
    # artifact; a value two components must agree on is written once and read.
    # Issue #35: a memory recovery's retried attempt runs a different
    # resolved spec than the frozen one -- `resolved_hp` is the recovery's
    # answer (batch halved, accumulation doubled, ...) computed by
    # `memory_retry.MemoryEscalator` and carries the effective batch
    # unchanged. The trainer still resolves nothing: it applies whatever
    # spec arrives, exactly as before.
    job_spec = {
        "job_id": job["id"],
        "base_model": model.repo,
        "base_revision": revision,
        "method": method,
        "hyperparameters": resolved_hp
        or hyperparams.effective(job["hyperparameters"] or {}, method=method),
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
    # Issue #60: a resumed attempt continues from its last checkpoint, so the
    # spec carries the machine-local path axolotl resumes from. The trainer
    # already passes the key through to axolotl (which restores the optimiser,
    # scheduler and step position -- the whole checkpoint directory), so this
    # side only names where the archive was extracted to; the resumed step
    # itself is recorded on the attempt, never smuggled into the trainer spec
    # as a key it does not know. `resumed_spec` writes the key from the single
    # definition the trainer reads, so the two cannot drift (ADR-0010).
    if resume_from_checkpoint:
        job_spec = resume_logic.resumed_spec(job_spec, resume_from_checkpoint)
    # Issue #74: the frozen delivery request (which formats the launch asked
    # for) rides into the spec so the trainer knows what to produce, and each
    # produced format gets its own scoped write grant, one object per format.
    requested = job.get("delivery_request")
    if requested:
        job_spec["delivery"] = list(requested)
    if delivery_grants:
        job_spec["delivery_grants"] = [
            {
                "url": grant.url,
                "key": grant.key,
                "expires_at": grant.expires_at,
                "format": format_id,
            }
            for format_id, grant in delivery_grants
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
{_resume_extract(resume_from_checkpoint)}
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
# The container is named after the job (issue #46) so the spend ceiling's
# emergency checkpoint can signal it by name: `request_checkpoint` sends
# SIGTERM to the job's container and the trainer converts that into a final
# save + report.
sudo docker run --rm --gpus all \\
  --name {container_name(job["id"])} \\
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
    structured fields, a layer-pull or model-download line becomes a `progress`
    record (issue #49), and everything else becomes a `log` event. Classifying
    here rather than when the log is read is what makes a chart possible later
    without re-parsing prose — by then the job is over and the format the line
    was written in is whatever the framework happened to use that day.

    Progress is promoted, not filtered (issue #49): a progress line updates the
    phase's record — one superseding row per phase, with the rate measured live
    between readings — and the raw line is retained as the job's collapsed
    detail, so the hundreds of lines a pull produces never flood the event log
    and nothing is discarded. The events table therefore holds no progress
    rows; the log/metric/state/error history is what stays small enough for a
    finished page to render to its end.

    Divergence is detected on the measurements the platform already streams
    (issue #36): training loss as `metric` events and a non-finite loss line
    that the classifier keeps as `log`. A NaN/Inf loss or a loss that exceeds
    ``multiplier * trailing-average(window)`` for ``consecutive`` steps aborts
    with a plain cause and a stable code; instability short of divergence
    (the same exceedance for ``warning_consecutive`` steps) is surfaced as a
    warning rather than an abort, exactly as the acceptance criteria require.
    The thresholds are read from ``config`` (derived in
    ``temper_core.divergence`` and ``config`` beside them, the way ADR-0036's
    dataset ceiling records 21.6 MB/s times 60 seconds).

    Everything after the marker is the trainer's result document, which is
    machinery rather than output and is not logged as such.

    `lines` is consumed lazily and each event is written the moment its line is
    read, which is what stamps it with the time it actually happened. Reading
    the stream into a list first would put every event back on the same
    timestamp -- which is exactly what the log looked like before.
    """
    result_lines: list[str] = []
    seen_marker = False
    tracker = progress.ProgressTracker()
    detector = divergence.DivergenceDetector(
        multiplier=config.DIVERGENCE_MULTIPLIER,
        window=config.DIVERGENCE_WINDOW,
        consecutive=config.DIVERGENCE_CONSECUTIVE,
        warning_consecutive=config.WARNING_CONSECUTIVE,
    )
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
                if event.kind == events.PROGRESS:
                    data = event.data or {}
                    record = tracker.update(
                        progress.ProgressReading(
                            phase=data["phase"],
                            done=data.get("done"),
                            total=data.get("total"),
                            unit=data.get("unit"),
                        )
                    )
                    db.upsert_progress(
                        job_id,
                        record.phase,
                        record.done,
                        record.total,
                        record.rate,
                        record.eta_s,
                        record.ts,
                        text,
                    )
                    db.append_output(job_id, record.phase, text)
                else:
                    db.add_event(
                        job_id, event.kind, event.message[:500], event.data
                    )
                    # Divergence detection uses the measurements the platform
                    # already streams (issue #36), rather than adding a second
                    # measurement path. Two signals: a finite loss in a metric
                    # event, and a non-finite loss that the classifier kept as
                    # a log line (events.py deliberately does not promote NaN).
                    loss_value: float | None = None
                    if event.kind == events.METRIC and event.data is not None:
                        raw = event.data.get("loss")
                        if isinstance(raw, (int, float)) and not isinstance(
                            raw, bool
                        ):
                            loss_value = float(raw)
                    # A non-finite loss line is immediate divergence even though
                    # it never became a metric -- the same loss the metric
                    # detector would have seen if it had been finite. This is
                    # how the fault-surface's ``divergence`` fault trips the
                    # detector against the fake provider (it emits "{'loss': nan}").
                    if loss_value is not None:
                        result = detector.observe(loss_value)
                        if result.status == "diverged":
                            raise OrchestratorError(
                                result.code or divergence.DIVERGED_CODE,
                                result.message or divergence.DIVERGED_MESSAGE,
                            )
                        if result.status == "warning":
                            db.add_event(
                                job_id,
                                "log",
                                result.message
                                or divergence.INSTABILITY_MESSAGE,
                                {
                                    "code": result.code
                                    or divergence.INSTABILITY_CODE,
                                    "warning": True,
                                    "consecutive": result.consecutive,
                                },
                            )
                    elif divergence.is_non_finite_loss_line(text):
                        raise OrchestratorError(
                            divergence.DIVERGED_CODE,
                            divergence.DIVERGED_MESSAGE,
                        )
        else:
            result_lines.append(line)

    if not seen_marker:
        # The stream ended without ever producing a result document: the
        # machine or the worker died mid-run. That is an interruption (issue
        # #60), named as such -- not a training failure, which is a result
        # document that says the run failed -- so the job's history and the
        # resumption path can tell the two apart.
        raise OrchestratorError(
            resume_logic.INTERRUPTED_CODE,
            resume_logic.INTERRUPTED_MESSAGE,
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
    if not result.get("artifact_path"):
        return None

    weights_key = storage.artifact_key(job_id, storage.ADAPTER_WEIGHTS_NAME)
    want = result.get("artifact_sha256")
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

    # The member's arcname is the artifact's own file name -- the adapter's
    # `adapter_model.safetensors`, or a full fine-tune's `model.tar.gz`
    # (issue #66) -- so what the user extracts is named by what it is. The
    # storage key is the same one object the machine wrote through its grant
    # (ADR-0009); the record pairs the honest name with the key.
    members: list[dict] = [
        {"name": Path(result["artifact_path"]).name, "key": weights_key}
    ]
    adapter_config = result.get("adapter_config")
    if adapter_config:
        # A bare .safetensors is not a loadable adapter: PEFT needs
        # adapter_config.json beside it to know the rank, alpha and target
        # modules. Shipping only the weights would have handed the user a file
        # that looks like the deliverable and cannot be used. The config is
        # already inside result.json, so this costs no extra transfer. A full
        # model carries its own config inside the archive, so there is no
        # second member to store.
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
            "log",
            "Artifact is its own complete model (no separate adapter config).",
        )
    return {
        "kind": artifacts.kind_for(method),
        "members": members,
        "weights_key": weights_key,
        "bytes": received,
        "sha256": want,
    }


# The archive name each delivery format is written as (issue #74). The trainer
# writes `delivery/merged.tar.gz` and `delivery/quantised.gguf`; this side
# must mint the grant's key from the same name, so the two are defined here
# once -- a value two components must agree on is never retyped.
DELIVERY_ARCHIVE_NAMES: dict[str, str] = {
    delivery.DELIVERY_FORMAT_MERGED: "merged.tar.gz",
    delivery.DELIVERY_FORMAT_QUANTISED: "quantised.gguf",
}


def _delivery_archive_name(format_id: str) -> str:
    """The archive name a delivery format is written as, or a loud refusal."""
    try:
        return DELIVERY_ARCHIVE_NAMES[format_id]
    except KeyError:
        raise OrchestratorError(
            "unknown_delivery_format",
            f"'{format_id}' is not a delivery format this orchestrator can "
            "grant a write for.",
        ) from None


def _collect_delivery(job_id: str, result: dict) -> list[dict]:
    """Verify the delivery formats the machine wrote, and record the verdicts.

    Issue #74. The trainer reports each produced format in
    `result["delivery"]` -- its archive name, checksum and upload outcome --
    and writes it to its own scoped grant (ADR-0009, one object per format).
    This side verifies each object that landed against the checksum the
    trainer reported, exactly as `_collect_artifact` verifies the canonical
    artifact, and records the per-format verdict: a format whose bytes do not
    match is refused rather than served, because a download that looks fine
    and is not is the silent failure this project treats as the enemy.

    A format whose archive name is unknown to the vocabulary is refused the
    same way: the trainer and the control plane read the same
    `delivery-formats.json`, so an unknown name is a drift, not a new format.
    """
    reported = result.get("delivery") or []
    records: list[dict] = []
    for reported_rec in reported:
        if not isinstance(reported_rec, dict):
            continue
        format_id = reported_rec.get("format")
        try:
            archive_name = _delivery_archive_name(str(format_id))
        except OrchestratorError:
            records.append(
                {
                    **reported_rec,
                    "verified": False,
                    "error": f"unknown delivery format {format_id!r}",
                }
            )
            continue
        want = reported_rec.get("sha256")
        if not want:
            records.append(
                {
                    **reported_rec,
                    "verified": False,
                    "error": "no checksum reported for this format",
                }
            )
            continue
        key = storage.delivery_key(job_id, str(format_id), archive_name)
        digest = hashlib.sha256()
        received = 0
        try:
            for chunk in storage.STORE.get_stream(key):
                digest.update(chunk)
                received += len(chunk)
        except storage.ObjectNotFound:
            recorded = reported_rec.get("upload") or {}
            cause = ""
            if isinstance(recorded, dict) and recorded.get("error"):
                cause = f" The machine recorded: {recorded['error']}"
            records.append(
                {
                    **reported_rec,
                    "verified": False,
                    "error": "no object landed for this format." + cause,
                }
            )
            continue
        got = digest.hexdigest()
        if got != want:
            records.append(
                {
                    **reported_rec,
                    "verified": False,
                    "error": (
                        f"delivery format checksum mismatch: machine "
                        f"reported {want[:16]}..., stored is {got[:16]}..."
                    ),
                }
            )
            continue
        records.append(
            {
                **reported_rec,
                "verified": True,
                "key": key,
                "bytes": received,
                "members": [
                    {
                        "name": archive_name,
                        "key": key,
                    }
                ],
            }
        )
    return records


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


class _HashingStream(io.RawIOBase):
    """A file-like over a storage stream that feeds every byte read into a
    SHA-256.

    Lets one pass both verify a stored checkpoint and parse its
    `trainer_state.json`, without ever holding the object whole (ADR-0010):
    tarfile pulls blocks through `readinto`, and each block is hashed as it
    crosses, so `digest` covers exactly the bytes that are in storage.
    """

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._it = iter(chunks)
        self._buf = b""
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:  # noqa: N802 - the io protocol spells it this way
        if not self._buf:
            try:
                self._buf = next(self._it)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buf))
        piece = self._buf[:n]
        self._buf = self._buf[n:]
        self.digest.update(piece)
        self.bytes_read += n
        b[:n] = piece
        return n


def _checkpoint_identity(
    stream,
) -> tuple[int | None, float | None, float | None]:
    """The step, training loss and held-out loss a checkpoint tar records.

    Reads the checkpoint's own `trainer_state.json` the same way the
    trainer's uploader writes it (issue #37): `global_step` names the step,
    and the step's entry in `log_history` carries the losses. Returns
    (None, None, None) when the object is not a parseable checkpoint -- a
    slot that was overwritten mid-write, say -- so the caller can skip it
    rather than resume from bytes that are not a checkpoint.
    """
    step: int | None = None
    train_loss: float | None = None
    held_out: float | None = None
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                if not member.name.endswith("trainer_state.json"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = b""
                while chunk := f.read(1 << 20):
                    data += chunk
                try:
                    state = json.loads(data.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                raw_step = state.get("global_step")
                if isinstance(raw_step, int) and not isinstance(
                    raw_step, bool
                ):
                    step = raw_step
                for row in reversed(state.get("log_history") or []):
                    if not isinstance(row, dict) or row.get("step") != step:
                        continue
                    if isinstance(row.get("loss"), (int, float)):
                        train_loss = float(row["loss"])
                    if isinstance(row.get("eval_loss"), (int, float)):
                        held_out = float(row["eval_loss"])
                    break
    except (tarfile.TarError, EOFError, OSError):
        return None, None, None
    return step, train_loss, held_out


def _discover_interrupted_checkpoints(job_id: str) -> list[dict]:
    """Find and verify what a dead machine left in the checkpoint slots.

    An interrupted run produced no result document, so there is no manifest
    of what its machine wrote off itself before it died (issue #37 uploads
    checkpoints as it trains). The slots are the record: each retained slot
    is streamed, hashed, and parsed for its `trainer_state.json` (issue #60),
    and anything that verifies becomes a checkpoint record exactly like a
    machine-reported one -- step, slot, key, checksum, losses -- so a
    resumption can come back to it and the job's checkpoint record is honest
    even when the run cannot resume. `verified: True` here means *structural*
    verification -- the bytes stream end to end, hash deterministically, and
    parse as a complete checkpoint tar -- because there is no machine-reported
    manifest to hash against; the meaning is stated, never dressed up as the
    machine-reported kind. Streaming, never whole (ADR-0010); a slot that
    does not parse is skipped, because bytes that are not a checkpoint are
    nothing to resume from.
    """
    records: list[dict] = []
    for slot, key in enumerate(
        storage.checkpoint_keys(job_id, config.CHECKPOINT_RETENTION)
    ):
        try:
            chunks = storage.STORE.get_stream(key)
        except storage.ObjectNotFound:
            continue
        reader = _HashingStream(chunks)
        step, train_loss, held_out = _checkpoint_identity(reader)
        if step is None:
            continue
        record: dict = {
            "step": step,
            "slot": slot,
            "key": key,
            "sha256": reader.digest.hexdigest(),
            "bytes": reader.bytes_read,
            "verified": True,
        }
        if train_loss is not None:
            record["loss"] = train_loss
        if held_out is not None:
            record["held_out_loss"] = held_out
        records.append(record)
        db.add_event(
            job_id,
            "log",
            f"Discovered surviving checkpoint at step {step} "
            f"({reader.bytes_read / 1e6:.1f} MB)",
        )
    return sorted(records, key=lambda r: r.get("step", -1))


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


def _emergency_checkpoint(provider: Provider, machine, job_id: str) -> None:
    """Save what a spend-ceiling stop can save: the machine's own checkpoints.

    Issue #46's ordered shutdown is checkpoint, terminate, destroy. The
    checkpoint half is issue #37's point — the machine writes checkpoints off
    itself as it trains — so the control plane asks it to report them
    (`request_checkpoint`), records what it can verify, and lets the
    held-out-loss selection (ADR-0049) see the result: a checkpoint written at
    the ceiling must be visible to that selection, not stored somewhere it
    cannot see.

    Best-effort and bounded, and **failure-isolated**: this runs in the middle
    of the budget_exhausted handling, so nothing it does may prevent the job
    reaching its terminal state. A machine that does not answer — it is, after
    all, the machine that has gone wrong — records nothing and the shutdown
    proceeds regardless: an unresponsive trainer cannot be asked to save, and
    the record says so rather than claiming a checkpoint it does not have. A
    recording step that itself fails (a malformed manifest, a storage error)
    is logged and the shutdown proceeds; the terminal reason is
    budget_exhausted either way.
    """
    db.add_event(
        job_id,
        "log",
        "Spend ceiling reached — requesting an emergency checkpoint before "
        "shutdown",
    )
    manifest = None
    try:
        manifest = provider.request_checkpoint(machine, job_id)
    except Exception as e:  # noqa: BLE001 - a failed request must not block the shutdown
        db.add_event(
            job_id,
            "error",
            f"Could not request an emergency checkpoint: {e}",
        )
    reported = manifest if isinstance(manifest, list) else []
    if not reported:
        db.add_event(
            job_id,
            "log",
            "No checkpoints were saved at the ceiling — the machine reported "
            "none",
        )
        return
    try:
        records = _collect_checkpoints(job_id, {"checkpoints": reported})
        db.set_checkpoints(job_id, records)
        _record_best_checkpoint(job_id, records)
        verified = sum(1 for r in records if r.get("verified") is True)
        db.add_event(
            job_id,
            "log",
            f"{verified} checkpoint(s) saved and verified at the spend "
            "ceiling",
        )
    except Exception as e:  # noqa: BLE001 - a recording failure must not block the shutdown
        db.add_event(
            job_id,
            "error",
            f"Could not record the emergency checkpoint: {e}",
        )


def confirmed_destroy(
    provider: Provider,
    machine,
    record: _Record,
) -> bool:
    """Destroy a machine, then confirm it independently. Records via `record`.

    A destroy call that returns cleanly is a claim. The evidence is the machine
    no longer being listed, and a machine that is still listed is billing right
    now — so it is reported as an error an operator cannot miss.

    The listing itself is eventually consistent (spec 010, spike/teardown.py
    C17): after a destroy the provider can report absent, then reappear as
    `destroying`, then go absent for good. A single absent observation is
    therefore not proof; confirmation requires `TEARDOWN_CONFIRM_SAMPLES`
    consecutive absent listings, and a machine reported as `destroying` is
    treated as not yet confirmed rather than as a stray. A `destroy` the
    provider refuses is retried `DESTROY_ATTEMPTS` times and, when exhausted,
    escalated as a loud error — an orphaned GPU bills until someone notices.

    `record(kind, message, data=None)` is how one line of what happened is
    written wherever the caller's history lives: a job's event log for
    `_teardown`, the reconciler's log for the machine-lifetime pass (issue
    #61). The retry/escalation and consecutive-absence rules live here,
    defined once, so the two cannot drift about what a destroy looks like or
    when it is confirmed (ADR-0057).

    Returns True when the machine was confirmed absent across consecutive
    observations, False when the confirmation window expired with the machine
    still listed (the STRAY case). Callers that only care about the side
    effect — `_teardown` — ignore it; the reconciler records it.
    """
    last_exc: Exception | None = None
    destroyed = False
    for attempt in range(DESTROY_ATTEMPTS):
        try:
            provider.destroy(machine.machine_id)
            record("log", f"Machine {machine.machine_id} destroyed")
            destroyed = True
            break
        except Exception as e:
            last_exc = e
            record(
                "error",
                f"Destroy attempt failed: {e} (attempt {attempt + 1}/{DESTROY_ATTEMPTS})",
            )
            if attempt < DESTROY_ATTEMPTS - 1:
                time.sleep(DESTROY_RETRY_DELAY_S)
    if not destroyed:
        record(
            "error",
            f"Machine {machine.machine_id} destroy refused after "
            f"{DESTROY_ATTEMPTS} attempts: {last_exc}; still billing — "
            f"manual removal required",
        )
    # Confirmation: require consecutive absent observations. `destroying`
    # is not absent — it is still billing and still present, so it resets
    # the counter rather than being counted as a stray.
    consecutive_absent = 0
    start = time.time()
    while time.time() - start < TEARDOWN_CONFIRM_TIMEOUT_S:
        try:
            machines = provider.list_machines()
            match = next(
                (m for m in machines if m.machine_id == machine.machine_id),
                None,
            )
            if match is None:
                status = "ABSENT"
            else:
                status = normalize_status(getattr(match, "status", None))

            if status == "ABSENT":
                consecutive_absent += 1
                if consecutive_absent >= TEARDOWN_CONFIRM_SAMPLES:
                    record(
                        "log",
                        f"Teardown confirmed: machine {machine.machine_id} "
                        f"absent in {TEARDOWN_CONFIRM_SAMPLES} consecutive listings",
                    )
                    return True
            elif status == "destroying":
                record(
                    "log",
                    f"Machine {machine.machine_id} still destroying — "
                    f"not yet confirmed",
                )
                consecutive_absent = 0
            else:
                consecutive_absent = 0
        except Exception as e:
            record(
                "error",
                f"Could not confirm teardown of machine {machine.machine_id}: {e}",
            )
            consecutive_absent = 0
        time.sleep(TEARDOWN_CONFIRM_INTERVAL_S)
    record(
        "error",
        f"STRAY MACHINE {machine.machine_id} still listed — "
        f"destroy it manually, it is billing",
    )
    return False


def _teardown(provider: Provider, job_id: str, machine) -> None:
    """Destroy the job's machine, then confirm it independently.

    A destroy call that returns cleanly is a claim. The evidence is the machine
    no longer being listed, and a machine that is still listed is billing right
    now — so it is reported as an error an operator cannot miss. The retry,
    escalation and consecutive-absence confirmation rules live in
    `confirmed_destroy`, defined once and shared with the reconciler (issue
    #61) so the two cannot drift (ADR-0057); this wrapper records the outcome
    on the job's own event log.
    """

    def record(kind: str, message: str, data: dict | None = None) -> None:
        db.add_event(job_id, kind, message, data)

    confirmed_destroy(provider, machine, record)


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
    *,
    resolved_hp: dict | None = None,
    availability_override: Sequence | None = None,
    resume_checkpoint: dict | None = None,
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
    # money path is a hope). The policy (which switches permit which faults)
    # is the one in `config`. When the surface is on, the injected fault is
    # named in the run's own history, so a deliberately broken run can never
    # be mistaken for a real one.
    fault_spec = fault_surface.from_hyperparameters(job.get("hyperparameters"))
    if fault_spec is not None:
        name = fault_spec.get("name")
        if not isinstance(name, str) or not fault_surface.is_known(name):
            raise OrchestratorError(
                "fault_unknown",
                f"'{name}' is not a fault the surface knows; the job was "
                "refused rather than run under a fault nobody can explain.",
            )
        problem = fault_surface.spec_error(fault_spec)
        if problem is not None:
            raise OrchestratorError("fault_invalid", problem)
        refusal = config.fault_surface_refusal(name)
        if refusal is not None:
            raise OrchestratorError(refusal["code"], refusal["message"])
        db.add_event(
            job_id,
            "log",
            f"Simulated fault injected: {name} - "
            f"{fault_surface.describe(name)}. This run is deliberately "
            "broken and cannot be mistaken for a real one.",
        )

    dataset = db.require_dataset(job["dataset_id"])
    from . import admission as _admission

    model = _admission.get(job["base_model"]) or catalog.get(
        catalog.DEFAULT_MODEL
    )
    if model is None:
        # Unreachable while creation validates against this catalog, but a
        # re-read that outlives its guard fails by name on a money path.
        raise OrchestratorError(
            "unknown_model",
            f"Model '{job['base_model']}' is not in the catalog or admitted models.",
        )
    enable_thinking = bool(dataset.get("enable_thinking"))
    revision = job.get("base_revision") or model.revision
    facts = models.resolve(model.repo, revision)
    # Issue #35: a memory recovery's retried attempt arrives with its spec
    # already resolved by the escalator (the frozen spec, transformed with
    # the effective batch preserved). The first attempt resolves from the
    # frozen record exactly as before; nothing here chooses a value, so the
    # trainer's "applies values, resolves nothing" contract is unchanged.
    if resolved_hp is not None:
        hp = resolved_hp
    else:
        hp = hyperparams.effective(job["hyperparameters"] or {})

    # The decisions the launch committed to (issue #79), re-applied here so
    # provisioning honours them rather than silently re-picking the
    # predictor's cheapest configuration: a run that provisioned something
    # other than what the plan froze would be lying about what it did. An
    # override that is no longer available at provisioning fails here, by
    # name, rather than being silently dropped. A retried attempt keeps the
    # user's hardware pins (they are the user's, not the recovery's to
    # override) and pins the method the first attempt ran -- a memory
    # recovery must not quietly switch an adapter run to a full fine-tune.
    frozen_overrides = job.get("overrides") or []
    resolved_overrides = overrides.resolve(
        hp,
        [overrides.from_dict(d) for d in frozen_overrides],
    )
    sequence_len = int(hp[memory_retry.SEQUENCE_LEN_KEY])
    if resolved_hp is not None:
        method_pin = resolved_overrides.method or job.get("method")
    else:
        method_pin = resolved_overrides.method

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
        # Issue #35: a hardware-rung retry narrows the provider's rows to
        # cards with strictly more memory than the one that OOMed -- the
        # only thing that can help once every in-place reduction is spent.
        # Every other attempt reads the provider's availability as usual.
        availability = (
            availability_override
            if availability_override is not None
            else provider.gpu_availability()
        )
        plan = selection.select_hardware(
            facts,
            lora_r=hp["lora_r"],
            sequence_len=sequence_len,
            micro_batch_size=hp["micro_batch_size"],
            availability=availability,
            currency=provider.currency(),
            method=method_pin,
            gpu_type=resolved_overrides.gpu_type,
            device_count=resolved_overrides.device_count,
        )
    except selection.NoFittingHardwareError as e:
        raise OrchestratorError("provider_capacity_unavailable", str(e)) from e
    # Issue #46: the plan chose the machine, so the price it will bill at is
    # now known -- and only now can the spend ceiling become a deadline. The
    # ceiling is a cost, and a cost is enforced against the job's own frozen
    # rate, not against a fixed number of minutes: an expensive machine is
    # stopped sooner than a cheap one at the same ceiling. `with_spend`
    # refuses an unenforceable combination before anything is provisioned.
    limits = limits.with_spend(
        SPEND_CEILING_MINOR, plan.price_per_hour, plan.currency
    )
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
    # Issue #60: a resumed attempt ships its last checkpoint to the machine
    # before the container runs. The archive is streamed straight out of
    # storage -- the object is already a tar, the shape the trainer's
    # uploader wrote -- so nothing is held whole (ADR-0010), and the remote
    # script extracts it under the trainer's own output_dir before axolotl
    # starts. The step names the container path axolotl resumes from.
    resume_from_checkpoint: str | None = None
    if resume_checkpoint is not None:
        step = resume_checkpoint.get("step")
        # The checkpoint a resumption comes back to always carries the step it
        # is named by (the discovery that produced it records one); a record
        # that somehow lacks it is refused rather than resumed from a path
        # nobody could have named.
        if not isinstance(step, int):
            raise OrchestratorError(
                "resume_invalid",
                "The checkpoint selected for resumption carried no step, so "
                "it cannot be resumed from.",
            )
        resume_from_checkpoint = resume_logic.resume_path(step)
        provider.push_stream(
            machine,
            storage.STORE.get_stream(resume_checkpoint["key"]),
            "/tmp/checkpoint.tar",
        )

    # Checked between stages as well as inside the stream: the guard below can
    # only notice the ceiling while lines are arriving, and everything above
    # this line happened before any line existed. The spend ceiling (issue
    # #46) is checked beside the duration ceiling at the same boundaries, so a
    # machine that costs more than the cap while the control plane is still
    # setting it up is stopped like any other runaway.
    limits.check_duration()
    limits.check_spend()
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
    # Issue #74: the delivery formats the launch asked for each get their own
    # scoped write grant, one object per format, minted the same way and with
    # the same lifetime. The key for each is derived from the format and the
    # archive name the trainer writes; the machine PUTs the produced format to
    # it, and this side verifies what landed against the trainer's checksum.
    delivery_request = job.get("delivery_request") or []
    delivery_grants: list[tuple[str, storage.WriteGrant]] = []
    for fmt_id in delivery_request:
        if fmt_id == delivery.DELIVERY_FORMAT_ADAPTER:
            continue
        archive = _delivery_archive_name(fmt_id)
        key = storage.delivery_key(job_id, fmt_id, archive)
        delivery_grants.append(
            (fmt_id, storage.STORE.mint_write_grant(key, remaining))
        )
    script = _remote_script(
        job,
        model,
        enable_thinking,
        reference,
        plan.method,
        artifact_grant=grant,
        checkpoint_grants=checkpoint_grants,
        delivery_grants=delivery_grants,
        resolved_hp=resolved_hp,
        resume_from_checkpoint=resume_from_checkpoint,
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
        # a user otherwise sends them to read the wrong logs. A run that
        # produced no result document at all (NO_RESULT_CODE) is an
        # interruption (issue #60), named as such, so the resumption path can
        # recover it rather than reading it as a bare training failure.
        code = result.get("error_code") or "training_failed"
        if code == NO_RESULT_CODE:
            raise OrchestratorError(
                resume_logic.INTERRUPTED_CODE,
                resume_logic.INTERRUPTED_MESSAGE,
            )
        raise OrchestratorError(
            code,
            result.get("error") or "Training did not complete.",
        )

    # Cancellation is honoured to the last moment an adapter could appear, not
    # only while the stream is open. The machine writes its artifact directly
    # now, so by the time the result is parsed the object may already be in
    # storage: a request answered with "no adapter will be produced" that then
    # produced one would be the single worst thing this path could tell a user,
    # so what already landed is discarded rather than kept.
    return _finalize_packaging(job_id, result, plan.method)


def _finalize_packaging(
    job_id: str, result: dict, method: str | None
) -> tuple[str, str, dict]:
    """Verify a finished run's outputs from storage and produce the terminal
    outcome.

    Shared by the fresh path (`_attempt`) and a packaging recovery (issue
    #68): the machine already wrote its artifact, checkpoints and delivery
    formats to storage before reporting (ADR-0009, #37, #74), so collection is
    verify-and-record, safe to run once or again. The result manifest is
    persisted the moment packaging begins -- not only at the terminal state --
    so a recovery re-verifies the same run rather than re-running it: a worker
    killed during collection would otherwise take the run's own manifest with
    it, the one record that ties the stored artifact to its checksum.

    Cancellation is honoured to the last moment an adapter could appear: an
    upload that landed before the cancellation was seen is discarded rather
    than left readable as the deliverable.
    """
    cancelled = _cancellation_check(job_id)
    _discard_if_cancelled(job_id, cancelled)
    db.set_state(job_id, "packaging", "Verifying artifact", result_json=result)
    artifact = _collect_artifact(job_id, result, method)
    checkpoints = _collect_checkpoints(job_id, result)
    db.set_checkpoints(job_id, checkpoints)
    # The choice is recorded beside the verified checkpoints, not derived at
    # download time (issue #62): a run's answer must not move under retention
    # or a rule edit, so the chosen step and its reason are frozen here.
    _record_best_checkpoint(job_id, checkpoints)
    # Issue #74: verify the delivery formats the machine wrote, one object per
    # format, and record the verdicts -- a format whose bytes did not land or
    # do not match is never served as if it were complete.
    delivery_records = _collect_delivery(job_id, result)
    if delivery_records:
        db.set_delivery(job_id, delivery_records)
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


def _attempt_spec(job: dict, resolved_hp: dict | None) -> dict:
    """The memory-relevant spec one attempt ran, for the attempts record.

    The frozen record is the user's request; a memory recovery's retried
    attempt runs a transformed spec. The record keeps the three values that
    can change between attempts -- per-step batch, accumulation and sequence
    length -- so "what changed" is visible without duplicating the whole
    spec, and the effective batch is carried beside them.
    """
    hp = resolved_hp or hyperparams.effective(
        job["hyperparameters"] or {}, method=job.get("method")
    )
    return {
        memory_retry.MICRO_BATCH_KEY: hp[memory_retry.MICRO_BATCH_KEY],
        memory_retry.ACCUMULATION_KEY: hp[memory_retry.ACCUMULATION_KEY],
        memory_retry.SEQUENCE_LEN_KEY: hp[memory_retry.SEQUENCE_LEN_KEY],
        "method": job.get("method"),
    }


def _recovery_record(step: memory_retry.MemoryRetryStep | None) -> dict | None:
    """The structured record of one memory-recovery step, or None.

    What changed (per key, old -> new), which rung fired, and the preserved
    effective batch -- the half of "told the user what changed" that is
    queryable rather than prose.
    """
    if step is None:
        return None
    return {
        "rung": step.rung,
        "action": step.action,
        "changed": {k: [old, new] for k, (old, new) in step.changed.items()},
        "effective_batch": step.effective_batch,
        "notes": list(step.notes),
    }


def _memory_recovery_message(
    step: memory_retry.MemoryRetryStep, attempt_no: int
) -> str:
    """The user-facing narrative for one automatic memory recovery.

    Says what changed, what was already in force (the notes), and -- the
    distinction ADR-0055 depends on -- that this is a memory recovery, not a
    divergence retry.
    """
    parts = [
        f"Out of device memory during attempt {attempt_no}. Automatically "
        f"retrying on a fresh machine: {step.action}.",
        *step.notes,
        "This is a memory recovery: the effective batch is preserved, and it "
        "is not a divergence retry (a diverging run offers a half-rate retry "
        "as a choice instead).",
    ]
    return " ".join(parts)


def _bigger_than(provider: Provider, capacity_gb: float) -> list:
    """The provider's rows for cards with strictly more memory than `capacity_gb`.

    The hardware rung of the escalation ladder: once every in-place reduction
    is spent (batch, gradient checkpointing, sequence length), the only thing
    that can help is a card the failed one cannot be. An empty list means the
    ladder is exhausted and the job surfaces.
    """
    return [
        row
        for row in provider.gpu_availability()
        if gpus.CAPACITY_GB.get(row.gpu_type, 0) > capacity_gb
    ]


def _memory_surface(
    job_id: str, attempts: list[dict]
) -> tuple[str, str, dict]:
    """The terminal outcome when the escalation ladder or the retry cap is
    exhausted: the job fails with the stable code and the attempts recorded.

    The criterion is that exhausting the retries fails the job with the
    reason *and the attempts recorded* -- the attempts list is already
    persisted by the loop, so the surface says how many were made and names
    the code a client can branch on. This is a memory failure, never a
    divergence abort and never a bare `training_failed`.
    """
    message = memory_retry.MEMORY_RETRIES_EXHAUSTED_MESSAGE
    db.add_event(
        job_id,
        "error",
        message,
        {
            "code": memory_retry.MEMORY_RETRIES_EXHAUSTED_CODE,
            "attempts": len(attempts),
        },
    )
    return (
        "failed",
        message,
        {
            "error_code": memory_retry.MEMORY_RETRIES_EXHAUSTED_CODE,
            "error_message": message,
        },
    )


def _attempt_rate(job: dict) -> dict | None:
    """The billing rate one attempt's machine was provisioned at, or None.

    The rate is frozen onto the job row at provisioning (`set_state` with
    `price_per_hour` and `currency`), and `_record_attempt` runs after the
    attempt has provisioned -- so the row holds the *just-provisioned* rate
    when the attempt's record is written, which is the coupling that makes
    each attempt's `rate` its own machine's. A resumed attempt re-provisions
    and overwrites the row before it is recorded, so its `rate` is the fresh
    machine's, not the interrupted one's. An attempt that failed before
    provisioning records None -- honest absence, never a guessed rate.
    """
    price = job.get("price_per_hour")
    currency = job.get("currency")
    if isinstance(price, (int, float)) and isinstance(currency, str):
        return {"price_per_hour": float(price), "currency": currency}
    return None


def _record_attempt(
    job_id: str,
    attempts: list[dict],
    attempt_no: int,
    machines: list,
    outcome: str,
    error_code: str | None,
    resolved_hp: dict | None,
    recovery: dict | None = None,
    resumed_from: int | None = None,
) -> None:
    """Persist one attempt's record: what it ran, what it ended as, and (for
    a memory-failed attempt) what the recovery changed for the next one.

    This is the "attempts recorded" half of the retry-cap criterion (issue
    #35) and the shape a resumption ticket (#60) reads: each attempt
    carries its own machine, its own rate, its own spec and its own outcome,
    so the history says what actually happened rather than one continuous
    run. A resumed attempt names the step it came back to (`resumed_from`),
    and an interrupted attempt's `outcome` is `interrupted`, never `failed`
    -- a run cut off is not a run that failed on its own terms.
    """
    job = db.require_job(job_id)
    record: dict = {
        "attempt": attempt_no,
        "outcome": outcome,
        "error_code": error_code,
        "machine_id": machines[-1].machine_id if machines else None,
        "rate": _attempt_rate(job),
        "spec": _attempt_spec(job, resolved_hp),
        "recovery": recovery,
    }
    if resumed_from is not None:
        record["resumed_from"] = resumed_from
    attempts.append(record)
    db.set_attempts(job_id, attempts)


def _resume_surface(
    job_id: str, attempts: list[dict], checkpoint_record: dict
) -> tuple[str, str, dict]:
    """The terminal outcome when resumption is exhausted: the job fails with
    the stable code and the attempts recorded.

    A checkpoint survived but the run kept being interrupted past the cap
    (issue #60), so this is not a bare `interrupted` (which is what a job
    with nothing to resume from fails with) and never a `training_failed`:
    the run was cut off, and the record says how many times and from what
    step rather than leaving it to inference.
    """
    message = resume_logic.RESUME_EXHAUSTED_MESSAGE
    db.add_event(
        job_id,
        "error",
        message,
        {
            "code": resume_logic.RESUME_EXHAUSTED_CODE,
            "attempts": len(attempts),
            "step": checkpoint_record.get("step"),
        },
    )
    return (
        "failed",
        message,
        {
            "error_code": resume_logic.RESUME_EXHAUSTED_CODE,
            "error_message": message,
        },
    )


def _resumed_step(resume_checkpoint: dict | None) -> int | None:
    """The step a current attempt resumed from, or None for the first one."""
    if resume_checkpoint is None:
        return None
    return resume_checkpoint.get("step")


# The stable code and sentence a reclaimed job's first recovery event carries.
# The reclaim event itself is written by ``db.claim_next_job`` (which owns the
# claim vocabulary); this is the orchestrator's own narration when it actually
# performs the recovery, and the two say the same thing from their own sides.
RECLAIM_RECOVERY_CODE = "durable_recovery"


def _resolved_hp_from_attempts(job: dict) -> dict | None:
    """The hyperparameter spec the last attempt ran, reconstructed for a
    reclaimed job (issue #68), or None when the job has no attempts.

    A reclaimed job's new driver must resume with the spec the interrupted
    attempt was actually running -- a memory recovery's halved batch is the
    job's spec now, not the frozen request's -- or the resumption would
    quietly change the optimisation, the exact failure ADR-0069 exists to
    prevent. The attempts record keeps only the memory-relevant keys
    (per-step batch, accumulation, sequence length) plus the method, so the
    frozen effective spec is taken and those keys overlaid. When the last
    attempt ran the frozen spec unchanged, this returns None so the fresh
    path re-resolves exactly as before (a reconstructed-but-equal dict would
    otherwise change the method pinning in `_attempt` for no reason).
    """
    attempts = job.get("attempts") or []
    if not attempts:
        return None
    last_spec = attempts[-1].get("spec") or {}
    method = last_spec.get("method") or job.get("method")
    frozen = hyperparams.effective(
        job.get("hyperparameters") or {}, method=method
    )
    overlay = dict(frozen)
    for key in (
        memory_retry.MICRO_BATCH_KEY,
        memory_retry.ACCUMULATION_KEY,
        memory_retry.SEQUENCE_LEN_KEY,
    ):
        if key in last_spec:
            overlay[key] = last_spec[key]
    if all(
        overlay.get(k) == frozen.get(k)
        for k in (
            memory_retry.MICRO_BATCH_KEY,
            memory_retry.ACCUMULATION_KEY,
            memory_retry.SEQUENCE_LEN_KEY,
        )
    ):
        return None
    return overlay


def _recover_abandoned(
    provider: Provider,
    job_id: str,
    machines: list,
    job: dict,
    status: str,
) -> tuple[str, str, dict]:
    """Drive an abandoned job -- one reclaimed after its worker died (issue
    #68) -- to its next state.

    Returns an outcome tuple when the run had already finished training and
    is re-collected from storage (``packaging``); raises the interruption
    error otherwise, so the caller's resumption loop (issue #60) discovers
    the surviving checkpoints and resumes on a fresh machine. The recorded
    machine is appended to `machines` when it is still listed, so the loop's
    per-attempt ``finally`` tears it down once before anything new provisions
    -- recovery never stacks machines, which is how "no second machine
    appears as a result of recovery" is kept true when the machine was
    already created. A machine no longer listed is logged as already gone and
    not appended, so a ghost is never issued a second destroy.

    This is the durable half of the two-recovery split Spec 010 draws: the
    reconciler protects the money (it would destroy this job's machine only
    if the job stopped owning it), and this path recovers the job. The two
    address different failures and both stay.
    """
    machine_id = job.get("machine_id")
    if machine_id is not None:
        machine = Machine(machine_id=machine_id)
        if any(m.machine_id == machine_id for m in provider.list_machines()):
            db.add_event(
                job_id,
                "log",
                f"Worker driving this job was lost; its machine "
                f"{machine_id} will be destroyed before resuming",
            )
            machines.append(machine)
        else:
            db.add_event(
                job_id,
                "log",
                f"Worker driving this job was lost; its machine "
                f"{machine_id} is already gone",
            )

    db.add_event(
        job_id,
        "log",
        f"Resuming job from '{status}' after its driver was lost",
        {"code": RECLAIM_RECOVERY_CODE, "step": status},
    )

    if status == "packaging":
        # Training finished; only collection remains. The result manifest was
        # persisted when packaging began (it is the record of what the run
        # produced), so re-verifying from storage and finishing is safe and
        # exact: nothing is re-run, only re-collected. A manifest that is
        # somehow missing (a row predating the persistence) leaves nothing to
        # re-collect against, so the run is surfaced as interrupted rather
        # than guessed at.
        result = job.get("result")
        if not isinstance(result, dict):
            raise OrchestratorError(
                resume_logic.INTERRUPTED_CODE,
                resume_logic.INTERRUPTED_MESSAGE,
            )
        return _finalize_packaging(job_id, result, job.get("method"))

    # preparing / training: the run was cut off mid-flight. What survived is
    # in the checkpoint slots; raising the interruption error hands the job to
    # the same resumption loop a live worker's interrupted run uses (issue
    # #60), so the two recoveries compose rather than duplicating each other.
    raise OrchestratorError(
        resume_logic.INTERRUPTED_CODE,
        resume_logic.INTERRUPTED_MESSAGE,
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
    # Bind the request's correlation identifier before anything logs, so
    # every structured line this job emits carries the same identifier the
    # request's error response will. A missing correlation is an honest absence
    # for pre-migration rows, not a redaction.
    _bind_correlation_from_job(job_id)
    cid = get_correlation_id()
    logger.info(
        "job starting",
        job_id=job_id,
        correlation_id=cid,
        machine_id=None,
    )

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
            logger.error(
                "job failed before provisioning",
                job_id=job_id,
                correlation_id=get_correlation_id(),
                error_code=e.code,
                exc_info=e,
            )
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
            logger.exception(
                "job failed before provisioning: unexpected error",
                job_id=job_id,
                correlation_id=get_correlation_id(),
            )
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
        # Issue #68: whoever drives this job owns it. Stamp the claim lease
        # (the worker already did when it claimed; this also makes a direct
        # driver -- tests, the boot-time seed -- own the job), then seed the
        # durable state from the row: a job reclaimed after its driver died
        # must continue its attempts, its resume bounds and its recovery spec
        # rather than start over.
        db.touch_claim(job_id)
        job = db.require_job(job_id)
        attempts = list(job.get("attempts") or [])
        # Issue #60's resume bounds and issue #35's memory-retry bounds are
        # carried over from the attempts record so a reclaimed job does not
        # reset them: the caps bound the job's total resumptions and retries,
        # not the work of one driver. A resumed attempt is one that recorded
        # `resumed_from`; a memory recovery is one that recorded `recovery`.
        resumptions_used = sum(
            1 for a in attempts if a.get("resumed_from") is not None
        )
        retries_used = sum(
            1 for a in attempts if a.get("recovery") is not None
        )
        # The spec the last attempt actually ran (a memory recovery's halved
        # batch, not the frozen request) -- None when it ran the frozen spec.
        resolved_hp = _resolved_hp_from_attempts(job)

        # Issue #68: whether this drive's first attempt recovers an abandoned
        # job from its recorded step rather than starting fresh. A reclaimed
        # job resumes where its dead driver stopped: the recovery either
        # returns a terminal outcome (packaging: the run finished training and
        # is re-collected from storage) or raises the interruption error this
        # loop's own handler turns into a checkpoint resumption (preparing /
        # training: issue #60) -- the two recoveries compose rather than
        # duplicating each other.
        reclaim_status: str | None = job.get("status")
        if reclaim_status in ("queued", "provisioning"):
            # No machine is recorded to resume from; run the normal attempt.
            reclaim_status = None
        # Issue #35: an out-of-memory failure retries **automatically**,
        # climbing the escalation ladder one rung per retry (halve the
        # per-step batch, gradient checkpointing, sequence length, more
        # capable hardware), each retry on its own machine -- torn down
        # before the next one provisions -- until the ladder or the retry
        # cap is exhausted and the failure is surfaced with the attempts
        # recorded. Every other failure, and a cancellation, ends the job
        # exactly as before. The retry being automatic is precisely what
        # distinguishes it from a divergence retry (ADR-0055), which is a
        # *choice*: a diverging run usually means the data or the rate is
        # wrong, while a memory retry preserves the effective batch and so
        # cannot change the optimisation. The two decisions never look
        # alike in the record or the database (the codes are asserted
        # disjoint by a core test).
        escalator: memory_retry.MemoryEscalator | None = None
        availability_override: Sequence | None = None
        last_rung: str | None = None
        # Issue #60: the checkpoint the current attempt came back from
        # (None for the first). The count is above, seeded from the row.
        resume_checkpoint: dict | None = None
        outcome: tuple[str, str, dict] | None = None
        while True:
            attempt_no = len(attempts) + 1
            attempt_started = time.time()
            try:
                if reclaim_status is not None:
                    # First attempt of a reclaimed job: recover it from its
                    # recorded step. preparing/training raise the interruption
                    # error handled below (the machine is in `machines` for
                    # the attempt record and is torn down by this iteration's
                    # finally); packaging returns a terminal outcome, which is
                    # recorded and breaks exactly like a completed `_attempt`.
                    pending_status = reclaim_status
                    reclaim_status = None
                    outcome = _recover_abandoned(
                        provider,
                        job_id,
                        machines,
                        job,
                        pending_status,
                    )
                else:
                    outcome = _attempt(
                        provider,
                        job_id,
                        machines,
                        limits,
                        models,
                        resolved_hp=resolved_hp,
                        availability_override=availability_override,
                        resume_checkpoint=resume_checkpoint,
                    )
                _record_attempt(
                    job_id,
                    attempts,
                    attempt_no,
                    machines,
                    outcome[0],
                    None,
                    resolved_hp,
                    resumed_from=_resumed_step(resume_checkpoint),
                )
                break
            except Cancelled as e:
                # No error code and no error message: the user's own decision
                # is not a defect, and a `cancelled` job carrying an error
                # code would be read as one by every client that branches on
                # codes.
                #
                # The machine writes its artifact directly, so an upload that
                # landed before the cancellation was seen is discarded here
                # rather than left readable as the deliverable. Checkpoints
                # are discarded with it (issue #37): a job the user stopped
                # keeps no recovery material.
                _delete_stored_artifact(job_id)
                _delete_stored_checkpoints(job_id)
                _record_attempt(
                    job_id,
                    attempts,
                    attempt_no,
                    machines,
                    "cancelled",
                    None,
                    resolved_hp,
                    resumed_from=_resumed_step(resume_checkpoint),
                )
                outcome = ("cancelled", str(e), {})
                break
            except OrchestratorError as e:
                if e.code == resume_logic.INTERRUPTED_CODE:
                    # Issue #60: a run that ended without a result document is
                    # an interruption, not a training failure. What survived
                    # the machine is in the checkpoint slots (issue #37 wrote
                    # them off as training produced them); what can be parsed
                    # back and verified is recorded, and the job resumes from
                    # the latest on a fresh machine -- restoring the optimiser,
                    # scheduler and step position, the whole checkpoint
                    # directory, not only the weights. Bounded: a job the
                    # infrastructure keeps interrupting surfaces with the
                    # attempts recorded rather than billing forever.
                    discovered = _discover_interrupted_checkpoints(job_id)
                    db.set_checkpoints(job_id, discovered)
                    _record_best_checkpoint(job_id, discovered)
                    source = resume_logic.latest_checkpoint(discovered)
                    _record_attempt(
                        job_id,
                        attempts,
                        attempt_no,
                        machines,
                        "interrupted",
                        e.code,
                        resolved_hp,
                        resumed_from=_resumed_step(resume_checkpoint),
                    )
                    if source is None:
                        # Nothing survived to resume from. If the run never
                        # reached training -- a `preparing` job reclaimed after
                        # its driver died, whose machine was still being set up
                        # so nothing was produced -- restart the attempt rather
                        # than fail a run that never started (issue #68). A run
                        # that did train and lost everything surfaces as
                        # interrupted, as before.
                        current = db.get_job(job_id)
                        if (
                            current is not None
                            and current.get("status") == "preparing"
                        ):
                            db.add_event(
                                job_id,
                                "log",
                                "Run was interrupted before training began; "
                                "starting a fresh attempt",
                                {"code": RECLAIM_RECOVERY_CODE},
                            )
                            continue
                        # The job fails with the honest interruption code and
                        # the attempts recorded, rather than pretending a
                        # resumption was possible when no checkpoint exists to
                        # come back to.
                        outcome = (
                            "failed",
                            str(e),
                            {
                                "error_code": e.code,
                                "error_message": str(e),
                            },
                        )
                        break
                    if not resume_logic.may_resume(resumptions_used):
                        outcome = _resume_surface(job_id, attempts, source)
                        break
                    resumptions_used += 1
                    resume_checkpoint = source
                    db.add_event(
                        job_id,
                        "log",
                        f"Run interrupted after reaching step "
                        f"{source['step']}; resuming from checkpoint step "
                        f"{source['step']} on a new machine",
                        {
                            "code": resume_logic.RESUME_RECOVERY_CODE,
                            "step": source["step"],
                            "attempt": attempt_no,
                        },
                    )
                    db.set_state(
                        job_id,
                        "training",
                        f"Resuming from checkpoint step {source['step']} on a "
                        "new machine",
                    )
                    continue
                is_memory = e.code in memory_retry.MEMORY_FAILURE_CODES
                if is_memory:
                    if retries_used >= memory_retry.MEMORY_RETRY_CAP:
                        # The retry cap is exhausted: every permitted retry is
                        # spent, so the job surfaces with the attempts recorded
                        # rather than failing with a bare out-of-memory code.
                        _record_attempt(
                            job_id,
                            attempts,
                            attempt_no,
                            machines,
                            "failed",
                            e.code,
                            resolved_hp,
                            resumed_from=_resumed_step(resume_checkpoint),
                        )
                        outcome = _memory_surface(job_id, attempts)
                        break
                    job = db.require_job(job_id)
                    if escalator is None:
                        # The escalator starts from the frozen spec's
                        # effective values -- the recovery's first decision is
                        # relative to what the user actually launched with --
                        # resolved against the method the first attempt ran,
                        # so the retried spec keeps the method's own defaults
                        # (a full fine-tune's learning rate is the full
                        # fine-tune's, never the adapter's).
                        escalator = memory_retry.MemoryEscalator(
                            resolved_hp
                            or hyperparams.effective(
                                job["hyperparameters"] or {},
                                method=job.get("method"),
                            )
                        )
                    step = escalator.step()
                    _record_attempt(
                        job_id,
                        attempts,
                        attempt_no,
                        machines,
                        "failed",
                        e.code,
                        resolved_hp,
                        recovery=_recovery_record(step),
                        resumed_from=_resumed_step(resume_checkpoint),
                    )
                    if step is None:
                        # Every reduction the ladder can express is spent;
                        # surfacing is the last rung.
                        outcome = _memory_surface(job_id, attempts)
                        break
                    retries_used += 1
                    resolved_hp = step.hyperparameters
                    last_rung = step.rung
                    if step.rung == memory_retry.RUNG_HARDWARE:
                        # Narrow the provider's rows to cards with strictly
                        # more memory than the one that OOMed; when none
                        # exists the ladder is exhausted and the job surfaces
                        # rather than provisioning the same card again.
                        job = db.require_job(job_id)
                        gpu_type = job.get("gpu_type")
                        capacity = (
                            gpus.CAPACITY_GB.get(gpu_type, 0.0)
                            if isinstance(gpu_type, str)
                            else 0.0
                        )
                        availability_override = _bigger_than(
                            provider, capacity
                        )
                        if not availability_override:
                            outcome = _memory_surface(job_id, attempts)
                            break
                    else:
                        availability_override = None
                    db.add_event(
                        job_id,
                        "log",
                        _memory_recovery_message(step, attempt_no),
                        {
                            "code": memory_retry.MEMORY_RECOVERY_CODE,
                            "attempt": attempt_no,
                            "rung": step.rung,
                            "effective_batch": step.effective_batch,
                            "retries_used": retries_used,
                        },
                    )
                    continue
                if (
                    last_rung == memory_retry.RUNG_HARDWARE
                    and e.code == "provider_capacity_unavailable"
                ):
                    # The hardware rung found no bigger card that fits; the
                    # ladder is exhausted, so this surfaces as a memory
                    # exhaustion rather than a capacity failure to retry.
                    _record_attempt(
                        job_id,
                        attempts,
                        attempt_no,
                        machines,
                        "failed",
                        e.code,
                        resolved_hp,
                        resumed_from=_resumed_step(resume_checkpoint),
                    )
                    outcome = _memory_surface(job_id, attempts)
                    break
                if e.code == BUDGET_EXHAUSTED_CODE and machines:
                    # Issue #46's ordered shutdown: checkpoint, terminate,
                    # destroy. The checkpoint runs here, inside the handler and
                    # before the `finally` destroys the machine, so reaching the
                    # ceiling does not also destroy the work -- what the machine
                    # already wrote off itself (issue #37) is recorded and
                    # selected while it can still be.
                    _emergency_checkpoint(provider, machines[0], job_id)
                _record_attempt(
                    job_id,
                    attempts,
                    attempt_no,
                    machines,
                    "failed",
                    e.code,
                    resolved_hp,
                    resumed_from=_resumed_step(resume_checkpoint),
                )
                outcome = (
                    "failed",
                    str(e),
                    {"error_code": e.code, "error_message": str(e)},
                )
                break
            except Exception as e:
                _record_attempt(
                    job_id,
                    attempts,
                    attempt_no,
                    machines,
                    "failed",
                    "internal_error",
                    resolved_hp,
                    resumed_from=_resumed_step(resume_checkpoint),
                )
                outcome = (
                    "failed",
                    f"{type(e).__name__}: {e}",
                    {"error_code": "internal_error", "error_message": str(e)},
                )
                break
            finally:
                # Before the terminal transition and between attempts, on
                # every path including one nobody anticipated: each attempt's
                # machine is destroyed before the next one provisions, so the
                # retry never stacks machines.
                for machine in machines:
                    _teardown(provider, job_id, machine)
                machines.clear()
                db.add_event(
                    job_id,
                    "log",
                    f"Attempt {attempt_no} ended in "
                    f"{time.time() - attempt_started:.0f}s",
                )

        # Every break above sets `outcome`; the loop cannot fall through.
        assert outcome is not None
        state, message, fields = outcome
        # The measured half of issue #77's comparison, frozen the moment the
        # run ends -- the mirror of the quote frozen at launch -- so no run is
        # wasted even before anything consumes the record. Frozen *before*
        # the terminal status is written, so a reader can never observe a
        # terminal job without its actuals beside it.
        _record_actuals(job_id, fields.get("result_json"), state, time.time())
        # Before the terminal transition, so the terminal state stays the
        # last word on the job (the teardown-confirmation test pins that).
        db.add_event(
            job_id,
            "log",
            f"Job finished in {time.time() - wall_started:.0f}s",
        )
        if state == "cancelled":
            # A job cancelled while the result was already persisted at
            # packaging start (issue #68) must not keep that document: the
            # cancel discards the artifact it names, and a `cancelled` job
            # reporting a result it was told to destroy would contradict the
            # answer the button gave (ADR-0003). The terminal transition
            # clears it, so a cancelled job's record says no result was kept.
            fields = {**fields, "result_json": None}
        db.set_state(job_id, state, message, **fields)
        logger.info(
            "job finished",
            job_id=job_id,
            correlation_id=get_correlation_id(),
            state=state,
            duration_s=round(time.time() - wall_started, 1),
        )
    finally:
        if owns_provider:
            provider.close()
            logger.info(
                "provider closed",
                job_id=job_id,
                correlation_id=get_correlation_id(),
            )
