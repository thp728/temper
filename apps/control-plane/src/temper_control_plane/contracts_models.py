"""The published shapes of the API's responses.

These exist because the web client is generated from this API's schema. A
handler returning a dict merge publishes no shape at all: the generator emits
`unknown`, and the interface ends up hand-typing what the contract refused to
-- which is precisely the drift the generated client exists to prevent.

They describe what crosses the HTTP boundary only. Stored rows keep their
storage-seam addresses -- the dataset's object key and the job's artifact
key -- and these models do not publish them: where an object lives is the
seam's business, so neither a page nor a client learns it.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from temper_core import catalog


class ValidationIssue(BaseModel):
    """One problem, named where it is: line number when one applies."""

    line: int | None = None
    code: str
    message: str


class PreviewTurn(BaseModel):
    """One message turn inside a preview row.

    Rows reach preview before validation has judged them, so their turns can
    be malformed in ways the type cannot assume away. Rather than publish
    `unknown` and push casting onto the interface, malformed values are
    preserved as their JSON representation: nothing silently drops, and the
    client renders strings, period.
    """

    model_config = ConfigDict(extra="allow")

    role: str | None = None
    content: str | None = None

    @field_validator("role", "content", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> Any:
        return v if v is None or isinstance(v, str) else json.dumps(v)


class PreviewRow(BaseModel):
    """A parsed row as Temper understood it, before validation judges it.

    Arbitrary keys are allowed because the row is whatever the JSON line held;
    `messages` is typed because it is the key every supported schema shares.
    """

    model_config = ConfigDict(extra="allow")

    messages: list[PreviewTurn] | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def _coerce_turns(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [
                item
                if isinstance(item, dict)
                else {"content": json.dumps(item)}
                for item in v
            ]
        return v


class TokenDistribution(BaseModel):
    """The distribution of token counts across a dataset's rows (issue #42).

    Bounded by construction: `histogram` is one count per `histogram_edges`
    bucket, never one entry per row, so the recorded distribution cannot grow
    with the file -- the same cap that keeps the streaming validator flat.
    `truncated_rows` is the exact number of rows that would be truncated at
    `sequence_len` (the trainer default), counted during the pass rather than
    derived from the histogram afterwards."""

    total_tokens: int
    rows_counted: int
    sequence_len: int
    truncated_rows: int
    max_row_tokens: int
    histogram: list[int]
    histogram_edges: list[int]


class DatasetReport(BaseModel):
    """What validation says about a dataset. The same dict
    `temper_core.validation.Report.to_dict()` produces, published typed.

    The `error_count`/`warning_count` totals are the truth about how broken a
    file is; the `errors`/`warnings` lists are capped at a hundred because the
    report is held in memory and a file broken on every line must not become a
    report proportional to the file. `*_suppressed` reconciles the two, so a
    capped report says what it is not showing.

    `token_count`/`token_distribution` (issue #42) are produced by the
    counting phase that runs after validation, so they are merged into the
    report the API publishes and are null until the phase lands -- the quote
    reads `token_count` and renders the count absent while it is null
    (ADR-0031)."""

    valid: bool
    row_count: int
    usable_rows: int
    schema_type: str | None = None
    enable_thinking: bool | None = None
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]
    preview: list[PreviewRow]
    error_count: int = 0
    warning_count: int = 0
    errors_suppressed: int = 0
    warnings_suppressed: int = 0
    token_count: int | None = None
    token_distribution: TokenDistribution | None = None


class DatasetAccepted(BaseModel):
    """The response to a successful upload: where validation is happening, not
    the report -- the report lands on the dataset's record
    (`GET /v1/datasets/{id}`) when validation finishes, so a large upload can
    be watched rather than waited on."""

    id: str
    filename: str
    status: str


class DatasetImportRequest(BaseModel):
    """A dataset to import by reference (issue #45): a public repository,
    optionally narrowed to a configuration (subset) and a split.

    The response is the same `DatasetAccepted` an upload answers with, and
    validation runs through the identical path -- the reference is only a
    different way for the bytes to arrive, never a shortcut for them. A
    reference that cannot be fetched, or a split that resolves to nothing, is
    refused up front with its reason."""

    repo: str
    config: str | None = None
    split: str | None = None


class ValidationProgress(BaseModel):
    """Where validation has got to while a dataset is `validating`.

    `bytes_read`/`bytes_total` let a page draw a proportion complete; `rows`
    is how many rows have been read. Published so a large upload does not look
    like a frozen page."""

    bytes_read: int = 0
    bytes_total: int | None = None
    rows: int = 0


class DatasetRecord(BaseModel):
    """A stored dataset with its report attached -- including a dataset that
    failed validation, whose report is the reason it was kept. While it is
    `validating`, `progress` says how far validation has got and `report` is
    absent.

    The token fields (issue #42) belong to the counting phase that runs after
    validation, so they ride on the record rather than inside the validation
    dict: `token_count_status` is the phase's own state, typed as the literal
    vocabulary (`counting` | `done` | `failed` | null) so the contract is the
    single source of the state names both halves use, and `counting_progress`
    shows where it has got to while it runs. The count itself is merged into
    `report` once it lands."""

    id: str
    filename: str
    created_at: float
    status: str
    report: DatasetReport | None = None
    progress: ValidationProgress | None = None
    token_count_status: Literal["counting", "done", "failed"] | None = None
    counting_progress: ValidationProgress | None = None


class DatasetList(BaseModel):
    """The response to listing datasets."""

    datasets: list[DatasetRecord]


class PeakMemoryEstimate(BaseModel):
    """The predicted peak VRAM for training a model with its default
    hyperparameters, and the headroom that leaves against its recommended
    card. Published typed rather than folded into `CatalogEntry`'s flat
    fields, because the breakdown -- not just the total -- is what makes the
    prediction an explanation rather than a number to trust blindly."""

    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    overhead_gb: float
    total_gb: float
    trainable_params: int
    tolerance: float
    gpu_type: str
    gpu_capacity_gb: float
    headroom_gb: float


class CatalogEntry(BaseModel):
    """One base model in the curated catalog: what it is, under what terms,
    and pinned to which revision.

    Every field is shown to a user choosing a model -- licence and revision
    are not metadata here; they are part of what the user is agreeing to
    train against."""

    id: str
    repo: str
    revision: str
    params_b: float
    license: str
    license_url: str
    context_length: int
    good_for: str
    min_gpu: str
    peak_memory: PeakMemoryEstimate


class ModelCatalog(BaseModel):
    """The catalog and its default. The default is published beside the list
    because "which one starts selected" is a product decision, not something
    a client should guess by ordering.

    `admitted` (issue #58) are the models probed from outside the catalog,
    each carrying its persisted probe result -- verdict and findings -- so the
    interface shows what the user is taking on before they commit to a launch.
    """

    models: list[CatalogEntry]
    admitted: list[AdmittedModel] = Field(default_factory=list)
    default: str


class ProbeFinding(BaseModel):
    """One compatibility-probe finding (issue #58): the stable code, whether
    it blocks or warns, and the product reason. A `block` means the model is
    not usable; a `warn` means it is usable and the user is told what they
    took on."""

    code: str
    severity: Literal["block", "warn"]
    message: str
    details: dict[str, Any] | None = None


class ProbeMemory(BaseModel):
    """The probe's predicted-memory line (issue #58): the peak at the default
    configuration against the smallest card this platform can provision, with
    the arithmetic shown. `fits` is never False against stale availability,
    because availability is a creation-time concern (#54); this is the same
    `memory.predict_peak` arithmetic the launch refusal is built on."""

    fits: bool
    peak_gb: float | None = None
    card: str | None = None
    capacity_gb: float | None = None
    headroom_gb: float | None = None
    note: str | None = None


class ProbeResult(BaseModel):
    """A persisted compatibility-probe verdict (issue #58), shown rather than
    merely enforced: a model that passes with warnings is usable, and the user
    knows what they took on. `ok` is what a caller branches on for usability;
    `verdict` is the human-facing label derived from it."""

    repo: str
    revision: str
    ok: bool
    verdict: Literal["blocked", "usable_with_warnings", "usable"]
    architecture: str
    params_b: float = 0.0
    context_length: int = 0
    license: str = ""
    is_moe: bool = False
    findings: list[ProbeFinding] = Field(default_factory=list)
    memory: ProbeMemory | None = None


class AdmittedModel(BaseModel):
    """One model admitted from outside the catalog (issue #58), pinned to a
    revision and carrying the persisted probe result that lets a job be
    created against it. A blocked probe is persisted too, so the reason is
    shown rather than retried blindly."""

    id: str
    repo: str
    revision: str
    created_at: float
    probe: ProbeResult


class SurfaceField(BaseModel):
    """One trainer field as the advanced surface classifies it (issue #33).

    Every field carries its tier and the reason it landed there; an exposed
    field additionally carries the specific thing that goes wrong if it is set
    badly (`failure_mode`) and how its value is typed (`type`: the interface
    generates its input from this rather than hand-listing the vocabulary).
    """

    tier: str
    reason: str
    failure_mode: str | None = None
    type: str | None = None


class SurfaceTiers(BaseModel):
    """The three tiers of the generated surface, each a name-to-field map.

    One per Spec 009's three tiers so the shape of the document says what the
    tiers are; a client reads `tiers["exposed_with_named_failure_mode"]` to
    render the overridable settings and `tiers["known_but_unsupported"]` to
    show the trainer's settings this product does not offer, with their
    reasons."""

    calculated: dict[str, SurfaceField] = Field(default_factory=dict)
    exposed_with_named_failure_mode: dict[str, SurfaceField] = Field(
        default_factory=dict
    )
    known_but_unsupported: dict[str, SurfaceField] = Field(
        default_factory=dict
    )


class RuntimeOnlyValidators(BaseModel):
    """The combinations that cannot be known ahead of launch, named as such.

    Axolotl enforces these in arbitrary Python, so a generated form cannot
    know what they will refuse until the job is running (Spec 009). The model
    validators are named; the field-validator names were not captured by the
    introspection, so only their count is recorded."""

    model_validators: list[str] = Field(default_factory=list)
    field_validator_count: int = 0


class AdvancedSurface(BaseModel):
    """The generated advanced surface, published for the interface to render.

    The same document `temper_core.surface.surface_document()` produces (and
    `just contracts` materialises as `advanced-surface.json`), typed at the
    HTTP boundary so the generated client never falls back to `unknown`. The
    metadata fields validate from the document's leading-underscore keys and
    serialize under their plain names: the underscore is a "derived metadata"
    marker internal to the generator, not a shape worth publishing."""

    model_config = ConfigDict(populate_by_name=True)

    source_image: str = Field(validation_alias="_source_image")
    axolotl_version: str = Field(validation_alias="_axolotl_version")
    config_model: str = Field(validation_alias="_config_model")
    known_keys: list[str]
    platform_internal_keys: list[str]
    overrideable_keys: list[str]
    tiers: SurfaceTiers
    counts: dict[str, int]
    runtime_only_validators: RuntimeOnlyValidators


class FeasibilityWarning(BaseModel):
    """The duration-feasibility estimate, published as the dict
    `temper_core.feasibility.warning()` produces. An estimate everywhere it
    appears -- the message says so, so no client can present it as a quote."""

    code: str
    message: str
    estimated_duration_s: float
    max_duration_s: float
    rows_per_second: float


class QuotePhase(BaseModel):
    """One phase's predicted duration and GPU cost, both ranges.

    `duration_low_s`/`duration_high_s` are never equal: a quote shows a range,
    never a point, because the throughput figures behind it are the softest
    numbers in the model. A phase that cannot be estimated (training under a
    `max_steps` cap, where run length is bounded by steps rather than data)
    reports None rather than a guessed number."""

    name: str
    duration_low_s: float | None = None
    duration_high_s: float | None = None
    cost_low_minor: int | None = None
    cost_high_minor: int | None = None


class QuoteDecisionAlternative(BaseModel):
    """One configuration the predictor considered and did not choose, with
    what it would have cost (in memory, or in the account's currency per
    hour) and why it lost. Structured so the API, the interface and the
    artifact manifest read the same record."""

    value: str
    cost: str
    constraint: str


class QuoteDecision(BaseModel):
    """One decision the predictor made on the user's behalf (issue #76): the
    decision, the value chosen, the constraint that forced it, and the
    alternatives with what each would have cost. The shape of an ADR turned
    into a product surface; `alternatives` may be empty when nothing else
    fit, which is itself a reason. `overridden` (issue #79) marks a decision
    the user pinned rather than the predictor made, so the plan can show the
    difference."""

    decision: str
    chosen: str
    constraint: str
    alternatives: list[QuoteDecisionAlternative] = Field(default_factory=list)
    overridden: bool = False


class DecisionOverride(BaseModel):
    """One decision the user pinned instead of the predictor's (issue #79):
    which decision, and the value to pin it to. The value is the decision
    record's own vocabulary -- the same string the plan shows as `chosen` --
    so the control sits beside the explanation it edits."""

    decision: str
    value: str


class Quote(BaseModel):
    """What a job is predicted to cost and how long it should take.

    An estimate everywhere it appears (`is_estimate` and the interface both
    say so), composed per phase rather than as one blended rate, with the cost
    in the account's own currency as an integer in its smallest unit
    (`cost_*_minor`) with the currency code alongside. It pins what it was
    computed against -- `dataset_id`/`dataset_created_at` and `base_revision`
    -- and `expires_at` says how long the prices and availability it reflects
    are honoured for.

    `storage_cost_usd_*` is the separate storage line, deliberately in USD
    rather than the account's currency: ADR-0030 established that no live
    per-account storage price exists to convert it against, and a labelled
    USD figure is more honest than a silently assumed conversion.

    `decisions` are the structured reasons the configuration was chosen
    (issue #76), carried with the quote and frozen with it, so a completed
    job explains itself as completely as a planned one. Each decision's
    `overridden` flag (issue #79) marks the ones the user pinned.
    `override_options` maps each select-style decision to the legal values
    its control can offer (issue #79) -- the interface generates its controls
    from this rather than hand-listing the vocabulary."""

    currency: str
    minor_unit: int
    dataset_id: str
    dataset_created_at: float
    base_revision: str
    token_count: int | None = None
    expires_at: float
    phases: list[QuotePhase]
    duration_low_s: float
    duration_high_s: float
    cost_low_minor: int
    cost_high_minor: int
    storage_cost_usd_per_hour: float
    storage_cost_usd_total_low_minor: int
    storage_cost_usd_total_high_minor: int
    # The predicted per-device peak VRAM, from the same selection that chose
    # the card (issue #77 records it against the measured figure). A point,
    # not a range: memory is the half of the predictor that blocks, so it is
    # arithmetic rather than an estimate.
    peak_memory_gb: float | None = None
    is_estimate: bool = True
    decisions: list[QuoteDecision] = Field(default_factory=list)
    override_options: dict[str, list[str]] = Field(default_factory=dict)


class QuoteRequest(BaseModel):
    """The recompute a plan screen asks for when a decision is overridden
    (issue #79): the same inputs as `GET /v1/quotes`, plus the decisions to
    pin. The response is the recomputed quote, or a coded refusal with the
    arithmetic that refused it -- an override never mutates the plan it
    landed on, it re-requests it.

    `hyperparameters` (issue #80) are the advanced-surface settings the user
    has overridden; they ride on the same recompute so the plan re-prices
    around them and refuses a configuration that no longer fits before it is
    launched."""

    dataset_id: str
    base_model: str = catalog.DEFAULT_MODEL
    overrides: list[DecisionOverride] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class JobSpecPreview(BaseModel):
    """What a launch would train with, before anything is launched.

    The whole point of the launch screen is that nothing is a surprise after
    committing: this is that answer as one published shape. `hyperparameters`
    are the effective specification resolved exactly as the trainer resolves
    overrides -- showing anything else would describe a job the trainer will
    not run.

    `delivery_formats` (issue #74) are the delivery formats a launch may ask
    for beyond the canonical artifact, each with its plain-language purpose --
    read from the one `temper_core.delivery` vocabulary, so the launch screen
    and the finished page describe a format the same way.

    Deliberately quote-free: the plan page renders immediately and fetches the
    quote for the selected model afterwards, because an estimate never blocks
    the surface it appears on (spec 005)."""

    dataset: DatasetRecord
    hyperparameters: dict[str, Any]
    warning: FeasibilityWarning | None = None
    delivery_formats: list[DeliveryFormatOption] = Field(default_factory=list)


class DeliveryFormatOption(BaseModel):
    """One delivery format a launch may ask for, with its plain-language
    purpose. `id` is the request value the launch sends; `what_for` is the
    sentence a user chooses by, defined once in the domain (issue #74)."""

    id: str
    what_for: str


class StageActual(BaseModel):
    """One measured stage of a run: its name and how long the machine spent
    in it, from the job's own state transitions (issue #77). A stage never
    reached -- a job cancelled in provisioning has no `preparing` to measure
    -- records None rather than a guessed number."""

    name: str
    duration_s: float | None = None


class JobActuals(BaseModel):
    """The measured half of issue #77's comparison, frozen at terminal.

    Duration and peak memory are **measured**; the cost lines are **derived**
    from measured duration and the rate the job froze at launch -- nothing in
    the product reads a bill (spec 005's out-of-scope note) -- so the
    interface marks each figure measured or derived wherever both appear.
    `currency` travels with the cost so it is never shown without the unit
    that gives it meaning."""

    duration_s: float | None = None
    peak_memory_gb: float | None = None
    cost_minor: int | None = None
    storage_cost_usd_minor: int | None = None
    currency: str | None = None
    # The measured stages are stages, not phases: the quote prices phases
    # (issue #72) and the machine passes through states, and reconciling the
    # two vocabularies is calibration's job -- calling them the same word
    # would hide that.
    stages: list[StageActual] = Field(default_factory=list)


class ArtifactRecord(BaseModel):
    """The artifact a job produced, as the interface is allowed to see it.

    The deliverable's declared kind and what it consists of (issue #32): the
    artifact is what a user downloads, and an adapter is one kind of it. The
    kind is **derived from the job's method**, never stored, so a row written
    before this model existed reads correctly with no migration. `members` are
    the archive entry names (never the storage keys -- where an object lives
    is the storage seam's business, not the browser's), and `loading` is the
    per-kind load path, defined once in the domain so this shape, the download
    manifest and the docs cannot drift."""

    kind: str
    members: list[str]
    bytes: int | None = None
    loading: str


class DeliveryFormatRecord(BaseModel):
    """One produced delivery format, as the interface is allowed to see it.

    Issue #74: beyond the canonical artifact, a job can produce a merged
    single-file model and a quantised local-inference format. Each is served
    at the artifact route with a `format` query parameter, and each download
    carries a manifest generated from the run record (ADR-0054 flow-through).
    `what_for` is the plain-language purpose a user chooses by, defined once
    in the domain; `kind` is the artifact kind the format is delivered as.
    Like `ArtifactRecord`, this publishes member names and never storage keys.
    """

    format: str
    kind: str
    what_for: str
    members: list[str]
    bytes: int | None = None
    loading: str


class CheckpointRecord(BaseModel):
    """One checkpoint a job recorded, as the interface is allowed to see it.

    Issue #37's verified set, published without the storage addresses: `step`
    is the user-visible name a download is addressed by (the download route
    reads the job row's own record for the key), `slot` is the retention slot
    it landed in, and `held_out_loss` is the signal best-checkpoint selection
    reads. `selected` marks the checkpoint the run recorded as its result
    (issue #62); it is set from the stored choice at presentation, never a
    re-derivation of that choice."""

    step: int
    slot: int | None = None
    loss: float | None = None
    held_out_loss: float | None = None
    verified: bool = False
    superseded: bool = False
    selected: bool = False


class BestCheckpoint(BaseModel):
    """The run's recorded choice of result checkpoint (issue #62).

    Stored once at terminal time and never recomputed: a user who returns a
    week later sees the same answer even if retention evicts a checkpoint or
    the selection rule is edited. `basis` names the rule that decided, and
    `reason` says it in words -- the "why" that is shown with the choice."""

    step: int | None = None
    held_out_loss: float | None = None
    basis: str
    reason: str


class AttemptRecord(BaseModel):
    """One execution of a job (issue #35): its own machine, its own spec and
    its own outcome, so the history says what actually happened rather than
    one continuous run. A memory failure retries automatically and makes
    attempts plural; the attempt that OOMed carries its `error_code` (the
    fault's `simulated_oom` or a real `training_oom`) and the `recovery` --
    which rung fired, what changed, and the effective batch that was
    preserved -- so "the user is told a recovery happened and what changed"
    and "the attempts are recorded" are both queryable, not just narrated.

    `spec` is the memory-relevant slice of the resolved spec that attempt ran
    (per-step batch, accumulation, sequence length) -- the frozen record is
    the user's request, and the retried attempts are the platform's answer to
    a failure."""

    attempt: int
    outcome: str
    error_code: str | None = None
    machine_id: int | None = None
    spec: dict[str, Any] | None = None
    recovery: dict[str, Any] | None = None


class ComparisonTurn(BaseModel):
    """One message in a comparison prompt (issue #69).

    The prompt is the conversation up to the last user turn -- the held-out
    answer is never fed to either model, so the user is never shown an answer
    the model memorised. Roles and content are strings, the same shape the
    trainer's rows carry, published typed so the interface can render the
    question without hand-typing it."""

    role: str
    content: str


class ComparisonRow(BaseModel):
    """One prompt and both models' answers, as the interface may show them.

    `prompt` is the conversation up to the last user turn; `base` and `tuned`
    are the two models' completions under the recorded decoding settings."""

    prompt: list[ComparisonTurn]
    base: str
    tuned: str


class Comparison(BaseModel):
    """The run's side-by-side comparison (Spec 011 / issue #69).

    Produced on the warm machine after training and published exactly as the
    trainer recorded it. `ok` says whether rows were produced at all; `rows`
    are the per-prompt pair of generations; `decoding` is the fixed set of
    generation settings the comparison was made under, recorded so a reader
    can tell whether two outputs are comparable; and `selection` names which
    checkpoint the tuned side compared -- the run's own selection rule's
    answer, never the last one written. On a failure `reason` names it and
    the artifact is still delivered (evaluation failure never fails the run)."""

    ok: bool
    rows: list[ComparisonRow] = Field(default_factory=list)
    decoding: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    selection: BestCheckpoint | None = None


class JobRecord(BaseModel):
    """A job and the record of what became of it, published typed.

    Like `DatasetRecord` for stored rows, this deliberately does not publish
    the artifact's storage address (`artifact_key`): where a stored object
    lives is the storage seam's business, not the browser's. The artifact
    travels through the download endpoint instead, and `artifact` -- its
    declared kind and members -- is published so the interface can name the
    deliverable without ever touching where it lives. Checkpoints are
    published the same way: `checkpoints` names each retained checkpoint by
    step and loss, `best_checkpoint` is the run's recorded choice, and the
    bytes live behind the checkpoint download route.

    ``retry_from`` (issue #36) links a diverged job's single retry at half
    the learning rate to the job it retries -- the history can name what came
    from what, and the single-retry offer can be offered once.
    """

    id: str
    dataset_id: str
    base_model: str
    base_revision: str | None = None
    hyperparameters: dict[str, Any] | None = None
    status: str
    cancel_requested: bool = False
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    machine_id: int | None = None
    gpu_type: str | None = None
    device_count: int | None = None
    method: str | None = None
    price_per_hour: float | None = None
    currency: str | None = None
    disk_gb: int | None = None
    storage_cost_usd_per_hour: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    warnings: list[FeasibilityWarning] = Field(default_factory=list)
    quote: Quote | None = None
    overrides: list[DecisionOverride] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    actuals: JobActuals | None = None
    artifact: ArtifactRecord | None = None
    # Issue #74: the delivery formats this job produced (beyond the canonical
    # artifact), each with its plain-language purpose. `delivery_request` is
    # the frozen launch-time request (which formats were asked for);
    # `delivery_formats` is the verified set the machine produced. Both default
    # to empty so a pre-delivery row publishes cleanly. The storage keys behind
    # each format are not published -- they live in the job row's own record,
    # read by the download and teardown paths.
    delivery_request: list[str] = Field(default_factory=list)
    delivery_formats: list[DeliveryFormatRecord] = Field(default_factory=list)
    checkpoints: list[CheckpointRecord] = Field(default_factory=list)
    best_checkpoint: BestCheckpoint | None = None
    retry_from: str | None = None
    # The executions of this job (issue #35): a memory failure retries
    # automatically and makes attempts plural, each with its own machine,
    # spec and outcome.
    attempts: list[AttemptRecord] = Field(default_factory=list)
    # Whether the model is a mixture-of-experts -- frozen at creation
    # (issue #65): the label travels with the job so a finished run says
    # what it was trained on, untested here rather than refused.
    is_moe: bool | None = None
    comparison: Comparison | None = None


class JobList(BaseModel):
    """The response to listing jobs."""

    jobs: list[JobRecord]


class JobProgress(BaseModel):
    """One phase's current progress, the latest superseding the previous
    (issue #49).

    `done`/`total` are bytes (image pull aggregates across the layers docker
    pulls in parallel; model download is the single file). `rate` is measured
    live between consecutive readings, in bytes per second, and `eta_s` is the
    seconds remaining derived from it -- so the estimate corrects itself when
    actual throughput differs from whatever came before. `message` is the
    latest raw line that produced the record, so the summary still carries the
    concrete line it came from."""

    phase: str
    done: float | None = None
    total: float | None = None
    rate: float | None = None
    eta_s: float | None = None
    ts: float
    message: str | None = None


class JobOutputLine(BaseModel):
    """One promoted raw line, retained on the job's output record (issue #49).

    The line became a progress record and was deliberately not emitted as an
    event; keeping it here is what makes "nothing is discarded" true. `phase`
    is the phase it was promoted into, so the interface can offer the lines as
    collapsed detail per phase."""

    id: int
    phase: str
    line: str


class JobEvent(BaseModel):
    """One entry in a job's durable history.

    Every state transition appends one, in the same transaction as the
    transition itself. `data` carries the fields promoted from training
    output at read time -- loss, step, epoch -- when the line carried them;
    `message` is always the original line or narration, so structuring the
    numbers never costs the reader the text they arrived in."""

    id: int
    job_id: str
    ts: float
    kind: Literal["state", "metric", "log", "error"]
    message: str | None = None
    data: dict[str, Any] | None = None


class EventPage(BaseModel):
    """A job's durable history: the events, the current per-phase progress, and
    the retained raw lines that were promoted into it (issue #49).

    `after` is what the client holds; `last_id` is where this page reaches, so
    a polling client resumes instead of re-reading or skipping. `progress` is
    the whole per-phase snapshot (it supersedes, so it is small by
    construction); `output` is the whole retained record of promoted lines --
    the collapsed detail that keeps "we keep everything" true.

    `total` is the job's event count regardless of the paging window
    (issue #56): where a limit still applies the interface says what it is
    showing and of how many, and a silent truncation becomes a stated one.
    `total` travels even when the page is not truncated, so the same shape
    answers "how many did this run produce?" without a second request.
    """

    events: list[JobEvent]
    last_id: int
    total: int
    progress: list[JobProgress] = Field(default_factory=list)
    output: list[JobOutputLine] = Field(default_factory=list)


class CalibrationMetric(BaseModel):
    """One metric's rollup across runs (issue #77): how many were compared,
    what was predicted on average, what actually happened, and the spread of
    the error ratio (actual over predicted midpoint; >1 under-predicts, <1
    over-predicts). `min_ratio`/`max_ratio` are where a systematically wrong
    estimate shows up rather than being absorbed into the mean."""

    count: int
    mean_predicted: float | None = None
    mean_actual: float | None = None
    mean_ratio: float | None = None
    min_ratio: float | None = None
    max_ratio: float | None = None


class CalibrationPhase(CalibrationMetric):
    """One measured stage's rollup, plus the quote phases (issue #72) that
    predict it. `preparing` is the SSH wait (readiness); `training` carries
    image_pull + model_download + training, because the on-machine image
    build and the weights download happen inside the orchestrator's
    `training` state."""

    name: str
    quotes_phases: list[str] = Field(default_factory=list)


class RangeComparison(BaseModel):
    """One ranged metric's per-run comparison (duration, cost): the predicted
    range with its midpoint, the measured actual, and where the actual landed
    relative to the range (`under`/`inside`/`over`), with the ratio against
    the midpoint."""

    predicted_low: float | None = None
    predicted_high: float | None = None
    predicted_midpoint: float | None = None
    actual: float | None = None
    ratio: float | None = None
    direction: Literal["under", "inside", "over"] | None = None


class PointComparison(BaseModel):
    """One point metric's per-run comparison (peak memory): the predicted
    point, the measured actual, and the ratio. Peak blocks rather than warns,
    so it is a point, not a range, and there is no `inside` to land in."""

    predicted: float | None = None
    actual: float | None = None
    ratio: float | None = None


class RunComparison(BaseModel):
    """One job's prediction-vs-measurement record, per metric, published
    typed so the interface renders it without hand-declaring the shape."""

    duration: RangeComparison = RangeComparison()
    peak_memory: PointComparison = PointComparison()
    cost: RangeComparison = RangeComparison()


class CalibrationRun(BaseModel):
    """One terminal job in the aggregate, so an outlier can be named rather
    than pointed at: its identity, and its per-metric prediction-vs-measurement
    record."""

    job_id: str
    base_model: str
    status: str
    created_at: float
    comparison: RunComparison = RunComparison()


class Calibration(BaseModel):
    """Predictions against measurements across every terminal job that has
    both (issue #77). Each metric and phase reports its own count, so
    "calibrated against N real runs" is only as honest as N is visible."""

    count: int
    metrics: dict[str, CalibrationMetric]
    phases: list[CalibrationPhase]
    runs: list[CalibrationRun]


class EndpointPreview(BaseModel):
    """What starting an endpoint would cost and when it would stop, before it starts.

    The hourly cost is the job's frozen price (the rate the job was
    provisioned at) and the stop times are now + idle and now + max, both
    domain constants (temper_core.serving). The preview is what the
    interface shows before the user confirms the start, so there is no
    surprise about the bill or the lifetime (spec 011: "the hourly cost and
    the stop time are shown before it starts")."""

    price_per_hour: float
    currency: str
    idle_timeout_s: float
    max_lifetime_s: float
    expires_at: float
    max_expires_at: float


class EndpointRecord(BaseModel):
    """A temporary authenticated endpoint (issue #78).

    The key itself is never published: the creation response carries the
    plaintext once, and every later read carries only the display prefix.
    A key that can be read back out of the store is a finding. The
    endpoint carries its own expiry from the moment it starts, extends on
    use, and stops itself via a timer -- the forgotten warm machine is the
    loudest complaint against the commercial baseline, so stopping itself
    is the feature."""

    id: str
    job_id: str
    status: str
    api_key_prefix: str
    created_at: float
    expires_at: float
    max_expires_at: float
    last_used_at: float
    price_per_hour: float | None = None
    currency: str | None = None
    machine_id: int | None = None
    idle_timeout_s: float | None = None
    max_lifetime_s: float | None = None
    stopped_at: float | None = None
    stop_reason: str | None = None


class EndpointCreated(EndpointRecord):
    """The creation response: the endpoint plus the plaintext key, once."""

    api_key: str


class InferRequest(BaseModel):
    """One prompt for the served model."""

    prompt: str


class InferResponse(BaseModel):
    """The model's completion, plus the refreshed expiry."""

    completion: str
    expires_at: float
    max_expires_at: float
