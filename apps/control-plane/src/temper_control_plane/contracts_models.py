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
    dict: `token_count_status` is the phase's own state (`counting` | `done` |
    `failed` | null), and `counting_progress` shows where it has got to while
    it runs. The count itself is merged into `report` once it lands."""

    id: str
    filename: str
    created_at: float
    status: str
    report: DatasetReport | None = None
    progress: ValidationProgress | None = None
    token_count_status: str | None = None
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
    a client should guess by ordering."""

    models: list[CatalogEntry]
    default: str


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
    landed on, it re-requests it."""

    dataset_id: str
    base_model: str = catalog.DEFAULT_MODEL
    overrides: list[DecisionOverride] = Field(default_factory=list)


class JobSpecPreview(BaseModel):
    """What a launch would train with, before anything is launched.

    The whole point of the launch screen is that nothing is a surprise after
    committing: this is that answer as one published shape. `hyperparameters`
    are the effective specification resolved exactly as the trainer resolves
    overrides -- showing anything else would describe a job the trainer will
    not run.

    Deliberately quote-free: the plan page renders immediately and fetches the
    quote for the selected model afterwards, because an estimate never blocks
    the surface it appears on (spec 005)."""

    dataset: DatasetRecord
    hyperparameters: dict[str, Any]
    warning: FeasibilityWarning | None = None


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


class JobRecord(BaseModel):
    """A job and the record of what became of it, published typed.

    Like `DatasetRecord` for stored rows, this deliberately does not publish
    the artifact's storage address (`artifact_key`): where a stored object
    lives is the storage seam's business, not the browser's. The artifact
    travels through the download endpoint instead."""

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


class JobList(BaseModel):
    """The response to listing jobs."""

    jobs: list[JobRecord]


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
    """A page of history with the last id served.

    `after` is what the client holds; `last_id` is where this page reaches,
    so a polling client resumes instead of re-reading or skipping."""

    events: list[JobEvent]
    last_id: int


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
