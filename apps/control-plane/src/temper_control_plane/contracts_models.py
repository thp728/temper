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


class DatasetReport(BaseModel):
    """What validation says about a dataset. The same dict
    `temper_core.validation.Report.to_dict()` produces, published typed."""

    valid: bool
    row_count: int
    usable_rows: int
    schema_type: str | None = None
    enable_thinking: bool | None = None
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]
    preview: list[PreviewRow]


class DatasetUploaded(DatasetReport):
    """The response to a successful upload: the report plus its identity."""

    id: str
    filename: str


class DatasetRecord(BaseModel):
    """A stored dataset with its report attached -- including a dataset that
    failed validation, whose report is the reason it was kept."""

    id: str
    filename: str
    created_at: float
    status: str
    report: DatasetReport | None = None


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


class JobSpecPreview(BaseModel):
    """What a launch would train with, before anything is launched.

    The whole point of the launch screen is that nothing is a surprise after
    committing: this is that answer as one published shape. `hyperparameters`
    are the effective specification resolved exactly as the trainer resolves
    overrides -- showing anything else would describe a job the trainer will
    not run."""

    dataset: DatasetRecord
    hyperparameters: dict[str, Any]
    warning: FeasibilityWarning | None = None


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
    result: dict[str, Any] | None = None


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
