"""The published shapes of the dataset endpoints.

These exist because the web client is generated from this API's schema. A
handler returning a dict merge publishes no shape at all: the generator emits
`unknown`, and the interface ends up hand-typing what the contract refused to
-- which is precisely the drift the generated client exists to prevent.

They describe what crosses the HTTP boundary only. The stored row keeps its
path column; `DatasetRecord` simply does not publish it, so an absolute
filesystem path reaches neither a page nor a client.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


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
