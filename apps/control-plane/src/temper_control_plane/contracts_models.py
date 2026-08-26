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

from typing import Any

from pydantic import BaseModel


class ValidationIssue(BaseModel):
    """One problem, named where it is: line number when one applies."""

    line: int | None = None
    code: str
    message: str


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
    preview: list[dict[str, Any]]


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
