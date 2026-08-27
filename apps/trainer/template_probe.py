"""The export-time template probe (Spec 009 / issue #59).

A fixed probe conversation is tokenised through the template used in training
and through the template serialised into the artifact, and the resulting token
ids must be identical. The probe runs on **every** export -- whether or not
anything was overridden -- because a wrong thinking-mode detection produces a
wrong template with no override involved, and once the template controls are
exposed (issue #80) it is the only thing standing between an advanced user and
a model that trains cleanly and answers wrongly.

Two templates are compared:

* **the training template** -- what Axolotl applied when it formatted the
  training rows. Today that is the model's own chat template
  (`tokenizer_default`) with `enable_thinking` detected from the dataset; an
  explicit template (issue #80's override surface) is used as given.
* **the serialised template** -- the concrete template the artifact records,
  so a downloader can reproduce the exact formatting training used. The
  concrete Jinja string is recorded, never the `tokenizer_default` directive,
  because a directive is for the trainer to interpret and a string is what a
  server can apply.

The probe compares the ids. A mismatch fails the export with the stable code
``template_probe_mismatch`` and a message that names what differs -- the id
counts, the first differing id, and the rendered text around the first
difference -- rather than only that something did. If the probe cannot run at
all, the export fails closed with ``template_probe_unavailable``: a guard that
silently disappears when it cannot run is no guard.

The tokenizer is the only external dependency and it is passed in, not
imported: the image provides transformers (the import happens in the
entrypoint), and the host test suite injects a double.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# A fixed conversation, identical on every run. It carries a
# system/user/assistant shape so the rendered form is representative of real
# training rows without depending on which model is being trained.
PROBE_CONVERSATION: list[dict[str, str]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]

# The stable codes a failing probe writes into the result document. A reader
# branches on the code; the message carries the specifics.
PROBE_MISMATCH_CODE = "template_probe_mismatch"
PROBE_UNAVAILABLE_CODE = "template_probe_unavailable"


class Tokenizer(Protocol):
    """The slice of a Hugging Face tokenizer the probe needs.

    The entrypoint constructs a real ``AutoTokenizer``; tests provide a fake.
    ``apply_chat_template`` returns the rendered text when ``tokenize=False``
    and a list of ids when ``tokenize=True``, exactly as transformers does.
    """

    chat_template: str

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        chat_template: str | None = ...,
        chat_template_kwargs: dict[str, Any] | None = ...,
    ) -> str | list[int]: ...


@dataclass(frozen=True)
class Tokenisation:
    """One side of the comparison: the ids and the rendered text."""

    ids: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class ProbeOutcome:
    """The probe's verdict, shaped for the artifact record.

    ``ok`` is the only field a caller may branch on for a pass. The rest name
    what differs so a failure is fixable from the record alone: id counts on
    both sides, the first differing id, and the serialised template the
    artifact records.
    """

    ok: bool
    error_code: str | None = None
    message: str | None = None
    training_id_count: int | None = None
    serialised_id_count: int | None = None
    first_id_index: int | None = None
    training_id: int | None = None
    serialised_id: int | None = None
    serialised_template: str | None = None
    serialised_kwargs: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """The record that lands in result.json beside the artifact."""
        out: dict[str, Any] = {"ok": self.ok}
        if self.error_code is not None:
            out["error_code"] = self.error_code
        if self.message is not None:
            out["message"] = self.message
        out["training_id_count"] = self.training_id_count
        out["serialised_id_count"] = self.serialised_id_count
        if self.first_id_index is not None:
            out["first_difference"] = {
                "id_index": self.first_id_index,
                "training_id": self.training_id,
                "serialised_id": self.serialised_id,
            }
        out["serialised_template"] = self.serialised_template
        out["serialised_kwargs"] = self.serialised_kwargs
        return out


class TemplateProbeFailure(Exception):
    """The export must not succeed: the probe failed or could not run.

    Carries the full outcome so the entrypoint records the specifics, and the
    stable ``error_code`` for the result document.
    """

    def __init__(self, outcome: ProbeOutcome) -> None:
        self.outcome = outcome
        super().__init__(
            outcome.message or outcome.error_code or "template probe failed"
        )

    @property
    def error_code(self) -> str | None:
        """The stable code the result document records."""
        return self.outcome.error_code


def resolve_template(
    tokenizer: Tokenizer,
    *,
    chat_template: str | None,
    kwargs: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """The concrete template a config directive resolves to.

    ``tokenizer_default`` is Axolotl's directive for "the model's own
    template"; the artifact records that concrete Jinja string, so a
    downloader never has to interpret the directive. An explicit template
    (issue #80's override surface) is passed through as given.
    """
    if chat_template == "tokenizer_default":
        template = tokenizer.chat_template
    else:
        template = chat_template or tokenizer.chat_template
    return template, dict(kwargs or {})


def _tokenise(
    tokenizer: Tokenizer,
    conversation: list[dict[str, str]],
    chat_template: str | None,
    kwargs: dict[str, Any] | None,
) -> Tokenisation:
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        chat_template=chat_template,
        chat_template_kwargs=kwargs,
    )
    ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        chat_template=chat_template,
        chat_template_kwargs=kwargs,
    )
    return Tokenisation(ids=tuple(ids), text=str(text))


def _first_id_difference(
    ids_a: tuple[int, ...], ids_b: tuple[int, ...]
) -> tuple[int | None, int | None, int | None]:
    """(index, a, b) of the first position where the id sequences disagree."""
    for i, (a, b) in enumerate(zip(ids_a, ids_b, strict=False)):
        if a != b:
            return i, a, b
    return None, None, None


def _first_text_difference(
    text_a: str, text_b: str
) -> tuple[int | None, str, str]:
    """The first character where the rendered texts differ, with context.

    The window is wide enough that a template name or a thinking-mode marker
    at the start of the rendered form is visible whole: a 12-char window
    truncated the very difference the message exists to show.
    """
    for i, (ca, cb) in enumerate(zip(text_a, text_b, strict=False)):
        if ca != cb:
            lo, hi = max(0, i - 24), i + 24
            return i, text_a[lo:hi], text_b[lo:hi]
    return None, "", ""


def _mismatch_message(
    training: Tokenisation,
    serialised: Tokenisation,
    *,
    training_template: str,
    serialised_template: str,
) -> str:
    """The one wording for a failed probe, naming what differs.

    The message is built to be actionable: it names both templates when they
    are different strings, the id counts on both sides, the first differing
    id, and the rendered text around the first difference. "The templates
    differ" tells no one what to fix.
    """
    lines = [
        "the artifact's serialised template does not tokenise the probe "
        "conversation identically to the training template:"
    ]
    if training_template != serialised_template:
        lines.append(
            f"the training template {training_template!r} and the "
            f"serialised template {serialised_template!r} are not the same"
        )
    lines.append(
        f"training rendered {len(training.ids)} ids, the serialised template "
        f"rendered {len(serialised.ids)}"
    )
    index, id_a, id_b = _first_id_difference(training.ids, serialised.ids)
    if index is not None:
        lines.append(
            f"the first differing id is at index {index} "
            f"(training {id_a}, serialised {id_b})"
        )
    char, ctx_a, ctx_b = _first_text_difference(training.text, serialised.text)
    if char is not None:
        lines.append(
            f"the rendered text first differs at character {char}: "
            f"training {ctx_a!r} vs serialised {ctx_b!r}"
        )
    return " ".join(lines)


def compare(
    training: Tokenisation,
    serialised: Tokenisation,
    *,
    training_template: str,
    serialised_template: str,
    serialised_kwargs: dict[str, Any],
) -> ProbeOutcome:
    """Compare the two sides and produce the probe's verdict."""
    if training.ids == serialised.ids:
        return ProbeOutcome(
            ok=True,
            training_id_count=len(training.ids),
            serialised_id_count=len(serialised.ids),
            serialised_template=serialised_template,
            serialised_kwargs=serialised_kwargs,
        )
    index, id_a, id_b = _first_id_difference(training.ids, serialised.ids)
    return ProbeOutcome(
        ok=False,
        error_code=PROBE_MISMATCH_CODE,
        message=_mismatch_message(
            training,
            serialised,
            training_template=training_template,
            serialised_template=serialised_template,
        ),
        training_id_count=len(training.ids),
        serialised_id_count=len(serialised.ids),
        first_id_index=index,
        training_id=id_a,
        serialised_id=id_b,
        serialised_template=serialised_template,
        serialised_kwargs=serialised_kwargs,
    )


def run_probe(
    tokenizer: Tokenizer,
    *,
    training_template: str,
    training_kwargs: dict[str, Any],
    serialised_template: str,
    serialised_kwargs: dict[str, Any],
    conversation: list[dict[str, str]] = PROBE_CONVERSATION,
) -> ProbeOutcome:
    """Tokenise the fixed conversation through both templates and compare.

    Both sides are the concrete template strings: the training side is what
    Axolotl applied (resolved from the config directive), the serialised side
    is what the artifact records. Identical ids are required.
    """
    training = _tokenise(
        tokenizer, conversation, training_template, training_kwargs
    )
    serialised = _tokenise(
        tokenizer, conversation, serialised_template, serialised_kwargs
    )
    return compare(
        training,
        serialised,
        training_template=training_template,
        serialised_template=serialised_template,
        serialised_kwargs=serialised_kwargs,
    )
