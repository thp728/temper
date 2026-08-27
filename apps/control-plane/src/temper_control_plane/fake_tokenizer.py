"""The test and journey tokenizer: deterministic, offline, cheap.

`temper_control_plane.tokenize.TOKENIZER` is the seam a real tokenizer sits
behind. Unit tests replace it with this fake, and the tokenizer seam honours
`TEMPER_FAKE_PROVIDER` by defaulting to it -- the same deal `fake_models.py`
makes for the model facts, so the browser journeys never reach Hugging Face.
One token per character means every count in a test or journey is exact
arithmetic the assertion can predict.
"""

from __future__ import annotations


class _Encoded:
    """The shape of `tokenizers`' encode result that counting reads: `.ids`."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    """One token per character, so token counts equal character counts."""

    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoded:
        return _Encoded(list(range(len(text))))


def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()
