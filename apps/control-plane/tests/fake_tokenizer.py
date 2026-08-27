"""The test tokenizer seam: deterministic, offline, cheap.

`temper_control_plane.tokenize.TOKENIZER` is the seam a real tokenizer sits
behind. Tests replace it with this fake, which reports one token per character
of content -- so every count in a test is exact arithmetic the test can assert
on, no network, no 11 MB download, exactly as the model-facts seam is replaced
with `fake_models`.
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
