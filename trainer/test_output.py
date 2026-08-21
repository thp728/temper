"""The innermost layer of the output channel.

The framework's output has to leave the container line by line, and progress
bars are the reason that is not simply `for line in pipe`: tqdm redraws with a
carriage return and no newline, so a reader that waits for `\n` sees nothing
for the entire length of a bar. On a job whose whole training phase is a bar,
that is indistinguishable from the silence this ticket exists to remove.
"""

import io

from entrypoint import iter_output_lines


def lines_of(payload: bytes, **kw) -> list[str]:
    return list(iter_output_lines(io.BytesIO(payload), **kw))


def test_newline_separated_output_arrives_as_lines():
    assert lines_of(b"first\nsecond\nthird\n") == ["first", "second", "third"]


def test_a_carriage_return_ends_a_line_too():
    """Each redraw of a progress bar is output; none of them ends in a newline."""
    assert lines_of(b"10%|##\r20%|####\r30%|######\r") == [
        "10%|##", "20%|####", "30%|######"]


def test_crlf_does_not_produce_an_empty_line_between_lines():
    assert lines_of(b"first\r\nsecond\r\n") == ["first", "second"]


def test_a_trailing_partial_line_is_yielded_at_end_of_stream():
    """A prompt or a final bar with no terminator is still output."""
    assert lines_of(b"done\nno terminator here") == ["done", "no terminator here"]


def test_a_line_split_across_reads_is_reassembled():
    assert lines_of(b"abcdefgh\nij\n", chunk_size=3) == ["abcdefgh", "ij"]


def test_blank_lines_are_dropped_rather_than_flooding_the_event_log():
    assert lines_of(b"a\n\n\n  \nb\n") == ["a", "b"]


def test_undecodable_bytes_do_not_end_the_stream():
    """A byte the framework never meant as text must not lose what follows."""
    assert lines_of(b"ok\n\xff\xfe\nafter\n")[0] == "ok"
    assert lines_of(b"ok\n\xff\xfe\nafter\n")[-1] == "after"
