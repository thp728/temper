"""Turning "pour bytes into a sink" into an iterator of bounded chunks.

Spec 006's contract half: every payload crosses this process in pieces. Two
producers need the same machinery -- the tarball pushed to a machine and the
zip streamed to a browser -- and both have the shape "a callable that writes
the whole payload into a writable sink". `piped_chunks` is that shape turned
into a stream: the pour runs on a thread against one end of a pipe, the
iterator drains the other, and peak held memory is one chunk however large
the payload is.

What is asserted is external behaviour: joined chunks equal what was poured,
memory stays flat as payloads grow, a pour that dies part-way raises instead
of delivering a truncated stream looking whole, and abandoning the iterator
never strands the writer.
"""

import hashlib
import threading
import tracemalloc

import pytest

from temper_control_plane.chunks import PIPED_CHUNK_BYTES, piped_chunks


def test_joined_chunks_are_exactly_what_was_poured():
    def pour(sink):
        for n in range(7):
            sink.write(bytes([n]) * 1000)

    want = b"".join(bytes([n]) * 1000 for n in range(7))
    assert b"".join(piped_chunks(pour)) == want


def test_a_large_payload_arrives_as_many_bounded_chunks():
    """One blob would mean no caller could stay flat; many chunks, none over
    the bound, is the property every streaming caller depends on."""
    total = 3 * PIPED_CHUNK_BYTES + 17

    received = list(piped_chunks(lambda s: s.write(b"x" * total)))

    assert len(received) > 1
    assert sum(len(c) for c in received) == total
    assert max(len(c) for c in received) <= PIPED_CHUNK_BYTES


def test_peak_memory_stays_flat_while_draining_a_large_payload():
    """The writer waits for the reader, or memory scales with payload.

    Draining an 8 MiB pour under tracemalloc: if the producer ran ahead of
    the consumer, the unread bytes would have to live somewhere and the peak
    would show them.
    """
    tracemalloc.start()
    try:
        seen = 0

        def pour(sink):
            # In bounded pieces, as a real producer does: one giant bytes
            # object would put the payload in the measurement itself.
            piece = b"y" * (64 << 10)
            for _ in range(8 << 20 >> 16):
                sink.write(piece)

        # Counted, never retained -- retaining would be exactly the
        # accumulating-consumer defect the clause forbids.
        for chunk in piped_chunks(pour):
            seen += len(chunk)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert seen == 8 << 20
    assert peak < (4 << 20), f"producer ran ahead of the reader: {peak}"


def test_a_failing_pour_raises_instead_of_looking_complete():
    """EOF from a dead producer must not read as EOF from a finished one.

    This is the truncation clause: a pour that dies mid-write leaves a short
    stream, and a consumer that saw only a clean end would deliver it
    intact-looking. The real failure surfaces here instead.
    """

    def pour(sink):
        sink.write(b"half")
        raise OSError("the dataset vanished mid-read")

    with pytest.raises(OSError, match="vanished"):
        list(piped_chunks(pour))


def test_abandoning_the_iterator_does_not_strand_the_writer():
    """A consumer that walks away -- which is what cancellation does -- must
    not leave a thread blocked forever on a pipe nobody drains."""
    started = threading.Event()

    def pour(sink):
        started.set()
        while True:
            sink.write(b"z" * PIPED_CHUNK_BYTES)

    stream = piped_chunks(pour)
    next(stream)
    assert started.wait(5)
    stream.close()  # returning at all is the assertion


def test_chunks_cross_a_hash_unchanged():
    """The drained stream's digest equals the poured bytes' digest -- the
    same identity the transport tier asserts about its own transfers."""

    block = bytes(range(256)) * 1024

    def pour(sink):
        for _ in range(12):  # 3 MiB
            sink.write(block)

    h = hashlib.sha256()
    for chunk in piped_chunks(pour):
        h.update(chunk)

    want = hashlib.sha256(block * 12).hexdigest()
    assert h.hexdigest() == want
