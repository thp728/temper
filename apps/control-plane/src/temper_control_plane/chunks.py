"""One shape shared by every big payload this process produces: chunks.

Spec 006's rule is that no code path holds a full payload in memory. Two
producers on that path -- the tarball pushed to a machine and the zip served
to a browser -- are written naturally as "pour the whole thing into a
writable sink": `tarfile` and `zipfile` both work that way. `piped_chunks`
turns exactly that shape into a stream, so a producer keeps its natural code
and its caller still receives bounded pieces.

The mechanism is a pipe between threads. The pour runs on one thread against
the pipe's write end; the iterator drains the read end in fixed-size chunks.
Each side blocks when the other falls behind, which is the property that
makes memory flat: an unread byte lives in the pipe buffer, never in a
growing Python list. It is the same bounded-chunk shape the provider seam's
`fetch_stream` yields, so callers downstream of either see one grammar.

Ownership is stated once because a double close or a missed close is exactly
the kind of bug that only shows up under load: the pour writes and never
closes; the producing thread closes the write end when the pour finishes,
*which is what signals EOF*; the consuming generator closes the read end --
first thing in its cleanup, so an abandoned stream unblocks the writer
instead of joining against a blocked one.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from typing import BinaryIO

# How much one drained step hands back. Bounded, not tuned: the tests assert
# memory stays flat as payloads grow, not that a particular size was used.
# Matches the provider's own fetch chunk for the same reason.
PIPED_CHUNK_BYTES = 256 * 1024

# Ceiling on waiting for the writer after the reader stops. With the read end
# closed first, a blocked write fails immediately, so this normally returns
# in microseconds; it exists so a mistake fails rather than hangs the
# process on exit.
ABANDONED_JOIN_TIMEOUT_S = 30.0


class ChunkReader:
    """A ``read(amt)`` view over an iterator of byte chunks.

    Archive writers -- ``tarfile.addfile`` above all -- want a file object,
    and the storage seam hands out chunk iterators. This is the bridge, and
    it holds no more than one chunk plus whatever the last ``read`` asked
    for: the archive layer pulls bounded blocks through it, so an object can
    be written into a tar without ever being held whole.
    """

    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks = iter(chunks)
        self._buffer = bytearray()

    def read(self, amt: int = -1) -> bytes:
        # amt < 0 means "everything left", per file semantics -- which for
        # an unbounded object means holding it whole. The archive callers
        # here always pass a positive block size; that is the contract this
        # class is built for.
        if amt == 0:
            return b""
        while len(self._buffer) < amt or amt < 0:
            chunk = next(self._chunks, None)
            if chunk is None:
                break
            self._buffer.extend(chunk)
        if amt < 0:
            data = bytes(self._buffer)
            self._buffer.clear()
            return data
        data = bytes(self._buffer[:amt])
        del self._buffer[:amt]
        return data


def piped_chunks(pour: Callable[[BinaryIO], None]) -> Iterator[bytes]:
    """Run `pour(sink)` on a thread and yield what it wrote as chunks.

    `pour` writes the whole payload to the binary sink it is handed and
    returns; it must not close the sink -- closing once is this function's
    job. If `pour` raises, the exception is re-raised at the iterator after
    whatever crossed the pipe has been drained: EOF from a dead producer
    must not read as EOF from a finished one, or a truncated payload would
    be delivered looking intact.
    """
    read_fd, write_fd = os.pipe()
    sink = open(write_fd, "wb", closefd=True)
    failure: list[BaseException] = []

    def produce() -> None:
        try:
            pour(sink)
        except BaseException as e:
            # Recorded, not swallowed: raised below if the consumer was still
            # reading; dropped only when the consumer had already walked
            # away. A broken pipe from a departed reader lands here too.
            failure.append(e)
        finally:
            # Closing the write end is what tells the reader the stream is
            # over. Errors are suppressed rather than raised into a thread
            # nobody watches: on a pipe they mean only that the reader left,
            # which is the consumer's business, not the pour's.
            with suppress(OSError):
                sink.close()

    writer = threading.Thread(target=produce, daemon=True, name="chunk-pourer")
    writer.start()
    source = open(read_fd, "rb", closefd=True)
    try:
        while chunk := source.read(PIPED_CHUNK_BYTES):
            yield chunk
        # Clean EOF: the pour finished and shut its end, so joining is
        # prompt and the verdict -- including a pour that died part-way --
        # is known before the last chunk is released as delivered.
        writer.join()
        if failure:
            raise failure[0]
    finally:
        # Also reached on abandonment (GeneratorExit at the yield). The read
        # end closes FIRST, so a writer blocked on a full pipe fails instead
        # of being waited on; the timeout only covers a writer that has not
        # reached a write yet.
        source.close()
        writer.join(ABANDONED_JOIN_TIMEOUT_S)
