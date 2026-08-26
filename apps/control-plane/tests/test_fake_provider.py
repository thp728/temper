"""The zero-cost double's own contract.

`FakeProvider` exists so every ticket touching the orchestration path tests
against the same double instead of growing its own. These tests pin the parts
of that contract later tickets depend on independently of any orchestrator
behaviour: what it records, and which stages it reports entering.
"""

from temper_control_plane.fake_provider import FakeProvider


def test_push_stream_records_the_chunks_like_a_push():
    """Same observability as the buffered push, so streamed callers' tests
    read exactly like buffered ones: one entry in `pushed`, payload whole."""
    p = FakeProvider()
    p.push_stream(None, iter([b"alpha\n", b"omega\n"]), "/tmp/temper/ds")
    assert p.pushed == [("/tmp/temper/ds", b"alpha\nomega\n")]


def test_fetch_stream_yields_the_adapter_as_chunks():
    p = FakeProvider(adapter_bytes=b"weights")
    assert list(p.fetch_stream(None, "/tmp/temper/adapter")) == [b"weights"]


def test_streaming_transfers_enter_the_buffered_stages():
    """A streamed transfer fails and pauses where the buffered one does.

    `fail_at="push"` must stop a streamed upload at the same seam, so a
    caller migrating from buffered to streaming keeps its failure-path
    tests without rewriting them.
    """
    p = FakeProvider(fail_at="push", fail_code="source_upload_failed")
    try:
        p.push_stream(None, iter([b"x"]), "/tmp/temper/ds")
        raised = False
    except Exception as e:
        raised = True
        assert getattr(e, "code", None) == "source_upload_failed"
    assert raised

    q = FakeProvider(fail_at="fetch", fail_code="artifact_fetch_failed")
    try:
        list(q.fetch_stream(None, "/tmp/temper/adapter"))
        raised = False
    except Exception as e:
        raised = True
        assert getattr(e, "code", None) == "artifact_fetch_failed"
    assert raised
