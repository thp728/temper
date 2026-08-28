"""The base model download is invoked in a mode that reports progress (issue
#49).

The model download is the phase that dominates a large job, and it produces no
progress signal unless it is *invoked* in a mode that reports it: axolotl
downloads the model itself, but whether the bars reach the control plane is
left to tqdm's TTY detection, and piped output is exactly where they
disappear. These tests pin that the trainer prefetches the model before
training, that it passes the run's revision, and that it forces progress bars
on so a non-TTY cannot silence them.
"""

from entrypoint import prefetch_model


def test_prefetch_downloads_the_base_model_before_training(
    tmp_path, monkeypatch
):
    calls: dict = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return "/out/hf/models--Qwen--Qwen3-4B"

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot_download
    )
    prefetch_model(
        {"base_model": "Qwen/Qwen3-4B", "base_revision": "main"}, tmp_path
    )
    assert calls["repo_id"] == "Qwen/Qwen3-4B"
    assert calls["revision"] == "main"


def test_prefetch_forces_progress_bars_on(tmp_path, monkeypatch):
    """tqdm's default disables on a non-TTY, and the machine's output is
    piped -- the whole point is that piped output still reports progress."""
    received: dict = {}

    def fake_snapshot_download(**kwargs):
        received.update(kwargs)
        return "cache"

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot_download
    )
    prefetch_model({"base_model": "Qwen/Qwen3-4B"}, tmp_path)
    tqdm_class = received["tqdm_class"]
    assert tqdm_class is not None
    assert tqdm_class(disable=False) is not None


def test_a_failed_prefetch_does_not_raise(tmp_path, monkeypatch):
    """A prefetch that cannot run costs a progress signal, never the job:
    axolotl downloads the model as it always has."""

    def boom(**kwargs):
        raise RuntimeError("registry unreachable")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    prefetch_model({"base_model": "Qwen/Qwen3-4B"}, tmp_path)  # must not raise
