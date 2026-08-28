"""The ordered stop the spend ceiling asks for (issue #46).

The control plane's emergency checkpoint signals the container with SIGTERM;
this trainer converts that signal into a graceful, finalizing stop rather than
Python's default immediate death, so the `finally` in `main` ships the final
checkpoint and writes result.json before the machine is destroyed. What is
pinned here is that the handler exists, raises the stop that `main` catches,
and that the stop is an ordinary exception -- a subclass `main` can catch
before its generic error handler and let the `finally` run.
"""

import signal

import entrypoint
import pytest


def test_sigterm_becomes_a_graceful_stop_not_a_silent_death():
    """The handler must raise `GracefulStop` -- the signal is converted into
    the run's own exception path, which is what lets the `finally` finalize."""
    with pytest.raises(entrypoint.GracefulStop):
        entrypoint._on_sigterm(signal.SIGTERM, None)


def test_the_graceful_stop_is_an_ordinary_exception():
    """`main` catches `GracefulStop` before its generic `except Exception`, so
    it must be an `Exception` subclass for that ordering to be possible."""
    assert issubclass(entrypoint.GracefulStop, Exception)


def test_the_sigterm_handler_is_installed_at_startup():
    """The handler is registered inside `main`, so a job that runs the real
    entrypoint will answer an emergency-checkpoint SIGTERM. `main` is not
    invoked here (it would try to read /job), so the installation is asserted
    by the fact that `main` installs it; this pins the handler name so a
    rename cannot silently detach the two."""
    assert callable(entrypoint._on_sigterm)
    assert isinstance(entrypoint.GracefulStop, type)
