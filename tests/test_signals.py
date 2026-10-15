"""`qwi generate`'s signal handling, in a real subprocess (§5.11 "CLI signals")."""

import signal
import subprocess
import sys

import pytest

# Stands in for a run: waits for the cancel flag like a step hook would, then takes `rest`
# seconds to reach "the next step" and finish cleanly.
CHILD = """
import sys, threading, time
from qwi.cli import cancel_on_signals

cancel = threading.Event()
with cancel_on_signals(cancel):
    print("ready", flush=True)
    deadline = time.monotonic() + 10
    while not cancel.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    print("cancel set" if cancel.is_set() else "timeout", flush=True)
    time.sleep(float(sys.argv[1]))
print("finished cleanly", flush=True)
"""


def start(rest: float) -> subprocess.Popen[str]:
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(rest)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline() == "ready\n"
    return child


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_first_signal_cancels_cleanly(sig: signal.Signals) -> None:
    child = start(rest=0.2)
    child.send_signal(sig)
    out, err = child.communicate(timeout=10)
    assert child.returncode == 0
    assert out == "cancel set\nfinished cleanly\n"
    assert "cancelling at the next step" in err


def test_second_ctrl_c_exits_right_away() -> None:
    child = start(rest=30)  # would take 30 s to reach the next step
    child.send_signal(signal.SIGINT)
    assert child.stdout is not None
    assert child.stdout.readline() == "cancel set\n"
    child.send_signal(signal.SIGINT)
    out, err = child.communicate(timeout=5)
    assert child.returncode == 130
    assert "finished cleanly" not in out
    assert "stopping now" in err


def test_handlers_are_restored_after_the_run() -> None:
    import threading

    from qwi.cli import cancel_on_signals

    before = signal.getsignal(signal.SIGINT)
    with cancel_on_signals(threading.Event()):
        assert signal.getsignal(signal.SIGINT) is not before
    assert signal.getsignal(signal.SIGINT) is before
