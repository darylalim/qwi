import subprocess
import sys

import pytest

from qwi import lock, paths
from qwi.errors import PipelineBusy

HOLD = """
import sys, time
from qwi.lock import RunLock
with RunLock():
    print("held", flush=True)
    sys.stdin.readline()
"""


def test_second_lock_in_the_same_process_is_busy() -> None:
    with lock.RunLock():
        assert lock.is_held()
        with pytest.raises(PipelineBusy, match="started 20"), lock.RunLock():
            pass
    assert not lock.is_held()


def test_no_lock_file_means_not_held() -> None:
    assert not paths.lock_path().exists()
    assert not lock.is_held()


def test_another_process_holding_the_lock_makes_this_one_busy() -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline() == "held\n"
        assert lock.is_held()
        with pytest.raises(PipelineBusy), lock.RunLock():
            pass
    finally:
        holder.communicate("\n", timeout=10)
    assert not lock.is_held()


def test_the_os_drops_the_lock_when_its_process_dies() -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    assert holder.stdout is not None
    assert holder.stdout.readline() == "held\n"
    holder.kill()
    holder.wait(timeout=10)
    with lock.RunLock():
        pass


def test_unreadable_holder_info_still_reports_busy() -> None:
    paths.lock_path().parent.mkdir(parents=True)
    with lock.RunLock():
        paths.lock_path().write_text("{half")
        with pytest.raises(PipelineBusy) as excinfo, lock.RunLock():
            pass
    assert excinfo.value.started_at is None
