import pytest

from qwi import memory
from qwi.errors import DeviceOutOfMemory

# Copied from /usr/bin/vm_stat on the M2 Max (macOS 26), trimmed; note the quoted key.
VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   893661.
Pages active:                                 267761.
Pages inactive:                               749056.
Pages purgeable:                                8068.
"Translation faults":                       43722768.
Swapins:                                      625530.
Swapouts:                                    1885391.
"""


def test_vm_stat_uses_the_header_page_size() -> None:
    assert memory.parse_vm_stat(VM_STAT).swapout_bytes == 1885391 * 16384
    other_page = VM_STAT.replace("16384", "4096")
    assert memory.parse_vm_stat(other_page).swapout_bytes == 1885391 * 4096


def test_vm_stat_available_is_free_plus_inactive_plus_purgeable() -> None:
    stat = memory.parse_vm_stat(VM_STAT)
    assert stat.available_bytes == (893661 + 749056 + 8068) * 16384
    assert stat.pages["Translation faults"] == 43722768


def test_vm_stat_rejects_output_without_a_page_size() -> None:
    with pytest.raises(ValueError, match="page size"):
        memory.parse_vm_stat(VM_STAT.splitlines(keepends=True)[1])


def test_vm_stat_names_a_missing_counter() -> None:
    text = VM_STAT.replace("Swapouts", "Swapped")
    with pytest.raises(ValueError, match="no Swapouts line"):
        _ = memory.parse_vm_stat(text).swapout_bytes


def test_vm_stat_reads_this_machine() -> None:
    stat = memory.vm_stat()
    assert stat.swapout_bytes >= 0
    assert stat.available_bytes > 0


class Counter:
    def __init__(self, value: int) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


def test_watchdog_allows_swap_up_to_the_limit() -> None:
    counter = Counter(10_000)
    watchdog = memory.SwapWatchdog("generate", limit_bytes=500, read=counter)
    watchdog.start()
    counter.value += 500
    watchdog.call_in_loop(t=0, seed=0, prompt="p", latents=None, config=None, time_steps=None)
    assert watchdog.delta_bytes == 500


def test_watchdog_raises_device_out_of_memory_past_the_limit() -> None:
    counter = Counter(10_000)
    watchdog = memory.SwapWatchdog("generate", limit_bytes=500, read=counter)
    watchdog.start()
    counter.value += 501
    with pytest.raises(DeviceOutOfMemory, match="during generate") as excinfo:
        watchdog.call_in_loop(t=3, seed=0, prompt="p", latents=None, config=None, time_steps=None)
    assert excinfo.value.stage == "generate"


def test_watchdog_needs_a_baseline() -> None:
    with pytest.raises(RuntimeError, match="start"):
        memory.SwapWatchdog("generate", read=Counter(0)).check()


def test_track_peak_resets_then_reads_mlx_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlx.core as mx

    calls: list[str] = []
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: calls.append("reset"))
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 13_820_000_000)
    with memory.track_peak() as peak:
        assert calls == ["reset"]
    assert peak.gb == pytest.approx(13.82)


def test_configure_and_evict_call_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlx.core as mx

    calls: list[object] = []
    monkeypatch.setattr(mx, "set_cache_limit", calls.append)
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append("clear"))
    memory.configure()
    memory.evict()
    assert calls == [2 * 2**30, "clear"]
