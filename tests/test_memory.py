import pytest

from qwi import memory
from qwi.errors import DeviceOutOfMemory

# Shape copied from /usr/bin/vm_stat on the M2 Max (macOS 26), trimmed.
VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  1483722.
Pages active:                                 213119.
Swapins:                                      619608.
Swapouts:                                    1885391.
"""


def test_parse_swapouts_uses_the_header_page_size() -> None:
    assert memory.parse_swapouts(VM_STAT) == 1885391 * 16384
    other_page = VM_STAT.replace("16384", "4096")
    assert memory.parse_swapouts(other_page) == 1885391 * 4096


@pytest.mark.parametrize("drop", ["page size", "Swapouts"])
def test_parse_swapouts_rejects_unexpected_output(drop: str) -> None:
    text = "\n".join(line for line in VM_STAT.splitlines() if drop not in line)
    with pytest.raises(ValueError, match="vm_stat"):
        memory.parse_swapouts(text)


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


def test_swapouts_reads_this_machine() -> None:
    assert memory.swapouts() >= 0


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
