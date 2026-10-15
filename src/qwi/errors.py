"""Typed pipeline errors (PLAN.md §5.11).

M1 starts with the classes the generate stage raises; the failure-handling task adds the rest of
the §5.11 table, exit codes and `classify`.
"""


class QwiError(Exception):
    """Base class for every error the pipeline raises on purpose."""


class DeviceOutOfMemory(QwiError):
    """MLX/Metal ran out of memory, or the machine started swapping, during a stage."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"Ran out of GPU memory during {stage}: {detail}")
        self.stage = stage
        self.detail = detail
