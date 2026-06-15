from __future__ import annotations

from loguru import logger
from rich.console import Console


LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} - {message}"
_SHARED_CONSOLE: Console | None = None


def get_shared_console() -> Console:
    """Return the Rich console used by both logs and progress bars."""
    global _SHARED_CONSOLE
    if _SHARED_CONSOLE is None:
        _SHARED_CONSOLE = Console()
    return _SHARED_CONSOLE


def setup_cli_logging(*, level: str = "INFO") -> None:
    """Configure Loguru to print through the shared Rich console."""
    logger.remove()

    def sink(message: str) -> None:
        get_shared_console().print(message, end="")

    logger.add(
        sink,
        level=level,
        format=LOG_FORMAT,
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )


def setup_queue_logging(progress_queue: object, *, level: str = "INFO") -> None:
    """Configure worker Loguru output to be emitted by the parent process."""
    logger.remove()

    def sink(message: str) -> None:
        progress_queue.put(("log", str(message)))

    logger.add(
        sink,
        level=level,
        format=LOG_FORMAT,
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )
