from __future__ import annotations

import contextlib
import datetime as _dt
import os
import sys
import threading
from pathlib import Path
from typing import Iterator

_LOG_PATH: Path | None = None
_LOG_DIR: Path | None = None
_DEBUG = False
_MAX_LOG_FILES = 100
_MAX_LOG_BYTES = 3 * 1024 * 1024
_CONSOLE_FILTER: _ConsoleOutputFilter | None = None
_EXTERNAL_PATTERNS = (
    "I RKNN:",
    "W RKNN:",
    "E RKNN:",
    "W rknn-toolkit-lite2",
    "E rknn-toolkit-lite2",
    "I rknn-toolkit-lite2",
    "pkg_resources is deprecated",
    "UserWarning:",
    "import pkg_resources",
    "OpenCV",
    "V4L2",
    "GStreamer",
    "MPP",
    "mpp",
)
_EXTERNAL_LINE_PREFIXES = (
    "I RKNN:",
    "W RKNN:",
    "E RKNN:",
    "W rknn-toolkit-lite2",
    "E rknn-toolkit-lite2",
    "I rknn-toolkit-lite2",
    "OpenCV",
    "V4L2",
    "GStreamer",
    "MPP",
    "mpp",
)


def _is_external_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern in stripped for pattern in _EXTERNAL_PATTERNS)


class _ConsoleOutputFilter:
    def __init__(self, log_path: Path, debug: bool = False) -> None:
        self._log_path = log_path
        self._debug = debug
        self._buffer = b""
        self._started = False
        self._shutdown = False

    def start(self) -> None:
        if self._started:
            return
        self._old_stdout = os.dup(1)
        self._old_stderr = os.dup(2)
        self._read_fd, self._write_fd = os.pipe()
        self._reader_thread = threading.Thread(
            target=self._read_loop, name="ConsoleOutputFilter", daemon=True)
        self._reader_thread.start()
        os.dup2(self._write_fd, 1)
        os.dup2(self._write_fd, 2)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._shutdown = True
        os.dup2(self._old_stdout, 1)
        os.dup2(self._old_stderr, 2)
        os.close(self._write_fd)
        self._reader_thread.join(timeout=1.0)
        os.close(self._old_stdout)
        os.close(self._old_stderr)
        self._started = False

    def _read_loop(self) -> None:
        import select
        while not self._shutdown:
            r, _, _ = select.select([self._read_fd], [], [], 0.1)
            if not r:
                continue
            chunk = os.read(self._read_fd, 4096)
            if not chunk:
                break
            self._buffer += chunk
            while b"\n" in self._buffer:
                line, self._buffer = self._buffer.split(b"\n", 1)
                self._write_line(line.decode("utf-8", errors="replace") + "\n")
            if self._buffer and not self._looks_like_external_prefix(self._buffer):
                self._mirror(self._buffer)
                self._buffer = b""
        if self._buffer:
            self._write_line(self._buffer.decode("utf-8", errors="replace"))

    def _looks_like_external_prefix(self, data: bytes) -> bool:
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return False
        if text.startswith("/"):
            return True
        return any(text.startswith(pattern) for pattern in _EXTERNAL_LINE_PREFIXES)

    def _write_line(self, line: str) -> None:
        if _is_external_line(line):
            log_message("external", line.rstrip())
            if self._debug:
                self._mirror(line.encode("utf-8", errors="replace"))
            return
        self._mirror(line.encode("utf-8", errors="replace"))

    def _mirror(self, data: bytes) -> None:
        os.write(self._old_stdout, data)


class _TaggedFdWriter:
    def __init__(self, tag: str, log_path: Path, mirror: bool = False) -> None:
        self._tag = tag
        self._log_path = log_path
        self._mirror = mirror
        self._buffer = b""

    def fileno(self) -> int:
        return self._write_fd

    def __enter__(self) -> _TaggedFdWriter:
        self._read_fd, self._write_fd = os.pipe()
        self._reader_thread = threading.Thread(
            target=self._read_loop, name=f"{self._tag}LogReader", daemon=True)
        self._reader_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        os.close(self._write_fd)
        self._reader_thread.join(timeout=2.0)

    def _read_loop(self) -> None:
        with os.fdopen(self._read_fd, "rb", closefd=True) as pipe:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                self._buffer += chunk
                while b"\n" in self._buffer:
                    line, self._buffer = self._buffer.split(b"\n", 1)
                    self._write_line(line.decode("utf-8", errors="replace"))
            if self._buffer:
                self._write_line(self._buffer.decode("utf-8", errors="replace"))

    def _write_line(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        tagged = f"[{self._tag}] {line}"
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(tagged + "\n")
        if self._mirror:
            print(tagged, file=sys.__stderr__)


def init_logging(debug: bool = False, log_dir: str | os.PathLike[str] = "logs") -> Path:
    global _LOG_PATH, _LOG_DIR, _DEBUG, _CONSOLE_FILTER
    _DEBUG = debug
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    _LOG_DIR = Path(log_dir)
    path = _LOG_DIR / f"{timestamp}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[eyeTool] external log started {timestamp}\n", encoding="utf-8")
    _LOG_PATH = path
    _CONSOLE_FILTER = _ConsoleOutputFilter(path, debug=debug)
    _CONSOLE_FILTER.start()
    print(f"External library log: {path} ({timestamp})")
    configure_opencv_logging()
    return path


def close_logging() -> None:
    global _CONSOLE_FILTER
    try:
        if _LOG_PATH is not None:
            log_message("eyeTool", "external log closed")
        _prompt_and_cleanup_logs()
    finally:
        if _CONSOLE_FILTER is not None:
            _CONSOLE_FILTER.stop()
            _CONSOLE_FILTER = None


def get_log_path() -> Path | None:
    return _LOG_PATH


def _list_log_files() -> list[Path]:
    if _LOG_DIR is None or not _LOG_DIR.exists():
        return []
    return sorted(
        [p for p in _LOG_DIR.glob("*.log") if p.is_file()],
        key=lambda p: (p.stat().st_mtime, p.name),
    )


def _log_folder_size(logs: list[Path]) -> int:
    total = 0
    for path in logs:
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def _logs_needing_cleanup(logs: list[Path]) -> tuple[list[Path], int]:
    candidates = list(logs)
    to_delete: list[Path] = []
    total_size = _log_folder_size(candidates)
    while len(candidates) > _MAX_LOG_FILES or total_size > _MAX_LOG_BYTES:
        oldest = candidates.pop(0)
        to_delete.append(oldest)
        try:
            total_size -= oldest.stat().st_size
        except OSError:
            pass
    return to_delete, total_size


def _prompt_and_cleanup_logs() -> None:
    logs = _list_log_files()
    to_delete, final_size = _logs_needing_cleanup(logs)
    if not to_delete:
        return

    current_size = _log_folder_size(logs)
    print("\nLog cleanup recommended:")
    print(f"  Current: {len(logs)} file(s), {current_size / 1024 / 1024:.2f} MB")
    print(f"  Target:  <= {_MAX_LOG_FILES} file(s), <= {_MAX_LOG_BYTES / 1024 / 1024:.2f} MB")
    print(f"  Delete:  {len(to_delete)} oldest file(s)")
    print(f"  After:   {len(logs) - len(to_delete)} file(s), {final_size / 1024 / 1024:.2f} MB")
    answer = input("Delete old log files now? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print("Log cleanup skipped.")
        return

    deleted = 0
    for path in to_delete:
        try:
            path.unlink()
            deleted += 1
        except OSError as e:
            print(f"Failed to delete log '{path}': {e}")
    print(f"Log cleanup complete: deleted {deleted} file(s).")


def log_message(tag: str, message: str) -> None:
    if _LOG_PATH is None:
        return
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        for line in str(message).splitlines() or [""]:
            if line:
                f.write(f"[{tag}] {line}\n")


def configure_opencv_logging() -> None:
    try:
        with redirect_external("opencv"):
            import cv2
            cv2.setLogLevel(0)
    except Exception:
        pass


@contextlib.contextmanager
def redirect_external(tag: str) -> Iterator[None]:
    if _LOG_PATH is None:
        yield
        return
    with _TaggedFdWriter(tag, _LOG_PATH, mirror=_DEBUG) as writer:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(writer.fileno(), 1)
            os.dup2(writer.fileno(), 2)
            with contextlib.redirect_stdout(sys.__stdout__):
                with contextlib.redirect_stderr(sys.__stderr__):
                    yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


def open_external_log_for_subprocess(tag: str):
    if _LOG_PATH is None:
        return open(os.devnull, "wb")
    return _TaggedFdWriter(tag, _LOG_PATH, mirror=_DEBUG)
