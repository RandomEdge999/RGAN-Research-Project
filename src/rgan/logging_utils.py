"""Plain-text logging utilities for RGAN training.

No Rich dependency. Every log line is timestamped and includes the caller location
so that cloud logs (CloudWatch, SageMaker) show exactly where and when things happen.

All output goes to both stdout AND a log file (when configured) so that failed runs
always leave a complete trace on disk.
"""
import contextlib
import inspect
import os
import sys
import time
import traceback
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Log file — set once via setup_log_file(), then every log line is tee'd there
# ---------------------------------------------------------------------------
_LOG_FILE = None


def setup_log_file(results_dir: str) -> Path:
    """Create a run.log in the given results directory and tee all future log
    lines there.  Returns the path to the log file."""
    global _LOG_FILE
    p = Path(results_dir) / "run.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(p, "a", buffering=1)  # line-buffered
    _write_to_log(f"--- Log file opened: {p.resolve()} ---")
    _write_to_log(f"--- Python {sys.version} | {platform.platform()} ---")
    _write_to_log(f"--- PID {os.getpid()} | CWD {os.getcwd()} ---")
    return p


def close_log_file() -> None:
    """Flush and close the log file."""
    global _LOG_FILE
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.flush()
            _LOG_FILE.close()
        except Exception:
            pass
        _LOG_FILE = None


def _write_to_log(line: str) -> None:
    """Write a line to the log file if one is open."""
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.write(line + "\n")
            _LOG_FILE.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    """UTC timestamp for log lines."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _caller_location(depth: int = 2) -> str:
    """Return 'filename:lineno:funcname' of the caller (skipping wrapper frames)."""
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            if frame is not None:
                frame = frame.f_back
        if frame is not None:
            fname = os.path.basename(frame.f_code.co_filename)
            func = frame.f_code.co_name
            return f"{fname}:{frame.f_lineno}:{func}"
    finally:
        del frame
    return "?:?:?"


def log(msg: str, level: str = "INFO") -> None:
    """Print a timestamped, location-tagged log line to stdout and log file."""
    loc = _caller_location(depth=2)
    line = f"[{_timestamp()}] [{level}] [{loc}] {msg}"
    print(line, flush=True)
    _write_to_log(line)


def log_info(msg: str) -> None:
    loc = _caller_location(depth=2)
    line = f"[{_timestamp()}] [INFO] [{loc}] {msg}"
    print(line, flush=True)
    _write_to_log(line)


def log_warn(msg: str) -> None:
    loc = _caller_location(depth=2)
    line = f"[{_timestamp()}] [WARN] [{loc}] {msg}"
    print(line, flush=True)
    _write_to_log(line)


def log_error(msg: str) -> None:
    loc = _caller_location(depth=2)
    line = f"[{_timestamp()}] [ERROR] [{loc}] {msg}"
    print(line, flush=True)
    _write_to_log(line)


def log_debug(msg: str) -> None:
    loc = _caller_location(depth=2)
    line = f"[{_timestamp()}] [DEBUG] [{loc}] {msg}"
    print(line, flush=True)
    _write_to_log(line)


def log_exception(msg: str = "Exception caught") -> None:
    """Log an error message AND the full traceback of the current exception."""
    loc = _caller_location(depth=2)
    tb = traceback.format_exc()
    header = f"[{_timestamp()}] [ERROR] [{loc}] {msg}"
    print(header, flush=True)
    print(tb, flush=True)
    _write_to_log(header)
    _write_to_log(tb)


def log_traceback() -> None:
    """Log the full traceback of the current exception (no extra message)."""
    tb = traceback.format_exc()
    print(tb, flush=True)
    _write_to_log(tb)


def log_step(msg: str) -> None:
    """Log a granular step inside a larger operation.  Uses STEP level for easy
    grep filtering when debugging failed runs."""
    loc = _caller_location(depth=2)
    line = f"[{_timestamp()}] [STEP] [{loc}] {msg}"
    print(line, flush=True)
    _write_to_log(line)


def log_var(name: str, value: Any) -> None:
    """Log the name and value of a variable for debugging.  Truncates large repr."""
    loc = _caller_location(depth=2)
    val_str = repr(value)
    if len(val_str) > 500:
        val_str = val_str[:500] + "... (truncated)"
    line = f"[{_timestamp()}] [VAR] [{loc}] {name} = {val_str}"
    print(line, flush=True)
    _write_to_log(line)


def log_shape(name: str, arr: Any) -> None:
    """Log the shape and dtype of a numpy/torch array."""
    loc = _caller_location(depth=2)
    shape = getattr(arr, "shape", "N/A")
    dtype = getattr(arr, "dtype", type(arr).__name__)
    line = f"[{_timestamp()}] [SHAPE] [{loc}] {name}: shape={shape} dtype={dtype}"
    print(line, flush=True)
    _write_to_log(line)


# ---------------------------------------------------------------------------
# Console object — simple wrapper so existing code keeps calling console.print / console.log
# ---------------------------------------------------------------------------

class PlainConsole:
    """Minimal console that prints plain text with timestamps."""

    def rule(self, text: str = "") -> None:
        line = "=" * 80
        out = f"\n{line}"
        if text:
            out += f"\n{text}"
        out += f"\n{line}"
        print(out, flush=True)
        _write_to_log(out)

    def log(self, *args, **kwargs) -> None:
        msg = " ".join(str(a) for a in args)
        loc = _caller_location(depth=2)
        line = f"[{_timestamp()}] [LOG] [{loc}] {msg}"
        print(line, flush=True)
        _write_to_log(line)

    def print(self, *args, **kwargs) -> None:
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        _write_to_log(msg)


_CONSOLE: Optional[PlainConsole] = None


def get_console() -> PlainConsole:
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = PlainConsole()
    return _CONSOLE


def has_rich() -> bool:
    """Always False — Rich UI is disabled for robust cloud logging."""
    return False


# ---------------------------------------------------------------------------
# Epoch progress — plain print, no progress bars
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def epoch_progress(total: int, description: str = "Training"):
    """Yield (None, None); epoch logging goes through update_epoch()."""
    log_info(f"Starting {description} ({total} epochs)")
    t0 = time.perf_counter()
    yield None, None
    elapsed = time.perf_counter() - t0
    log_info(f"Finished {description} — total wall time {elapsed:.1f}s")


def update_epoch(progress: Any, task_id: Any, epoch: int, total: int, metrics: Dict[str, float]) -> None:
    """Log one epoch's metrics as a plain-text line."""
    parts = [f"{k}={v:.6f}" for k, v in metrics.items()]
    msg = " | ".join(parts)
    # Include system stats if possible
    sys_stats = ""
    try:
        import psutil
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        sys_stats = f" | CPU={cpu:.1f}% RAM={ram:.1f}%"
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**3)
            gpu_max = torch.cuda.max_memory_allocated() / (1024**3)
            sys_stats += f" GPU_MEM={gpu_mem:.2f}GB GPU_PEAK={gpu_max:.2f}GB"
    except Exception:
        pass
    line = f"Epoch {epoch:03d}/{total:03d} | {msg}{sys_stats}"
    print(line, flush=True)
    _write_to_log(line)


# ---------------------------------------------------------------------------
# Banner & key-value table
# ---------------------------------------------------------------------------

def print_banner(console: Any, title: str, subtitle: str = "") -> None:
    console.rule(title.upper())
    if subtitle:
        console.print(subtitle)
    sys_info = f"Python {sys.version.split()[0]} | {platform.system()} | PID {os.getpid()} | Plain-text logging"
    console.print(sys_info)


def print_kv_table(console: Any, title: str, rows: Dict[str, Any]) -> None:
    console.rule(title)
    for k, v in rows.items():
        console.print(f"  {k}: {v}")
