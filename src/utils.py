"""Shared helpers: reproducibility, device selection, logging, JSON, timing."""
from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """worker_init_fn so DataLoader workers are also reproducible."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(allow_cpu: bool = False) -> torch.device:
    """Return the CUDA device, or raise unless CPU was explicitly allowed."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise SystemExit(
        "ERROR: no CUDA GPU detected. This pipeline is built for the RTX 3050.\n"
        "       If you really want to run on CPU (very slow), pass --allow-cpu."
    )


def describe_device(device: torch.device) -> str:
    """Human-readable device string, including VRAM for a GPU."""
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        return f"{name} ({total:.1f} GB VRAM)"
    return "CPU"


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Console logger that optionally also writes to a file."""
    logger = logging.getLogger(name)
    if logger.handlers:                       # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def make_run_log_dir(results_dir: Path) -> Path:
    """Create a fresh timestamped log directory for one pipeline invocation.

    Returns ``results_dir/logs/run_<timestamp>``. If that name is already taken
    (two runs started within the same second) a ``_2``, ``_3`` ... suffix is
    appended. The chosen path is also recorded in ``results_dir/logs/latest.txt``
    (a plain text pointer -- robust on Windows, where symlinks need admin mode).
    """
    logs_root = Path(results_dir) / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = logs_root / f"run_{stamp}"
    suffix = 2
    while run_dir.exists():
        run_dir = logs_root / f"run_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    (logs_root / "latest.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    return run_dir


def stream_subprocess(
    cmd: list[str],
    cwd,
    log_path: Path,
    err_path: Path,
    env: dict | None = None,
) -> int:
    """Run ``cmd``, teeing its output live to the console while persisting it.

    Writes a merged ``log_path`` (stdout + stderr in chronological order) and a
    stderr-only ``err_path`` -- the SLURM ``--output`` / ``--error`` style.
    Returns the process exit code.
    """
    log_path = Path(log_path)
    err_path = Path(err_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    with open(log_path, "w", encoding="utf-8") as log_f, \
            open(err_path, "w", encoding="utf-8") as err_f:
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.write(f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_f.flush()

        # text mode -> universal newlines, so tqdm's '\r' progress updates are
        # split into individual lines, just as they appear in a SLURM .err file.
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        def pump(src, console, is_err: bool) -> None:
            """Forward one stream to the console, the merged log, and (stderr
            only) the .err file. Whole-line writes under a lock keep the merged
            log roughly chronological."""
            for line in src:
                with lock:
                    try:
                        console.write(line)
                        console.flush()
                    except (ValueError, OSError):
                        pass               # console closed/redirected -- ignore
                    log_f.write(line)
                    log_f.flush()
                    if is_err:
                        err_f.write(line)
                        err_f.flush()
            src.close()

        threads = [
            threading.Thread(target=pump, args=(proc.stdout, sys.stdout, False),
                             daemon=True),
            threading.Thread(target=pump, args=(proc.stderr, sys.stderr, True),
                             daemon=True),
        ]
        for thread in threads:
            thread.start()
        proc.wait()
        for thread in threads:
            thread.join()

        log_f.write(f"\n# finished {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"| exit {proc.returncode}\n")
    return proc.returncode


def save_json(obj, path: Path) -> None:
    """Write ``obj`` to ``path`` as pretty JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def load_json(path: Path):
    """Read JSON from ``path``."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class Timer:
    """Context manager that measures wall-clock time."""

    def __enter__(self) -> "Timer":
        self.start = time.time()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.time() - self.start

    @property
    def minutes(self) -> float:
        return (time.time() - self.start) / 60
