"""Sequential pipeline orchestrator (Windows-friendly, single machine).

Runs the whole study end to end:
    1. preprocessing          -- src/preprocess.py
    2. 8 training runs        -- src/train.py    (4 models x {full, partial})
    3. Grad-CAM comparisons   -- src/gradcam.py  (1 per model)
    4. cross-run analysis     -- src/analyze.py

Each stage runs as its OWN subprocess. That fully releases GPU memory between
runs -- important on a 4 GB card -- and isolates failures.

Every invocation creates a fresh per-run log directory under
``results/logs/run_<timestamp>/`` holding, for each step, a merged ``.log``
(stdout + stderr) and a stderr-only ``.err`` -- the SLURM ``--output`` /
``--error`` style -- plus a machine-readable ``run_summary.json``.

Examples
--------
    python run_pipeline.py                       # full pipeline
    python run_pipeline.py --skip-preprocess     # data already prepared
    python run_pipeline.py --models resnet50     # just one model
    python run_pipeline.py --continue-on-error   # don't stop on a failure
"""
from __future__ import annotations

import argparse
import platform
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.utils import (get_logger, make_run_log_dir, save_json,
                       stream_subprocess)

PYTHON = sys.executable


def run_step(cmd: list[str], step_name: str, run_log_dir: Path,
             logger, records: list[dict]) -> bool:
    """Run one subprocess step, capturing its output. Returns True on success.

    Tees the step's stdout/stderr live to the console while writing
    ``<step_name>.log`` (merged) and ``<step_name>.err`` (stderr only) into
    ``run_log_dir``, and appends a record of the step to ``records``.
    """
    log_path = run_log_dir / f"{step_name}.log"
    err_path = run_log_dir / f"{step_name}.err"
    logger.info(f"RUN [{step_name}]: " + " ".join(cmd))
    logger.info(f"  logs -> {log_path.name} / {err_path.name}")
    start = time.time()
    code = stream_subprocess(cmd, config.PROJECT_ROOT, log_path, err_path)
    minutes = (time.time() - start) / 60
    ok = code == 0
    records.append({
        "name": step_name,
        "cmd": " ".join(cmd),
        "exit_code": code,
        "minutes": round(minutes, 2),
        "status": "ok" if ok else "failed",
        "log": log_path.name,
        "err": err_path.name,
    })
    if not ok:
        logger.error(f"step [{step_name}] FAILED (exit {code}) after "
                     f"{minutes:.1f} min -- see {err_path}")
        return False
    logger.info(f"step [{step_name}] OK in {minutes:.1f} min")
    return True


def collect_environment() -> dict:
    """Best-effort snapshot of the runtime environment for run_summary.json."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    try:                                   # torch is heavy + optional here
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:               # noqa: BLE001 -- purely informational
        env["torch"] = f"<unavailable: {exc}>"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline.")
    parser.add_argument("--models", nargs="+", default=config.MODELS,
                        choices=config.MODELS, help="models to run")
    parser.add_argument("--modes", nargs="+", default=config.FINETUNE_MODES,
                        choices=config.FINETUNE_MODES, help="fine-tuning modes")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true",
                        help="rebuild the dataset even if it already exists")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="allow CPU execution if no GPU is present")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="keep going if a single step fails")
    args = parser.parse_args()

    # Re-emitted child output must not crash on a narrow console codepage.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_log_dir = make_run_log_dir(config.RESULTS_DIR)
    logger = get_logger("pipeline", run_log_dir / "pipeline.log")
    cpu_flag = ["--allow-cpu"] if args.allow_cpu else []
    failures: list[str] = []
    records: list[dict] = []
    step_no = 0
    pipeline_start = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 64)
    logger.info("RICE DISEASE -- CROSS-YEAR PIPELINE")
    logger.info(f"models={args.models}  modes={args.modes}")
    logger.info(f"run log directory: {run_log_dir}")
    logger.info("=" * 64)

    def next_name(label: str) -> str:
        """Ordered, zero-padded step name, e.g. '02_train_resnet50_full'."""
        nonlocal step_no
        step_no += 1
        return f"{step_no:02d}_{label}"

    try:
        # --- 1. preprocessing ---------------------------------------------
        if not args.skip_preprocess:
            cmd = [PYTHON, "-m", "src.preprocess"]
            if args.force_preprocess:
                cmd.append("--force")
            if (not run_step(cmd, next_name("preprocess"), run_log_dir,
                             logger, records)
                    and not args.continue_on_error):
                raise SystemExit("Preprocessing failed -- aborting.")
        else:
            logger.info("skipping preprocessing")

        # --- 2. training (up to 8 runs) -----------------------------------
        if not args.skip_train:
            for model in args.models:
                for mode in args.modes:
                    cmd = [PYTHON, "-m", "src.train",
                           "--model", model, "--mode", mode] + cpu_flag
                    if not run_step(cmd, next_name(f"train_{model}_{mode}"),
                                    run_log_dir, logger, records):
                        failures.append(f"train:{model}/{mode}")
                        if not args.continue_on_error:
                            raise SystemExit("Training failed -- aborting.")
        else:
            logger.info("skipping training")

        # --- 3. Grad-CAM analysis -----------------------------------------
        if not args.skip_gradcam:
            for model in args.models:
                cmd = [PYTHON, "-m", "src.gradcam", "--model", model] + cpu_flag
                if not run_step(cmd, next_name(f"gradcam_{model}"),
                                run_log_dir, logger, records):
                    failures.append(f"gradcam:{model}")
                    if not args.continue_on_error:
                        raise SystemExit("Grad-CAM failed -- aborting.")
        else:
            logger.info("skipping Grad-CAM")

        # --- 4. analysis ---------------------------------------------------
        if not args.skip_analyze:
            if not run_step([PYTHON, "-m", "src.analyze"], next_name("analyze"),
                            run_log_dir, logger, records):
                failures.append("analyze")
    finally:
        # Always leave a summary behind -- even on an early SystemExit abort.
        total_minutes = (time.time() - pipeline_start) / 60
        any_failed = failures or any(r["status"] == "failed" for r in records)
        summary = {
            "run_id": run_log_dir.name,
            "started": started_at,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_minutes": round(total_minutes, 2),
            "args": vars(args),
            "environment": collect_environment(),
            "steps": records,
            "failures": failures,
            "status": "completed_with_failures" if any_failed else "completed",
        }
        save_json(summary, run_log_dir / "run_summary.json")
        logger.info(f"run summary written to "
                    f"{run_log_dir / 'run_summary.json'}")

    logger.info("=" * 64)
    if failures:
        logger.warning(f"Pipeline finished in {total_minutes:.1f} min "
                       f"WITH FAILURES: {', '.join(failures)}")
        logger.info(f"Logs: {run_log_dir}")
        sys.exit(1)
    logger.info(f"PIPELINE COMPLETE in {total_minutes:.1f} min.")
    logger.info(f"Report: {config.RESULTS_DIR / 'SUMMARY.md'}")
    logger.info(f"Logs:   {run_log_dir}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
