"""Pipeline orchestrator for the rice disease cross-year study.

Default behavior runs preprocessing once, then runs each selected seed as an
isolated experiment under results/experiment_seed_<seed>/.

Useful commands:
    python run_pipeline.py --runs 10 --start-seed 42
    python run_pipeline.py --seeds 42 43 44 45 46 47 48 49 50 51
    python run_pipeline.py --aggregate-only --seeds 42 43 44 45 46 47 48 49 50 51
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

PYTHON = sys.executable


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def make_run_log_dir(results_dir: Path) -> Path:
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


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def stream_subprocess(
    cmd: list[str],
    cwd,
    log_path: Path,
    err_path: Path,
    env: dict | None = None,
) -> int:
    log_path = Path(log_path)
    err_path = Path(err_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with open(log_path, "w", encoding="utf-8") as log_f, open(err_path, "w", encoding="utf-8") as err_f:
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.write(f"# started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_f.flush()
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
            for line in src:
                with lock:
                    try:
                        console.write(line)
                        console.flush()
                    except (ValueError, OSError):
                        pass
                    log_f.write(line)
                    log_f.flush()
                    if is_err:
                        err_f.write(line)
                        err_f.flush()
            src.close()

        threads = [
            threading.Thread(target=pump, args=(proc.stdout, sys.stdout, False), daemon=True),
            threading.Thread(target=pump, args=(proc.stderr, sys.stderr, True), daemon=True),
        ]
        for thread in threads:
            thread.start()
        proc.wait()
        for thread in threads:
            thread.join()
        log_f.write(f"\n# finished {time.strftime('%Y-%m-%d %H:%M:%S')} | exit {proc.returncode}\n")
    return proc.returncode


def resolve_seeds(parser: argparse.ArgumentParser, args) -> list[int]:
    if args.seeds:
        seeds = list(args.seeds)
    else:
        if args.runs < 1:
            parser.error("--runs must be at least 1")
        seeds = list(range(args.start_seed, args.start_seed + args.runs))
    if len(set(seeds)) != len(seeds):
        parser.error("duplicate seeds would overwrite the same experiment")
    return seeds


def experiment_name(seed: int, prefix: str | None, multi_seed: bool) -> str:
    if not prefix:
        return f"experiment_seed_{seed}"
    if "{seed}" in prefix:
        return prefix.format(seed=seed)
    return f"{prefix}_seed_{seed}" if multi_seed else prefix


def seed_env(seed: int, seed_results_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["RICE_RESULTS_DIR"] = str(seed_results_dir)
    env["RICE_EXPERIMENT_NAME"] = "."
    env["PYTHONHASHSEED"] = str(seed)
    env["RICE_ACTIVE_SEED"] = str(seed)
    return env


def run_step(cmd: list[str], name: str, log_dir: Path, logger, records: list[dict],
             env: dict[str, str] | None = None, extra: dict | None = None) -> bool:
    log_path = log_dir / f"{name}.log"
    err_path = log_dir / f"{name}.err"
    logger.info(f"RUN [{name}]: " + " ".join(cmd))
    if env and env.get("RICE_RESULTS_DIR"):
        logger.info(f"  child RICE_RESULTS_DIR={env['RICE_RESULTS_DIR']}")
    start = time.time()
    code = stream_subprocess(cmd, config.PROJECT_ROOT, log_path, err_path, env=env)
    minutes = (time.time() - start) / 60
    ok = code == 0
    record = {
        "name": name,
        "cmd": " ".join(cmd),
        "exit_code": code,
        "minutes": round(minutes, 2),
        "status": "ok" if ok else "failed",
        "log": log_path.name,
        "err": err_path.name,
    }
    if extra:
        record.update(extra)
    records.append(record)
    if ok:
        logger.info(f"step [{name}] OK in {minutes:.1f} min")
    else:
        logger.error(f"step [{name}] FAILED with exit {code}; see {err_path}")
    return ok


def collect_environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        env["torch"] = f"<unavailable: {exc}>"
    return env


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_seed_rows(experiments: list[dict], models: list[str], modes: list[str]):
    rows = []
    missing = []
    for experiment in experiments:
        seed = experiment["seed"]
        root = Path(experiment["results_dir"])
        for model in models:
            for mode in modes:
                path = root / f"{model}_{mode}" / "summary.json"
                if not path.exists():
                    missing.append({
                        "seed": seed,
                        "experiment": experiment["name"],
                        "model": model,
                        "mode": mode,
                        "path": str(path),
                    })
                    continue
                summary = read_json(path)
                rows.append({
                    "seed": seed,
                    "experiment": experiment["name"],
                    "model": model,
                    "mode": mode,
                    "val_acc": round(summary["val"]["accuracy"], 3),
                    "val_f1": round(summary["val"]["macro_f1"], 3),
                    "test_acc": round(summary["test"]["accuracy"], 3),
                    "test_f1": round(summary["test"]["macro_f1"], 3),
                    "gen_gap_acc": round(summary["generalization_gap_acc"], 3),
                    "gen_gap_f1": round(summary["generalization_gap_f1"], 3),
                    "future_retention_pct": round(summary["future_retention_pct"], 3),
                    "best_epoch": summary["best_epoch"],
                    "epochs_run": summary["epochs_run"],
                    "train_minutes": round(summary["train_minutes"], 3),
                    "summary_path": str(path.resolve()),
                })
    return rows, missing


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, std


def aggregate_rows(rows: list[dict], models: list[str], modes: list[str]) -> list[dict]:
    aggregate = []
    metrics = ["val_f1", "test_f1", "test_acc", "gen_gap_f1", "future_retention_pct", "train_minutes"]
    for model in models:
        for mode in modes:
            group = [row for row in rows if row["model"] == model and row["mode"] == mode]
            if not group:
                continue
            item = {"model": model, "mode": mode, "n": len(group), "seeds": [row["seed"] for row in group]}
            for metric in metrics:
                mean, std = mean_std([float(row[metric]) for row in group])
                item[f"{metric}_mean"] = round(mean, 3)
                item[f"{metric}_std"] = round(std, 3)
            aggregate.append(item)
    return aggregate


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["seed", "experiment", "model", "mode"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(seeds: list[int], rows: list[dict], aggregate: list[dict], missing: list[dict], path: Path) -> None:
    lines = [
        "# Multi-Seed Summary",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"Completed training summaries found: {len(rows)}",
        "",
        "## Aggregate by model/mode",
        "",
        "| Model | Mode | N | Test F1 | Val F1 | Gap F1 | Retention |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    def cell(item: dict, metric: str) -> str:
        return f"{item[f'{metric}_mean']:.2f} +/- {item[f'{metric}_std']:.2f}"

    for item in aggregate:
        lines.append(
            f"| {item['model']} | {item['mode']} | {item['n']} | "
            f"{cell(item, 'test_f1')} | {cell(item, 'val_f1')} | "
            f"{cell(item, 'gen_gap_f1')} | {cell(item, 'future_retention_pct')} |"
        )

    lines += [
        "",
        "## Per-seed runs",
        "",
        "| Seed | Model | Mode | Val F1 | Test F1 | Gap F1 | Retention |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["seed"], item["model"], item["mode"])):
        lines.append(
            f"| {row['seed']} | {row['model']} | {row['mode']} | "
            f"{row['val_f1']:.2f} | {row['test_f1']:.2f} | "
            f"{row['gen_gap_f1']:+.2f} | {row['future_retention_pct']:.1f}% |"
        )

    if missing:
        lines += [
            "",
            "## Missing summaries",
            "",
            "| Seed | Model | Mode | Expected path |",
            "|---:|---|---|---|",
        ]
        for item in missing:
            lines.append(f"| {item['seed']} | {item['model']} | {item['mode']} | `{item['path']}` |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_multi_seed_outputs(seeds: list[int], experiments: list[dict], models: list[str], modes: list[str]) -> dict:
    rows, missing = collect_seed_rows(experiments, models, modes)
    aggregate = aggregate_rows(rows, models, modes)
    write_csv(rows, config.RESULTS_DIR / "multi_seed_summary.csv")
    save_json({"seeds": seeds, "experiments": experiments, "runs": rows, "aggregate": aggregate, "missing": missing},
              config.RESULTS_DIR / "multi_seed_summary.json")
    write_markdown(seeds, rows, aggregate, missing, config.RESULTS_DIR / "MULTI_SEED_SUMMARY.md")
    return {"run_count": len(rows), "missing_count": len(missing)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or aggregate the rice disease pipeline.")
    parser.add_argument("--models", nargs="+", default=config.MODELS, choices=config.MODELS)
    parser.add_argument("--modes", nargs="+", default=config.FINETUNE_MODES, choices=config.FINETUNE_MODES)
    parser.add_argument("--start-seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--experiment-prefix", default=os.environ.get("RICE_EXPERIMENT_NAME"))
    parser.add_argument("--aggregate-only", action="store_true", help="only rebuild multi-seed summary files from existing seed folders")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    seeds = resolve_seeds(parser, args)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = make_run_log_dir(config.RESULTS_DIR)
    logger = get_logger("pipeline", log_dir / "pipeline.log")
    records = []
    failures = []
    step_no = 0
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    start = time.time()

    def next_name(label: str) -> str:
        nonlocal step_no
        step_no += 1
        return f"{step_no:02d}_{label}"

    multi = len(seeds) > 1
    experiments = [
        {
            "seed": seed,
            "name": experiment_name(seed, args.experiment_prefix, multi),
            "results_dir": str(config.RESULTS_DIR / experiment_name(seed, args.experiment_prefix, multi)),
        }
        for seed in seeds
    ]

    try:
        logger.info(f"models={args.models} modes={args.modes} seeds={seeds}")
        if not args.aggregate_only:
            if not args.skip_preprocess:
                cmd = [PYTHON, "-m", "src.preprocess"]
                if args.force_preprocess:
                    cmd.append("--force")
                if not run_step(cmd, next_name("preprocess"), log_dir, logger, records):
                    failures.append("preprocess")
                    if not args.continue_on_error:
                        raise SystemExit("Preprocessing failed")

            cpu_flag = ["--allow-cpu"] if args.allow_cpu else []
            for experiment in experiments:
                seed = experiment["seed"]
                env = seed_env(seed, Path(experiment["results_dir"]))
                if not args.skip_train:
                    for model in args.models:
                        for mode in args.modes:
                            cmd = [PYTHON, "-m", "src.train", "--model", model, "--mode", mode, "--seed", str(seed)] + cpu_flag
                            ok = run_step(cmd, next_name(f"seed{seed}_train_{model}_{mode}"), log_dir, logger, records, env=env)
                            if not ok:
                                failures.append(f"seed:{seed}:train:{model}/{mode}")
                                if not args.continue_on_error:
                                    raise SystemExit("Training failed")
                if not args.skip_gradcam:
                    for model in args.models:
                        cmd = [PYTHON, "-m", "src.gradcam", "--model", model] + cpu_flag
                        ok = run_step(cmd, next_name(f"seed{seed}_gradcam_{model}"), log_dir, logger, records, env=env)
                        if not ok:
                            failures.append(f"seed:{seed}:gradcam:{model}")
                            if not args.continue_on_error:
                                raise SystemExit("Grad-CAM failed")
                if not args.skip_analyze:
                    ok = run_step([PYTHON, "-m", "src.analyze"], next_name(f"seed{seed}_analyze"), log_dir, logger, records, env=env)
                    if not ok:
                        failures.append(f"seed:{seed}:analyze")
                        if not args.continue_on_error:
                            raise SystemExit("Analysis failed")

        multi_seed = write_multi_seed_outputs(seeds, experiments, args.models, args.modes)
        logger.info(f"multi-seed summary rows={multi_seed['run_count']} missing={multi_seed['missing_count']}")
    finally:
        total_minutes = (time.time() - start) / 60
        save_json({
            "started": started,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_minutes": round(total_minutes, 2),
            "args": vars(args),
            "seeds": seeds,
            "experiments": experiments,
            "environment": collect_environment(),
            "steps": records,
            "failures": failures,
            "status": "completed_with_failures" if failures else "completed",
        }, log_dir / "run_summary.json")

    if failures:
        sys.exit(1)
    logger.info(f"Done. Multi-seed report: {config.RESULTS_DIR / 'MULTI_SEED_SUMMARY.md'}")


if __name__ == "__main__":
    main()

