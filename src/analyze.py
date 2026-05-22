"""Stage 4 -- aggregate every run into one report.

Reads all eight training summaries plus the Grad-CAM attention statistics and
produces:
    results/summary.csv                  -- machine-readable table
    results/SUMMARY.md                   -- the human-readable report
    results/analysis/*.png               -- comparison charts
This is where the research question is answered:
    "Training on the past, how well can we predict the future?"

Run:  python -m src.analyze
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402
from matplotlib.patches import Rectangle     # noqa: E402

import config                                # noqa: E402
from src.utils import get_logger, load_json  # noqa: E402


# ===========================================================================
# LOADING
# ===========================================================================
def load_runs(logger) -> dict:
    """Return {(model, mode): summary_dict} for every completed run."""
    runs = {}
    for model in config.MODELS:
        for mode in config.FINETUNE_MODES:
            path = config.run_dir(model, mode) / "summary.json"
            if path.exists():
                runs[(model, mode)] = load_json(path)
            else:
                logger.warning(f"missing run summary: {path}")
    return runs


def load_attention(logger) -> dict:
    """Return {model: attention_shift_dict} for every Grad-CAM analysis."""
    attention = {}
    for model in config.MODELS:
        path = config.RESULTS_DIR / "gradcam" / model / "attention_shift.json"
        if path.exists():
            attention[model] = load_json(path)
        else:
            logger.warning(f"missing Grad-CAM summary: {path}")
    return attention


# ===========================================================================
# CHARTS
# ===========================================================================
def chart_val_vs_test(df: pd.DataFrame, path: Path) -> None:
    """Grouped bars: in-distribution (val) vs future (test) macro-F1."""
    labels = [f"{r.model}\n{r.mode}" for r in df.itertuples()]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    ax.bar(x - width / 2, df["val_f1"], width, label="VAL (2021-25 holdout)",
           color="#4C72B0")
    ax.bar(x + width / 2, df["test_f1"], width, label="TEST (2026 future)",
           color="#DD8452")
    ax.set_ylabel("macro-F1 (%)")
    ax.set_title("In-distribution vs future-year performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_generalization_gap(df: pd.DataFrame, path: Path) -> None:
    """Bar chart of the macro-F1 drop from past to future."""
    labels = [f"{r.model}\n{r.mode}" for r in df.itertuples()]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    colors = ["#C44E52" if g > 0 else "#55A868" for g in df["gen_gap_f1"]]
    ax.bar(x, df["gen_gap_f1"], 0.6, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("macro-F1 drop  (val - test, %)")
    ax.set_title("Generalization gap -- how much accuracy is lost on the future")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_partial_vs_full(pivot: pd.DataFrame, path: Path) -> None:
    """Grouped bars comparing test macro-F1 of full vs partial fine-tuning."""
    models = list(pivot.index)
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.4), 5))
    if "full" in pivot:
        ax.bar(x - width / 2, pivot["full"], width, label="full fine-tune",
               color="#4C72B0")
    if "partial" in pivot:
        ax.bar(x + width / 2, pivot["partial"], width,
               label="partial fine-tune", color="#8172B3")
    ax.set_ylabel("TEST macro-F1 (%)")
    ax.set_title("Future-year performance: full vs partial fine-tuning")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_cm_delta(model: str, runs: dict, path: Path) -> dict:
    """Heatmap of (full - partial) row-normalised test confusion matrices,
    with the single most-changed cell highlighted. Returns that cell's info.
    """
    full = runs[(model, "full")]["test"]
    partial = runs[(model, "partial")]["test"]
    classes = full["classes"]

    def norm(matrix):
        matrix = np.asarray(matrix, dtype=float)
        rows = matrix.sum(axis=1, keepdims=True)
        rows[rows == 0] = 1.0
        return matrix / rows

    delta = norm(full["confusion_matrix"]) - norm(partial["confusion_matrix"])

    # The "confusion portion that changed the most".
    bi, bj = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
    biggest = float(delta[bi, bj])

    fig, ax = plt.subplots(figsize=(6, 5))
    limit = max(abs(delta.min()), abs(delta.max()), 1e-6)
    image = ax.imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center",
                    color="black", fontsize=8)
    # Highlight the most-changed cell with a bright green box.
    ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                           edgecolor="#00CC44", linewidth=3))
    ax.set_title(
        f"{model}: TEST confusion-matrix change (full minus partial)\n"
        f"biggest shift -> true '{classes[bi]}' predicted '{classes[bj]}': "
        f"{biggest:+.2f}")
    fig.colorbar(image, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return dict(model=model, true_class=classes[bi], pred_class=classes[bj],
                delta=round(biggest, 3))


def chart_attention_change(attention: dict, path: Path) -> None:
    """Two intuitive panels: how SIMILAR the focus stays, and how FAR it moves,
    when going from partial to full fine-tuning."""
    models = [m for m in config.MODELS if m in attention]
    if not models:
        return
    correlation = [attention[m]["mean"]["correlation"] for m in models]
    hot_iou = [attention[m]["mean"]["hot_region_iou"] for m in models]
    centroid = [attention[m]["mean"]["centroid_shift"] for m in models]
    x = np.arange(len(models))
    width = 0.38

    fig, (ax_sim, ax_move) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: similarity (higher = full and partial look at the same place).
    ax_sim.bar(x - width / 2, correlation, width, label="CAM correlation",
               color="#4C72B0")
    ax_sim.bar(x + width / 2, hot_iou, width, label="hot-region IoU",
               color="#55A868")
    ax_sim.set_ylim(0, 1)
    ax_sim.set_ylabel("similarity  (1.0 = identical focus)")
    ax_sim.set_title("How similar partial & full attention stay")
    ax_sim.set_xticks(x)
    ax_sim.set_xticklabels(models, fontsize=9)
    ax_sim.legend()
    ax_sim.grid(axis="y", alpha=0.3)

    # Right: relocation (higher = the focus moved further).
    bars = ax_move.bar(x, centroid, 0.55, color="#C44E52")
    ax_move.set_ylabel("centroid shift  (fraction of image diagonal)")
    ax_move.set_title("How far the attention focus moves (partial -> full)")
    ax_move.set_xticks(x)
    ax_move.set_xticklabels(models, fontsize=9)
    ax_move.grid(axis="y", alpha=0.3)
    ax_move.bar_label(bars, fmt="%.3f", fontsize=8)

    fig.suptitle("Grad-CAM attention change: partial -> full fine-tuning")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_per_class_f1(runs: dict, path: Path) -> None:
    """Heatmap of per-class TEST F1 for every run -- shows which disease each
    model/mode actually handles well on the 2026 future set."""
    rows, labels, classes = [], [], None
    for model in config.MODELS:
        for mode in config.FINETUNE_MODES:
            if (model, mode) not in runs:
                continue
            test = runs[(model, mode)]["test"]
            classes = test["classes"]
            rows.append([test["per_class"][c]["f1"] * 100 for c in classes])
            labels.append(f"{model}\n{mode}")
    if not rows:
        return
    data = np.array(rows)

    fig, ax = plt.subplots(
        figsize=(1.5 * len(classes) + 3, 0.62 * len(labels) + 2))
    image = ax.imshow(data, cmap="YlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(classes)):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                    color="white" if data[i, j] > 60 else "black", fontsize=8)
    ax.set_title("Per-class F1 on the 2026 future set (per run)")
    fig.colorbar(image, fraction=0.046, pad=0.04, label="F1 (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===========================================================================
# REPORT
# ===========================================================================
def write_summary_md(df: pd.DataFrame, pivot: pd.DataFrame, attention: dict,
                     cm_changes: list, path: Path) -> None:
    """Write the human-readable SUMMARY.md."""
    lines: list[str] = []
    lines.append("# Rice Disease Recognition -- Cross-Year Results\n")
    lines.append("**Research question:** *Training on the past (2021-2025), "
                 "how well can the models predict the future (2026)?*\n")
    lines.append("- **VAL** = held-out 15% of 2021-2025 (in-distribution).")
    lines.append("- **TEST** = year 2026, never seen in training (the future).")
    lines.append("- **Generalization gap** = VAL macro-F1 - TEST macro-F1.")
    lines.append("- **Future retention** = TEST macro-F1 / VAL macro-F1.\n")

    # --- headline numbers ------------------------------------------------
    best = df.loc[df["test_f1"].idxmax()]
    lines.append("## Headline\n")
    lines.append(f"- Best future-year model: **{best['model']} "
                 f"({best['mode']})** -- TEST macro-F1 "
                 f"**{best['test_f1']:.2f}%**.")
    lines.append(f"- Average generalization gap across all runs: "
                 f"**{df['gen_gap_f1'].mean():.2f}** macro-F1 points.")
    lines.append(f"- Average future retention: "
                 f"**{df['future_retention_pct'].mean():.1f}%** "
                 f"of in-distribution performance.\n")

    # --- full results table ----------------------------------------------
    lines.append("## All runs\n")
    lines.append("| Model | Mode | VAL F1 | TEST F1 | TEST acc | "
                 "Gen. gap | Retention | Train min |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in df.itertuples():
        lines.append(
            f"| {r.model} | {r.mode} | {r.val_f1:.2f} | {r.test_f1:.2f} | "
            f"{r.test_acc:.2f} | {r.gen_gap_f1:+.2f} | "
            f"{r.future_retention_pct:.1f}% | {r.train_minutes:.1f} |")
    lines.append("")

    # --- partial vs full --------------------------------------------------
    lines.append("## Full vs partial fine-tuning (TEST macro-F1)\n")
    lines.append("| Model | Full | Partial | Full - Partial |")
    lines.append("|---|---|---|---|")
    for model in pivot.index:
        full_v = pivot.loc[model].get("full", float("nan"))
        part_v = pivot.loc[model].get("partial", float("nan"))
        lines.append(f"| {model} | {full_v:.2f} | {part_v:.2f} | "
                     f"{full_v - part_v:+.2f} |")
    lines.append("")

    # --- which confusion portion changed most ----------------------------
    if cm_changes:
        lines.append("## Confusion-matrix shift -- biggest change per model\n")
        lines.append("Row-normalised TEST confusion matrix, full minus "
                     "partial. A positive value means full fine-tuning sends "
                     "more of that true class to that prediction.\n")
        lines.append("| Model | True class | Predicted as | Change |")
        lines.append("|---|---|---|---|")
        for change in cm_changes:
            lines.append(f"| {change['model']} | {change['true_class']} | "
                         f"{change['pred_class']} | {change['delta']:+.2f} |")
        lines.append("\nThe highlighted (green box) cell in each "
                     "`analysis/cm_delta_<model>.png` marks this change.\n")

    # --- attention analysis ----------------------------------------------
    lines.append("## Grad-CAM: how far attention moves (partial -> full)\n")
    if attention:
        lines.append("| Model | CAM correlation | Hot-region IoU | "
                     "Centroid shift |")
        lines.append("|---|---|---|---|")
        for model in config.MODELS:
            if model in attention:
                m = attention[model]["mean"]
                lines.append(f"| {model} | {m['correlation']:.3f} | "
                             f"{m['hot_region_iou']:.3f} | "
                             f"{m['centroid_shift']:.3f} |")
        lines.append("\n- **CAM correlation** near 1.0 means full and partial "
                     "fine-tuning look at the same regions; lower means "
                     "fine-tuning the backbone substantially relocated the "
                     "model's focus.")
        lines.append("- **Hot-region IoU** = overlap of the top-25% most "
                     "attended pixels.")
        lines.append("- **Centroid shift** = how far the attention centre of "
                     "mass moved (fraction of the image diagonal).")
        lines.append("- See `gradcam/<model>/gradcam_summary.png` for the "
                     "5-image partial -> full progression.")
    else:
        lines.append("_No Grad-CAM results found -- run `python -m src.gradcam`._")
    lines.append("")

    # --- figures ----------------------------------------------------------
    lines.append("## Figures\n")
    lines.append("**Performance**")
    lines.append("- `analysis/val_vs_test_f1.png` -- in-distribution vs future.")
    lines.append("- `analysis/generalization_gap.png` -- accuracy lost on 2026.")
    lines.append("- `analysis/partial_vs_full_f1.png` -- fine-tuning strategies.")
    lines.append("- `analysis/per_class_f1.png` -- per-class F1 for every run.")
    lines.append("- `<model>_<mode>/training_curves.png` -- loss & F1 per epoch.")
    lines.append("- `<model>_<mode>/confusion_matrix_test.png` -- per run.\n")
    lines.append("**Fine-tuning & attention**")
    lines.append("- `analysis/cm_delta_<model>.png` -- confusion-matrix shift, "
                 "biggest-change cell highlighted.")
    lines.append("- `analysis/attention_change.png` -- how far Grad-CAM focus "
                 "moves, partial -> full.")
    lines.append("- `gradcam/<model>/gradcam_summary.png` -- 5 sample images, "
                 "partial -> full Grad-CAM progression.")
    lines.append("- `gradcam/<model>/NN_<class>_<img>.png` -- per-image "
                 "input | partial | full | change map.\n")

    path.write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    analysis_dir = config.RESULTS_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("analyze", config.RESULTS_DIR / "analyze.log")

    runs = load_runs(logger)
    if not runs:
        raise SystemExit("No run summaries found -- train the models first "
                         "(python run_pipeline.py).")

    rows = []
    for (model, mode), s in runs.items():
        rows.append(dict(
            model=model, mode=mode,
            val_acc=round(s["val"]["accuracy"], 2),
            val_f1=round(s["val"]["macro_f1"], 2),
            test_acc=round(s["test"]["accuracy"], 2),
            test_f1=round(s["test"]["macro_f1"], 2),
            test_balanced_acc=round(s["test"]["balanced_accuracy"], 2),
            gen_gap_f1=round(s["generalization_gap_f1"], 2),
            gen_gap_acc=round(s["generalization_gap_acc"], 2),
            future_retention_pct=round(s["future_retention_pct"], 1),
            trainable_params=s["trainable_params"],
            best_epoch=s["best_epoch"],
            train_minutes=round(s["train_minutes"], 1),
        ))
    df = pd.DataFrame(rows).sort_values(["model", "mode"]).reset_index(drop=True)
    df.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    logger.info(f"wrote {config.RESULTS_DIR / 'summary.csv'}")

    pivot = df.pivot_table(index="model", columns="mode", values="test_f1")

    # Performance charts.
    chart_val_vs_test(df, analysis_dir / "val_vs_test_f1.png")
    chart_generalization_gap(df, analysis_dir / "generalization_gap.png")
    chart_partial_vs_full(pivot, analysis_dir / "partial_vs_full_f1.png")
    chart_per_class_f1(runs, analysis_dir / "per_class_f1.png")

    # Confusion-matrix deltas (with the biggest change highlighted).
    cm_changes = []
    for model in config.MODELS:
        if (model, "full") in runs and (model, "partial") in runs:
            cm_changes.append(chart_cm_delta(
                model, runs, analysis_dir / f"cm_delta_{model}.png"))

    # Grad-CAM attention-change chart + report.
    attention = load_attention(logger)
    chart_attention_change(attention, analysis_dir / "attention_change.png")
    write_summary_md(df, pivot, attention, cm_changes,
                     config.RESULTS_DIR / "SUMMARY.md")

    logger.info("=" * 64)
    logger.info(f"Analysis complete. Report: {config.RESULTS_DIR / 'SUMMARY.md'}")
    logger.info(f"Average generalization gap: {df['gen_gap_f1'].mean():.2f} "
                f"macro-F1 points")
    logger.info(f"Average future retention  : "
                f"{df['future_retention_pct'].mean():.1f}%")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
