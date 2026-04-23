#!/usr/local/bin/python3.10
"""
Plot training curves (loss, accuracy, F1, precision, recall) for all models and splits.
Saves plots to healthy_disease/output/images/training_curves/

Run from repo root:
    python3.10 scripts/plot_training_curves.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE  = "/Users/ruthwiknarreddy/AI_DL_Project/AI-and-Deep-Learning-Group-8--6165"
INDIR = os.path.join(BASE, "healthy_disease/output/train_test_results")
OUT   = os.path.join(BASE, "healthy_disease/output/images/training_curves")
os.makedirs(OUT, exist_ok=True)

MODELS = ["alexnet", "googlenet"]
SPLITS = [".2", ".4", ".5", ".6", ".8"]
SPLIT_LABELS = {
    ".2": "80/10/10",
    ".4": "60/20/20",
    ".5": "50/25/25",
    ".6": "40/30/30",
    ".8": "20/40/40",
}
METRICS = ["loss", "accuracy", "F1", "precision", "recall"]
COLORS  = {"train": "#2196F3", "valid": "#F44336"}


# ── 1. Per-model per-split: all 5 metrics in one figure ───────────────────────

for model in MODELS:
    for split in SPLITS:
        train_path = os.path.join(INDIR, f"{model}_train_history_test-size_{split}.csv")
        valid_path = os.path.join(INDIR, f"{model}_valid_history_test-size_{split}.csv")
        if not (os.path.exists(train_path) and os.path.exists(valid_path)):
            continue

        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(
            f"{model.capitalize()} | Split {SPLIT_LABELS[split]} (train/val/test)",
            fontsize=13, fontweight="bold"
        )

        for ax, metric in zip(axes, METRICS):
            if metric in train_df.columns:
                ax.plot(train_df[metric], color=COLORS["train"], label="Train", linewidth=1.8)
            if metric in valid_df.columns:
                ax.plot(valid_df[metric], color=COLORS["valid"], label="Validation", linewidth=1.8)
            ax.set_title(metric.capitalize())
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(OUT, f"{model}_split{split}_all_metrics.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


# ── 2. Comparison across splits: F1 and Accuracy for each model ───────────────

for model in MODELS:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model.capitalize()} — Validation Performance Across Train/Test Splits",
                 fontsize=13, fontweight="bold")

    cmap = plt.cm.get_cmap("tab10")

    for idx, split in enumerate(SPLITS):
        valid_path = os.path.join(INDIR, f"{model}_valid_history_test-size_{split}.csv")
        if not os.path.exists(valid_path):
            continue
        valid_df = pd.read_csv(valid_path)
        color    = cmap(idx)
        label    = SPLIT_LABELS[split]

        axes[0].plot(valid_df["accuracy"], color=color, label=label, linewidth=1.8)
        axes[1].plot(valid_df["F1"],       color=color, label=label, linewidth=1.8)

    for ax, metric in zip(axes, ["Accuracy", "F1 Score"]):
        ax.set_title(f"Validation {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend(title="Train/Val/Test", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(OUT, f"{model}_split_comparison.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# ── 3. AlexNet vs GoogLeNet side-by-side at best split (.2) ───────────────────

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
fig.suptitle("AlexNet vs GoogLeNet — Best Split (80/10/10) — Train & Validation",
             fontsize=13, fontweight="bold")

model_colors = {"alexnet": ("#1565C0", "#42A5F5"), "googlenet": ("#B71C1C", "#EF9A9A")}

for row, model in enumerate(MODELS):
    train_df = pd.read_csv(os.path.join(INDIR, f"{model}_train_history_test-size_.2.csv"))
    valid_df = pd.read_csv(os.path.join(INDIR, f"{model}_valid_history_test-size_.2.csv"))
    tc, vc   = model_colors[model]

    for col, metric in enumerate(METRICS):
        ax = axes[row][col]
        if metric in train_df.columns:
            ax.plot(train_df[metric], color=tc, label="Train",      linewidth=1.8)
        if metric in valid_df.columns:
            ax.plot(valid_df[metric], color=vc, label="Validation", linewidth=1.8, linestyle="--")
        ax.set_title(f"{model.capitalize()} — {metric.capitalize()}", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
fname = os.path.join(OUT, "alexnet_vs_googlenet_best_split.png")
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fname}")


# ── 4. Final validation metrics bar chart (best epoch per model/split) ─────────

records = []
for model in MODELS:
    for split in SPLITS:
        valid_path = os.path.join(INDIR, f"{model}_valid_history_test-size_{split}.csv")
        if not os.path.exists(valid_path):
            continue
        valid_df  = pd.read_csv(valid_path)
        best      = valid_df.loc[valid_df["F1"].idxmax()]
        records.append({
            "Model": model.capitalize(),
            "Split": SPLIT_LABELS[split],
            "Accuracy":  best["accuracy"],
            "F1":        best["F1"],
            "Precision": best["precision"],
            "Recall":    best["recall"],
        })

results = pd.DataFrame(records)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Best Validation Metrics — All Models & Splits", fontsize=13, fontweight="bold")

bar_metrics = ["Accuracy", "F1", "Precision", "Recall"]
x = range(len(SPLITS))
width = 0.35

for ax, metric in zip(axes, bar_metrics):
    alex = results[results["Model"] == "Alexnet"][metric].values
    goog = results[results["Model"] == "Googlenet"][metric].values
    bars1 = ax.bar([i - width/2 for i in x], alex, width, label="AlexNet",   color="#2196F3", alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], goog, width, label="GoogLeNet", color="#F44336", alpha=0.85)

    ax.set_title(metric)
    ax.set_xticks(list(x))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], fontsize=8, rotation=15)
    ax.set_ylim(0.95, 1.0)
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6)

plt.tight_layout()
fname = os.path.join(OUT, "best_metrics_bar_chart.png")
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fname}")

print(f"\nAll training curve plots saved to:\n{OUT}")
