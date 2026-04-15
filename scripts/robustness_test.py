#!/usr/local/bin/python3.10
"""
Robustness Testing — Local version.

Evaluates AlexNet and GoogLeNet binary models (Healthy vs Disease)
under 5 types of image corruptions at 5 severity levels each.

Corruptions tested:
  1. Gaussian Noise
  2. Brightness Shift
  3. Contrast Shift
  4. Gaussian Blur
  5. JPEG Compression

Results saved to:
  healthy_disease/output/robustness/robustness_results.csv
  healthy_disease/output/images/robustness/
"""

import os, sys, io
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import f1_score, accuracy_score

BASE    = "/Users/ruthwiknarreddy/AI_DL_Project/AI-and-Deep-Learning-Group-8--6165"
IMG_DIR = os.path.join(BASE, "test_images")
OUT_CSV = os.path.join(BASE, "healthy_disease/output/robustness")
OUT_IMG = os.path.join(BASE, "healthy_disease/output/images/robustness")
os.makedirs(OUT_CSV, exist_ok=True)
os.makedirs(OUT_IMG, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, "scripts"))

# ── Normalization (same as training) ──────────────────────────────────────────
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def pil_to_tensor(img):
    return NORM(to_tensor(img.resize((224, 224)).convert("RGB"))).unsqueeze(0)

# ── Label parser ──────────────────────────────────────────────────────────────
def parse_label(filename):
    name = filename.upper()
    if "_HL " in name or "HEALTHY" in name or "___HL" in name:
        return 1  # healthy → positive
    return 0      # disease → negative

# ── Model loaders ─────────────────────────────────────────────────────────────
def load_alexnet(path):
    m = models.alexnet(weights=None)
    m.classifier[-1] = nn.Linear(4096, 1)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval(); return m

def load_googlenet(path):
    m = models.googlenet(weights=None, aux_logits=False)
    m.fc = nn.Sequential(OrderedDict([
        ("fc1",  nn.Linear(1024, 500)),
        ("relu", nn.ReLU()),
        ("fc2",  nn.Linear(500, 1)),
    ]))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval(); return m

def predict(model, tensor):
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor).squeeze()).item()
    return 1 if prob > 0.5 else 0  # 1=healthy, 0=disease

# ══════════════════════════════════════════════════════════════════════════════
# CORRUPTION FUNCTIONS  (each takes a PIL image, returns a PIL image)
# ══════════════════════════════════════════════════════════════════════════════

def add_gaussian_noise(img, severity):
    """severity = std of noise in [0,1] pixel space: 0.05 → 0.40"""
    stds = [0.05, 0.10, 0.20, 0.30, 0.40]
    arr  = np.array(img).astype(np.float32) / 255.0
    arr += np.random.randn(*arr.shape) * stds[severity - 1]
    arr  = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def shift_brightness(img, severity):
    """severity 1-5: darken to brighten (factor 0.25 → 2.0)"""
    factors = [0.25, 0.50, 0.75, 1.50, 2.00]
    return ImageEnhance.Brightness(img).enhance(factors[severity - 1])

def shift_contrast(img, severity):
    """severity 1-5: low contrast to high contrast"""
    factors = [0.25, 0.50, 0.75, 1.50, 2.00]
    return ImageEnhance.Contrast(img).enhance(factors[severity - 1])

def gaussian_blur(img, severity):
    """severity 1-5: blur radius 1 → 9"""
    radii = [1, 2, 3, 5, 9]
    return img.filter(ImageFilter.GaussianBlur(radius=radii[severity - 1]))

def jpeg_compression(img, severity):
    """severity 1-5: quality 10 → 80 (1=worst compression)"""
    qualities = [10, 20, 35, 50, 80]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=qualities[severity - 1])
    buf.seek(0)
    return Image.open(buf).copy()

CORRUPTIONS = {
    "Gaussian Noise":    add_gaussian_noise,
    "Brightness Shift":  shift_brightness,
    "Contrast Shift":    shift_contrast,
    "Gaussian Blur":     gaussian_blur,
    "JPEG Compression":  jpeg_compression,
}

SEVERITY_LABELS = {
    "Gaussian Noise":   ["σ=0.05", "σ=0.10", "σ=0.20", "σ=0.30", "σ=0.40"],
    "Brightness Shift": ["×0.25",  "×0.50",  "×0.75",  "×1.50",  "×2.00"],
    "Contrast Shift":   ["×0.25",  "×0.50",  "×0.75",  "×1.50",  "×2.00"],
    "Gaussian Blur":    ["r=1",    "r=2",    "r=3",    "r=5",    "r=9"],
    "JPEG Compression": ["q=10",   "q=20",   "q=35",   "q=50",   "q=80"],
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(42)

    alex = load_alexnet(
        os.path.join(BASE, "healthy_disease/models/alexnet_model_test-size_.2.pt"))
    goog = load_googlenet(
        os.path.join(BASE, "healthy_disease/models/googlenet_model_test-size_.2.pt"))

    images = sorted([f for f in os.listdir(IMG_DIR)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    true_labels = [parse_label(f) for f in images]

    # ── baseline (no corruption) ───────────────────────────────────────────
    print("Computing baseline...")
    alex_preds_base, goog_preds_base = [], []
    for fname in images:
        img    = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
        tensor = pil_to_tensor(img)
        alex_preds_base.append(predict(alex, tensor))
        goog_preds_base.append(predict(goog, tensor))

    base_alex_acc = accuracy_score(true_labels, alex_preds_base) * 100
    base_alex_f1  = f1_score(true_labels, alex_preds_base, average="macro") * 100
    base_goog_acc = accuracy_score(true_labels, goog_preds_base) * 100
    base_goog_f1  = f1_score(true_labels, goog_preds_base, average="macro") * 100
    print(f"  Baseline — AlexNet: Acc={base_alex_acc:.1f}% F1={base_alex_f1:.1f}%")
    print(f"  Baseline — GoogLeNet: Acc={base_goog_acc:.1f}% F1={base_goog_f1:.1f}%\n")

    # ── per-corruption per-severity ────────────────────────────────────────
    records = []
    for corr_name, corr_fn in CORRUPTIONS.items():
        print(f"Corruption: {corr_name}")
        for sev in range(1, 6):
            alex_preds, goog_preds = [], []
            for fname in images:
                img     = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
                img_c   = corr_fn(img, sev)
                tensor  = pil_to_tensor(img_c)
                alex_preds.append(predict(alex, tensor))
                goog_preds.append(predict(goog, tensor))

            alex_acc = accuracy_score(true_labels, alex_preds) * 100
            alex_f1  = f1_score(true_labels, alex_preds, average="macro") * 100
            goog_acc = accuracy_score(true_labels, goog_preds) * 100
            goog_f1  = f1_score(true_labels, goog_preds, average="macro") * 100

            sev_label = SEVERITY_LABELS[corr_name][sev - 1]
            print(f"  Sev {sev} ({sev_label:7s}) — AlexNet: Acc={alex_acc:.1f}% F1={alex_f1:.1f}%"
                  f"  |  GoogLeNet: Acc={goog_acc:.1f}% F1={goog_f1:.1f}%")

            records.append({
                "Corruption":    corr_name,
                "Severity":      sev,
                "SeverityLabel": sev_label,
                "AlexNet_Acc":   alex_acc,
                "AlexNet_F1":    alex_f1,
                "GoogLeNet_Acc": goog_acc,
                "GoogLeNet_F1":  goog_f1,
            })
        print()

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_CSV, "robustness_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}\n")

    # ── PLOTS ─────────────────────────────────────────────────────────────
    plot_accuracy_curves(df, base_alex_acc, base_goog_acc)
    plot_f1_curves(df, base_alex_f1, base_goog_f1)
    plot_heatmap(df)
    plot_sample_corruptions(images[5])  # show a disease image with all corruptions
    print(f"\nAll plots saved to: {OUT_IMG}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_curves(df, base_alex, base_goog):
    corrs = list(CORRUPTIONS.keys())
    fig, axes = plt.subplots(1, len(corrs), figsize=(20, 4), sharey=True)
    fig.suptitle("Model Accuracy Under Image Corruptions (Healthy vs Disease)",
                 fontsize=13, fontweight="bold")

    for ax, corr in zip(axes, corrs):
        sub = df[df["Corruption"] == corr]
        ax.axhline(base_alex, color="#2196F3", linestyle="--", alpha=0.5, linewidth=1, label="AlexNet baseline")
        ax.axhline(base_goog, color="#F44336", linestyle="--", alpha=0.5, linewidth=1, label="GoogLeNet baseline")
        ax.plot(sub["Severity"], sub["AlexNet_Acc"],   color="#2196F3", marker="o", linewidth=2, label="AlexNet")
        ax.plot(sub["Severity"], sub["GoogLeNet_Acc"], color="#F44336", marker="s", linewidth=2, label="GoogLeNet")
        ax.set_title(corr, fontsize=9)
        ax.set_xlabel("Severity")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(SEVERITY_LABELS[corr], fontsize=7, rotation=30)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy (%)")
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_IMG, "robustness_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: robustness_accuracy.png")


def plot_f1_curves(df, base_alex, base_goog):
    corrs = list(CORRUPTIONS.keys())
    fig, axes = plt.subplots(1, len(corrs), figsize=(20, 4), sharey=True)
    fig.suptitle("Model F1 Score Under Image Corruptions (Healthy vs Disease)",
                 fontsize=13, fontweight="bold")

    for ax, corr in zip(axes, corrs):
        sub = df[df["Corruption"] == corr]
        ax.axhline(base_alex, color="#2196F3", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(base_goog, color="#F44336", linestyle="--", alpha=0.5, linewidth=1)
        ax.plot(sub["Severity"], sub["AlexNet_F1"],   color="#2196F3", marker="o", linewidth=2, label="AlexNet")
        ax.plot(sub["Severity"], sub["GoogLeNet_F1"], color="#F44336", marker="s", linewidth=2, label="GoogLeNet")
        ax.set_title(corr, fontsize=9)
        ax.set_xlabel("Severity")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(SEVERITY_LABELS[corr], fontsize=7, rotation=30)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Macro F1 (%)")
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_IMG, "robustness_f1.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: robustness_f1.png")


def plot_heatmap(df):
    corrs = list(CORRUPTIONS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Robustness Heatmap — Accuracy Drop vs Baseline (%)",
                 fontsize=13, fontweight="bold")

    for ax, (model_col, model_name) in zip(axes,
            [("AlexNet_Acc", "AlexNet"), ("GoogLeNet_Acc", "GoogLeNet")]):

        matrix = np.zeros((len(corrs), 5))
        base   = df[df["Severity"] == 1][model_col].mean()  # rough baseline approx

        for i, corr in enumerate(corrs):
            sub = df[df["Corruption"] == corr]
            # show absolute accuracy per severity
            matrix[i] = sub[model_col].values

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_title(model_name, fontweight="bold")
        ax.set_yticks(range(len(corrs)))
        ax.set_yticklabels(corrs, fontsize=9)
        ax.set_xticks(range(5))
        ax.set_xlabel("Severity →")

        for i in range(len(corrs)):
            for j in range(5):
                ax.text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center",
                        fontsize=9, color="black" if matrix[i,j] > 40 else "white")

    plt.colorbar(im, ax=axes[1], label="Accuracy (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_IMG, "robustness_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: robustness_heatmap.png")


def plot_sample_corruptions(fname):
    """Show one disease image under all 5 corruptions × 5 severities."""
    img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB").resize((224, 224))
    corrs = list(CORRUPTIONS.keys())
    fig, axes = plt.subplots(len(corrs), 6, figsize=(16, 14))
    fig.suptitle(f"Sample Corruptions Applied to: {fname[:50]}",
                 fontsize=11, fontweight="bold")

    for row, (corr_name, corr_fn) in enumerate(CORRUPTIONS.items()):
        axes[row][0].imshow(img)
        axes[row][0].set_ylabel(corr_name, fontsize=8, rotation=30, labelpad=60, va="center")
        axes[row][0].set_title("Original" if row == 0 else "", fontsize=8)
        axes[row][0].axis("off")

        for sev in range(1, 6):
            img_c = corr_fn(img.copy(), sev)
            axes[row][sev].imshow(img_c)
            axes[row][sev].axis("off")
            if row == 0:
                axes[row][sev].set_title(SEVERITY_LABELS[corr_name][sev - 1], fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_IMG, "sample_corruptions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: sample_corruptions.png")


if __name__ == "__main__":
    main()
