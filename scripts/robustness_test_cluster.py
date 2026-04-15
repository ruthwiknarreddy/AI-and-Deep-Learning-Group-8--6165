#!/usr/bin/env python
"""
Robustness Testing — Cluster version.

Reads dataset_split.csv, applies same train/val/test split as training,
then evaluates AlexNet and GoogLeNet models under 5 corruption types
at 5 severity levels each.

Tasks:
  - Task 1: Healthy vs Disease (binary)
  - Task 2: Disease Type (multiclass, 33 classes)

Results saved to:
  healthy_disease/output/robustness/
  disease_type/output/robustness/

Run via sbatch:
  bash sbatch/submit_sbatch_robustness.sh 0.2
"""

import os, sys, io, glob, argparse, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

os.chdir(f"{os.path.expanduser('~')}/AI-and-Deep-Learning-Group-8--6165/")
BASE = os.getcwd()

parser = argparse.ArgumentParser(prog="Robustness Test cluster")
parser.add_argument("-ts", "--test_size", default="0.2")
args = parser.parse_args()

RANDOM_SEED       = 42
SAMPLES_PER_CLASS = 5   # images per class for evaluation

# ── Normalization ──────────────────────────────────────────────────────────────
NORM      = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def pil_to_tensor(img):
    return NORM(to_tensor(img.resize((224, 224)).convert("RGB"))).unsqueeze(0)

# ── Model loaders ──────────────────────────────────────────────────────────────
def load_alexnet(path, n_out):
    m = models.alexnet(weights=None)
    m.classifier[-1] = nn.Linear(4096, n_out)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval(); return m

def load_googlenet(path, n_out):
    m = models.googlenet(weights=None, aux_logits=False)
    m.fc = nn.Sequential(OrderedDict([
        ("fc1",  nn.Linear(1024, 500)),
        ("relu", nn.ReLU()),
        ("fc2",  nn.Linear(500, n_out)),
    ]))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval(); return m

def find_model(folder, arch, test_size):
    ts = test_size.replace("0.", ".")
    for suffix in [ts, f"_{test_size}"]:
        p = f"./{folder}/models/{arch}_model_test-size{suffix}.pt"
        if os.path.exists(p):
            return p
    matches = glob.glob(f"./{folder}/models/{arch}_model*.pt")
    return matches[0] if matches else None

def predict_binary(model, tensor):
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor).squeeze()).item()
    return 1 if prob > 0.5 else 0

def predict_multiclass(model, tensor):
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    return int(probs.argmax(dim=1).item())

# ── Data split ─────────────────────────────────────────────────────────────────
def get_test_df(label_col, test_size):
    df = pd.read_csv("./dataset/dataset_split.csv")[["files", label_col]].dropna()
    _, tmp  = train_test_split(df, test_size=float(test_size),
                               stratify=df[label_col], random_state=0)
    _, test = train_test_split(tmp, test_size=0.5,
                               stratify=tmp[label_col], random_state=0)
    return test

def sample_per_class(df, label_col, n):
    parts = []
    for cls in df[label_col].unique():
        rows = df[df[label_col] == cls]
        parts.append(rows.sample(min(n, len(rows)), random_state=RANDOM_SEED))
    return pd.concat(parts).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# CORRUPTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def add_gaussian_noise(img, severity):
    stds = [0.05, 0.10, 0.20, 0.30, 0.40]
    arr  = np.array(img).astype(np.float32) / 255.0
    arr += np.random.randn(*arr.shape) * stds[severity - 1]
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def shift_brightness(img, severity):
    factors = [0.25, 0.50, 0.75, 1.50, 2.00]
    return ImageEnhance.Brightness(img).enhance(factors[severity - 1])

def shift_contrast(img, severity):
    factors = [0.25, 0.50, 0.75, 1.50, 2.00]
    return ImageEnhance.Contrast(img).enhance(factors[severity - 1])

def gaussian_blur(img, severity):
    radii = [1, 2, 3, 5, 9]
    return img.filter(ImageFilter.GaussianBlur(radius=radii[severity - 1]))

def jpeg_compression(img, severity):
    qualities = [10, 20, 35, 50, 80]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=qualities[severity - 1])
    buf.seek(0)
    return Image.open(buf).copy()

CORRUPTIONS = {
    "Gaussian Noise":   add_gaussian_noise,
    "Brightness Shift": shift_brightness,
    "Contrast Shift":   shift_contrast,
    "Gaussian Blur":    gaussian_blur,
    "JPEG Compression": jpeg_compression,
}

SEVERITY_LABELS = {
    "Gaussian Noise":   ["σ=0.05", "σ=0.10", "σ=0.20", "σ=0.30", "σ=0.40"],
    "Brightness Shift": ["×0.25",  "×0.50",  "×0.75",  "×1.50",  "×2.00"],
    "Contrast Shift":   ["×0.25",  "×0.50",  "×0.75",  "×1.50",  "×2.00"],
    "Gaussian Blur":    ["r=1",    "r=2",    "r=3",    "r=5",    "r=9"],
    "JPEG Compression": ["q=10",   "q=20",   "q=35",   "q=50",   "q=80"],
}

# ══════════════════════════════════════════════════════════════════════════════
# SHARED EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(models_dict, img_paths, true_labels, predict_fn, out_dir, task_name):
    np.random.seed(RANDOM_SEED)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir.replace("robustness", "images/robustness"), exist_ok=True)

    records = []
    baseline = {}
    for model_name, model in models_dict.items():
        preds = []
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            preds.append(predict_fn(model, pil_to_tensor(img)))
        baseline[model_name] = {
            "acc": accuracy_score(true_labels, preds) * 100,
            "f1":  f1_score(true_labels, preds, average="macro") * 100,
        }
        print(f"  Baseline {model_name}: Acc={baseline[model_name]['acc']:.1f}% "
              f"F1={baseline[model_name]['f1']:.1f}%")

    for corr_name, corr_fn in CORRUPTIONS.items():
        print(f"\n  Corruption: {corr_name}")
        for sev in range(1, 6):
            row = {"Corruption": corr_name, "Severity": sev,
                   "SeverityLabel": SEVERITY_LABELS[corr_name][sev - 1]}
            for model_name, model in models_dict.items():
                preds = []
                for p in img_paths:
                    img   = Image.open(p).convert("RGB")
                    img_c = corr_fn(img, sev)
                    preds.append(predict_fn(model, pil_to_tensor(img_c)))
                acc = accuracy_score(true_labels, preds) * 100
                f1  = f1_score(true_labels, preds, average="macro") * 100
                row[f"{model_name}_Acc"] = acc
                row[f"{model_name}_F1"]  = f1
                print(f"    Sev {sev} ({SEVERITY_LABELS[corr_name][sev-1]:7s}) "
                      f"{model_name}: Acc={acc:.1f}% F1={f1:.1f}%")
            records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, f"robustness_{task_name}.csv"), index=False)
    print(f"\n  CSV saved to: {out_dir}/robustness_{task_name}.csv")

    # ── plots ──────────────────────────────────────────────────────────────
    img_out = out_dir.replace("robustness", "images/robustness")
    _plot_curves(df, baseline, models_dict, task_name, img_out)
    _plot_heatmap(df, models_dict, task_name, img_out)


def _plot_curves(df, baseline, models_dict, task_name, img_out):
    corrs  = list(CORRUPTIONS.keys())
    colors = {"AlexNet": "#2196F3", "GoogLeNet": "#F44336"}

    for metric, ylabel in [("Acc", "Accuracy (%)"), ("F1", "Macro F1 (%)")]:
        fig, axes = plt.subplots(1, len(corrs), figsize=(20, 4), sharey=True)
        fig.suptitle(f"{task_name} — Model {ylabel} Under Corruptions",
                     fontsize=12, fontweight="bold")
        for ax, corr in zip(axes, corrs):
            sub = df[df["Corruption"] == corr]
            for mname in models_dict:
                col = f"{mname}_{metric}"
                ax.axhline(baseline[mname][metric.lower()], color=colors[mname],
                           linestyle="--", alpha=0.4, linewidth=1)
                ax.plot(sub["Severity"], sub[col], color=colors[mname],
                        marker="o", linewidth=2, label=mname)
            ax.set_title(corr, fontsize=9)
            ax.set_xlabel("Severity")
            ax.set_xticks(range(1, 6))
            ax.set_xticklabels(SEVERITY_LABELS[corr], fontsize=7, rotation=30)
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            if ax == axes[0]:
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(img_out, f"robustness_{task_name}_{metric.lower()}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


def _plot_heatmap(df, models_dict, task_name, img_out):
    corrs = list(CORRUPTIONS.keys())
    fig, axes = plt.subplots(1, len(models_dict), figsize=(14, 5))
    fig.suptitle(f"{task_name} — Accuracy Heatmap by Corruption & Severity",
                 fontsize=12, fontweight="bold")
    axes = [axes] if len(models_dict) == 1 else axes
    for ax, mname in zip(axes, models_dict):
        matrix = np.array([
            df[df["Corruption"] == c][f"{mname}_Acc"].values for c in corrs
        ])
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_title(mname, fontweight="bold")
        ax.set_yticks(range(len(corrs)))
        ax.set_yticklabels(corrs, fontsize=9)
        ax.set_xticks(range(5))
        ax.set_xlabel("Severity →")
        for i in range(len(corrs)):
            for j in range(5):
                ax.text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center",
                        fontsize=9, color="black" if matrix[i,j] > 40 else "white")
    plt.colorbar(im, ax=axes[-1], label="Accuracy (%)")
    plt.tight_layout()
    out = os.path.join(img_out, f"robustness_{task_name}_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Binary
# ══════════════════════════════════════════════════════════════════════════════
def run_binary(test_size):
    print("\n" + "="*60)
    print("TASK 1: Healthy vs Disease (Binary)")
    print("="*60)

    alex_path = find_model("healthy_disease", "alexnet",   test_size)
    goog_path = find_model("healthy_disease", "googlenet", test_size)

    samples = sample_per_class(
        get_test_df("label_binary", test_size), "label_binary", SAMPLES_PER_CLASS)
    samples = samples[samples["files"].apply(os.path.exists)]
    print(f"  {len(samples)} test images")

    label_map  = {"healthy": 1, "disease": 0}
    true_labels = [label_map[r["label_binary"]] for _, r in samples.iterrows()]

    evaluate(
        {"AlexNet": load_alexnet(alex_path, 1), "GoogLeNet": load_googlenet(goog_path, 1)},
        list(samples["files"]), true_labels, predict_binary,
        "./healthy_disease/output/robustness", "binary"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Multiclass
# ══════════════════════════════════════════════════════════════════════════════
def run_multiclass(test_size):
    print("\n" + "="*60)
    print("TASK 2: Disease Type (Multiclass)")
    print("="*60)

    pkl = "./dataset/disease_type_labels.pkl"
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            label_to_idx = pickle.load(f)
        n_classes = len(label_to_idx)
    else:
        print("  Warning: label pkl not found, using 33 classes")
        label_to_idx = {}
        n_classes    = 33

    alex_path = find_model("disease_type", "alexnet",   test_size)
    goog_path = find_model("disease_type", "googlenet", test_size)

    samples = sample_per_class(
        get_test_df("disease_label", test_size), "disease_label", SAMPLES_PER_CLASS)
    samples = samples[samples["files"].apply(os.path.exists)]
    print(f"  {len(samples)} test images")

    true_labels = [label_to_idx.get(r["disease_label"], -1) for _, r in samples.iterrows()]

    evaluate(
        {"AlexNet": load_alexnet(alex_path, n_classes),
         "GoogLeNet": load_googlenet(goog_path, n_classes)},
        list(samples["files"]), true_labels, predict_multiclass,
        "./disease_type/output/robustness", "multiclass"
    )


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"=== Robustness Test | test_size={args.test_size} ===\n")
    run_binary(args.test_size)
    run_multiclass(args.test_size)
    print("\n=== All done ===")
