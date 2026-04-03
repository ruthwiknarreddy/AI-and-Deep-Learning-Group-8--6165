#!/usr/bin/env python
"""
Grad-CAM visualization script — cluster version.

Reads dataset_split.csv, applies the same train/val/test split used in training,
then runs Grad-CAM on sampled test images for:
  - Task 1: Healthy vs Disease  (binary)
  - Task 2: Disease Type        (multiclass, 33 classes)

Results saved to:
  healthy_disease/output/images/gradcam/
  disease_type/output/images/gradcam/

Run via sbatch:
  bash sbatch/submit_sbatch_gradcam.sh 0.2
"""

import os
import sys
import pickle
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

## set working directory to repo root
os.chdir(f"{os.path.expanduser('~')}/AI-and-Deep-Learning-Group-8--6165/")
BASE = os.getcwd()
sys.path.insert(0, os.path.join(BASE, "scripts"))
from gradcam import GradCAM, get_target_layer

parser = argparse.ArgumentParser(prog="Grad-CAM cluster")
parser.add_argument("-ts", "--test_size", default="0.2")
args = parser.parse_args()

SAMPLES_PER_CLASS = 3
RANDOM_SEED       = 42

# ── image transform (same normalization as training) ──────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess(img_path):
    img     = Image.open(img_path).convert("RGB")
    display = np.array(img.resize((224, 224)))
    tensor  = transform(img).unsqueeze(0)
    return tensor, display

# ── model builders ─────────────────────────────────────────────────────────────
def load_alexnet(path, n_out):
    model = models.alexnet(weights=None)
    model.classifier[-1] = nn.Linear(4096, n_out)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def load_googlenet(path, n_out):
    model = models.googlenet(weights=None, aux_logits=False)
    model.fc = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(1024, 500)),
        ("relu", nn.ReLU()),
        ("fc2", nn.Linear(500, n_out)),
    ]))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# ── find model file ────────────────────────────────────────────────────────────
def find_model(folder, arch, test_size):
    ts = test_size.replace("0.", ".")
    for suffix in [ts, f"_{test_size}"]:
        p = f"./{folder}/models/{arch}_model_test-size{suffix}.pt"
        if os.path.exists(p):
            return p
    matches = glob.glob(f"./{folder}/models/{arch}_model*.pt")
    return matches[0] if matches else None

# ── prediction ─────────────────────────────────────────────────────────────────
def predict_binary(model, tensor):
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor).squeeze()).item()
    label = "Healthy" if prob > 0.5 else "Disease"
    conf  = prob if prob > 0.5 else 1 - prob
    return label, conf

def predict_multiclass(model, tensor, idx_to_label):
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    conf, idx = probs.max(dim=1)
    return idx_to_label.get(idx.item(), f"Class {idx.item()}"), conf.item()

# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def run_gradcam(model, tensor, arch, class_idx=None):
    gcam   = GradCAM(model, get_target_layer(model, arch))
    cam, _ = gcam.generate(tensor.clone().requires_grad_(True), class_idx=class_idx)
    gcam.remove_hooks()
    return cam

def make_overlay(cam, display_np, alpha=0.45):
    import cv2
    cam_r = cv2.resize(cam, (224, 224))
    heat  = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blend = (alpha * heat + (1 - alpha) * display_np).astype(np.uint8)
    return blend, heat

def save_figure(display, heat, blend, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.suptitle(title, fontsize=9, fontweight="bold")
    axes[0].imshow(display); axes[0].set_title("Original");  axes[0].axis("off")
    axes[1].imshow(heat);    axes[1].set_title("Grad-CAM");  axes[1].axis("off")
    axes[2].imshow(blend);   axes[2].set_title("Overlay");   axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

# ── test split helper ──────────────────────────────────────────────────────────
def get_test_df(label_col, test_size):
    df    = pd.read_csv("./dataset/dataset_split.csv")[["files", label_col]].dropna()
    _, tmp = train_test_split(df, test_size=float(test_size),
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
# TASK 1 — Healthy vs Disease
# ══════════════════════════════════════════════════════════════════════════════
def run_binary(test_size):
    OUT = "./healthy_disease/output/images/gradcam"
    os.makedirs(OUT, exist_ok=True)

    alex_path = find_model("healthy_disease", "alexnet",   test_size)
    goog_path = find_model("healthy_disease", "googlenet", test_size)
    print(f"[Binary] AlexNet  : {alex_path}")
    print(f"[Binary] GoogLeNet: {goog_path}")

    alexnet   = load_alexnet(alex_path,   n_out=1)
    googlenet = load_googlenet(goog_path, n_out=1)

    samples = sample_per_class(get_test_df("label_binary", test_size),
                               "label_binary", SAMPLES_PER_CLASS)
    print(f"[Binary] {len(samples)} test images selected\n")

    for _, row in samples.iterrows():
        img_path   = row["files"]
        true_label = row["label_binary"].capitalize()
        if not os.path.exists(img_path):
            print(f"  Skipping (not found): {img_path}")
            continue

        tensor, display = preprocess(img_path)
        stem = os.path.splitext(os.path.basename(img_path))[0]

        for model, arch, name in [(alexnet, "alexnet", "AlexNet"),
                                   (googlenet, "googlenet", "GoogLeNet")]:
            pred, conf = predict_binary(model, tensor)
            cam        = run_gradcam(model, tensor, arch)
            blend, heat= make_overlay(cam, display)
            correct    = "CORRECT" if pred.lower() == true_label.lower() else "WRONG"
            print(f"  {name} | True: {true_label} | Pred: {pred} ({conf*100:.1f}%) [{correct}]")
            save_figure(display, heat, blend,
                        f"{name} | True: {true_label} | Pred: {pred} ({conf*100:.1f}%) [{correct}]",
                        os.path.join(OUT, f"{stem}_{arch}.png"))

    print(f"\n[Binary] Saved to {OUT}\n")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Disease Type (33 classes)
# ══════════════════════════════════════════════════════════════════════════════
def run_multiclass(test_size):
    OUT = "./disease_type/output/images/gradcam"
    os.makedirs(OUT, exist_ok=True)

    pkl = "./dataset/disease_type_labels.pkl"
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            label_to_idx = pickle.load(f)
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        n_classes    = len(idx_to_label)
    else:
        print("[Multiclass] Warning: disease_type_labels.pkl not found.")
        idx_to_label = {}
        label_to_idx = {}
        n_classes    = 33

    alex_path = find_model("disease_type", "alexnet",   test_size)
    goog_path = find_model("disease_type", "googlenet", test_size)
    print(f"[Multiclass] AlexNet  : {alex_path}")
    print(f"[Multiclass] GoogLeNet: {goog_path}")

    alexnet   = load_alexnet(alex_path,   n_out=n_classes)
    googlenet = load_googlenet(goog_path, n_out=n_classes)

    samples = sample_per_class(get_test_df("disease_label", test_size),
                               "disease_label", n=1)
    print(f"[Multiclass] {len(samples)} test images selected\n")

    for _, row in samples.iterrows():
        img_path   = row["files"]
        true_label = row["disease_label"]
        if not os.path.exists(img_path):
            print(f"  Skipping (not found): {img_path}")
            continue

        tensor, display = preprocess(img_path)
        true_idx = label_to_idx.get(true_label, None)

        for model, arch, name in [(alexnet, "alexnet", "AlexNet"),
                                   (googlenet, "googlenet", "GoogLeNet")]:
            pred, conf = predict_multiclass(model, tensor, idx_to_label)
            cam        = run_gradcam(model, tensor, arch, class_idx=true_idx)
            blend, heat= make_overlay(cam, display)
            correct    = "CORRECT" if pred == true_label else "WRONG"
            print(f"  {name} | True: {true_label[:35]} | Pred: {pred[:35]} ({conf*100:.1f}%) [{correct}]")

            safe = true_label[:30].replace(" ","_").replace(":","").replace("/","")
            save_figure(display, heat, blend,
                        f"{name} | True: {true_label[:40]}\nPred: {pred[:40]} ({conf*100:.1f}%) [{correct}]",
                        os.path.join(OUT, f"{safe}_{arch}.png"))

    print(f"\n[Multiclass] Saved to {OUT}\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"=== Grad-CAM | test_size={args.test_size} ===\n")
    run_binary(args.test_size)
    run_multiclass(args.test_size)
    print("=== All done ===")
