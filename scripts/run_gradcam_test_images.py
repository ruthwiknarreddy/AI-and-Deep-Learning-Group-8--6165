#!/usr/local/bin/python3.10
"""
Grad-CAM on real test images (local version).
Runs binary + multiclass Grad-CAM on images in test_images/
Saves results to healthy_disease/output/images/gradcam_test/
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

BASE = "/Users/ruthwiknarreddy/AI_DL_Project/AI-and-Deep-Learning-Group-8--6165"
IMG_DIR = os.path.join(BASE, "test_images")
OUT_BIN = os.path.join(BASE, "healthy_disease/output/images/gradcam_test")
OUT_MC  = os.path.join(BASE, "disease_type/output/images/gradcam_test")
os.makedirs(OUT_BIN, exist_ok=True)
os.makedirs(OUT_MC,  exist_ok=True)

sys.path.insert(0, os.path.join(BASE, "scripts"))
from gradcam import GradCAM, get_target_layer

# ── transform ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess(path):
    img     = Image.open(path).convert("RGB")
    display = np.array(img.resize((224, 224)))
    tensor  = transform(img).unsqueeze(0)
    return tensor, display

# ── parse true label from filename ────────────────────────────────────────────
def parse_label(filename):
    name = filename.upper()
    if "_HL " in name or "HEALTHY" in name or "___HL" in name:
        return "healthy", "Healthy"
    return "disease", os.path.splitext(filename)[0].split("___")[-1].strip()

# ── model builders ─────────────────────────────────────────────────────────────
def load_alexnet(path, n_out):
    m = models.alexnet(weights=None)
    m.classifier[-1] = nn.Linear(4096, n_out)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval(); return m

def load_googlenet(path, n_out):
    m = models.googlenet(weights=None, aux_logits=False)
    m.fc = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(1024, 500)),
        ("relu", nn.ReLU()),
        ("fc2", nn.Linear(500, n_out)),
    ]))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval(); return m

# ── Grad-CAM helpers ───────────────────────────────────────────────────────────
def run_gradcam(model, tensor, arch, class_idx=None):
    gcam   = GradCAM(model, get_target_layer(model, arch))
    cam, _ = gcam.generate(tensor.clone().requires_grad_(True), class_idx=class_idx)
    gcam.remove_hooks()
    return cam

def make_overlay(cam, display, alpha=0.45):
    import cv2
    cam_r = cv2.resize(cam, (224, 224))
    heat  = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blend = (alpha * heat + (1 - alpha) * display).astype(np.uint8)
    return blend, heat

def save_figure(display, heat, blend, title, path, analysis=""):
    fig = plt.figure(figsize=(12, 4))
    fig.suptitle(title, fontsize=10, fontweight="bold")

    ax1 = fig.add_subplot(1, 3, 1); ax1.imshow(display); ax1.set_title("Original");  ax1.axis("off")
    ax2 = fig.add_subplot(1, 3, 2); ax2.imshow(heat);    ax2.set_title("Grad-CAM");  ax2.axis("off")
    ax3 = fig.add_subplot(1, 3, 3); ax3.imshow(blend);   ax3.set_title("Overlay");   ax3.axis("off")

    if analysis:
        fig.text(0.5, -0.02, analysis, ha="center", fontsize=8,
                 style="italic", color="#444444")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

# ── analysis text ──────────────────────────────────────────────────────────────
def analyze(cam, true_binary, pred_label, conf):
    # cam is [224,224] normalized [0,1]
    # check if high activation (top 20%) is spread or focused
    threshold  = 0.7
    hot_ratio  = (cam > threshold).mean() * 100
    peak       = cam.max()
    correct    = pred_label.lower() == true_binary.lower()

    focus = "focused" if hot_ratio < 20 else "diffuse"
    if correct:
        if true_binary == "disease":
            msg = (f"Model correctly identified disease. Activation is {focus} "
                   f"({hot_ratio:.1f}% of image is high-attention). "
                   f"Peak activation={peak:.2f}. "
                   f"{'Likely focused on lesion/spot regions.' if focus=='focused' else 'Attention spread across leaf surface.'}")
        else:
            msg = (f"Model correctly identified healthy leaf. Activation is {focus} "
                   f"({hot_ratio:.1f}% of image is high-attention). "
                   f"No disease spots expected — model uses overall leaf texture.")
    else:
        msg = (f"Model MISCLASSIFIED (True: {true_binary}, Pred: {pred_label}). "
               f"Activation is {focus} ({hot_ratio:.1f}% high-attention). "
               f"Heatmap may reveal why the model was confused.")
    return msg

# ══════════════════════════════════════════════════════════════════════════════
# BINARY — Healthy vs Disease
# ══════════════════════════════════════════════════════════════════════════════
def run_binary():
    print("\n" + "="*60)
    print("TASK 1: Healthy vs Disease (Binary)")
    print("="*60)

    alex = load_alexnet(
        os.path.join(BASE, "healthy_disease/models/alexnet_model_test-size_.2.pt"), 1)
    goog = load_googlenet(
        os.path.join(BASE, "healthy_disease/models/googlenet_model_test-size_.2.pt"), 1)

    images = sorted([f for f in os.listdir(IMG_DIR)
                     if f.lower().endswith((".jpg",".jpeg",".png"))])

    results = []
    for fname in images:
        path              = os.path.join(IMG_DIR, fname)
        tensor, display   = preprocess(path)
        true_bin, true_disease = parse_label(fname)
        stem              = os.path.splitext(fname)[0][:40]

        print(f"\n  Image: {fname[:50]}")
        print(f"  True label: {true_bin} ({true_disease})")

        for model, arch, name in [(alex, "alexnet", "AlexNet"),
                                   (goog, "googlenet", "GoogLeNet")]:
            with torch.no_grad():
                prob = torch.sigmoid(model(tensor).squeeze()).item()
            pred = "healthy" if prob > 0.5 else "disease"
            conf = prob if prob > 0.5 else 1 - prob

            cam        = run_gradcam(model, tensor, arch)
            blend, heat= make_overlay(cam, display)
            correct    = pred == true_bin
            status     = "CORRECT" if correct else "WRONG"
            analysis   = analyze(cam, true_bin, pred, conf)

            print(f"  {name}: {pred} ({conf*100:.1f}%) [{status}]")
            print(f"    → {analysis}")

            title = f"{name} | True: {true_bin.upper()} | Pred: {pred.upper()} ({conf*100:.1f}%) [{status}]"
            safe  = stem.replace(" ","_")
            save_figure(display, heat, blend, title,
                        os.path.join(OUT_BIN, f"{safe}_{arch}.png"), analysis)

            results.append({"model": name, "true": true_bin,
                            "pred": pred, "conf": conf, "correct": correct})

    # summary
    for name in ["AlexNet", "GoogLeNet"]:
        r      = [x for x in results if x["model"] == name]
        acc    = sum(x["correct"] for x in r) / len(r) * 100
        print(f"\n  {name} accuracy on test images: {acc:.1f}% ({sum(x['correct'] for x in r)}/{len(r)})")

    print(f"\n  Results saved to: {OUT_BIN}")

# ══════════════════════════════════════════════════════════════════════════════
# MULTICLASS — Disease Type (33 classes)
# ══════════════════════════════════════════════════════════════════════════════
def run_multiclass():
    print("\n" + "="*60)
    print("TASK 2: Disease Type (Multiclass — 33 classes)")
    print("="*60)

    alex = load_alexnet(
        os.path.join(BASE, "disease_type/models/alexnet_model_test-size_.2.pt"), 33)
    goog = load_googlenet(
        os.path.join(BASE, "disease_type/models/googlenet_model_test-size_.2.pt"), 33)

    # only run on diseased images for multiclass
    images = sorted([f for f in os.listdir(IMG_DIR)
                     if f.lower().endswith((".jpg",".jpeg",".png"))])

    for fname in images:
        path             = os.path.join(IMG_DIR, fname)
        tensor, display  = preprocess(path)
        true_bin, true_disease = parse_label(fname)
        stem             = os.path.splitext(fname)[0][:40]

        print(f"\n  Image: {fname[:50]}")
        print(f"  True disease: {true_disease}")

        for model, arch, name in [(alex, "alexnet", "AlexNet"),
                                   (goog, "googlenet", "GoogLeNet")]:
            with torch.no_grad():
                logits = model(tensor)
            probs      = torch.softmax(logits, dim=1)
            conf, idx  = probs.max(dim=1)
            pred_idx   = idx.item()
            conf_val   = conf.item()

            # Grad-CAM for predicted class
            cam        = run_gradcam(model, tensor, arch, class_idx=pred_idx)
            blend, heat= make_overlay(cam, display)

            # analysis
            threshold = 0.7
            hot_ratio = (cam > threshold).mean() * 100
            focus     = "focused" if hot_ratio < 20 else "diffuse"
            analysis  = (f"Predicted class index {pred_idx} with {conf_val*100:.1f}% confidence. "
                         f"Grad-CAM attention is {focus} ({hot_ratio:.1f}% high-attention area). "
                         f"True disease: {true_disease}.")

            print(f"  {name}: class_idx={pred_idx} ({conf_val*100:.1f}%)")
            print(f"    → {analysis}")

            title = (f"{name} | True: {true_disease[:35]}\n"
                     f"Pred class: {pred_idx} ({conf_val*100:.1f}%)")
            safe  = stem.replace(" ","_")
            save_figure(display, heat, blend, title,
                        os.path.join(OUT_MC, f"{safe}_{arch}.png"), analysis)

    print(f"\n  Results saved to: {OUT_MC}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY GRID
# ══════════════════════════════════════════════════════════════════════════════
def make_summary_grid():
    print("\n" + "="*60)
    print("Generating Summary Grid...")

    images = sorted([f for f in os.listdir(IMG_DIR)
                     if f.lower().endswith((".jpg",".jpeg",".png"))])

    alex = load_alexnet(
        os.path.join(BASE, "healthy_disease/models/alexnet_model_test-size_.2.pt"), 1)

    n   = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    fig.suptitle("Grad-CAM Summary — AlexNet | Healthy vs Disease\nRed=High Attention · Blue=Low Attention",
                 fontsize=13, fontweight="bold", y=1.01)

    axes[0][0].set_title("Original",  fontweight="bold")
    axes[0][1].set_title("Grad-CAM",  fontweight="bold")
    axes[0][2].set_title("Overlay",   fontweight="bold")

    for i, fname in enumerate(images):
        path             = os.path.join(IMG_DIR, fname)
        tensor, display  = preprocess(path)
        true_bin, _      = parse_label(fname)

        with torch.no_grad():
            prob = torch.sigmoid(alex(tensor).squeeze()).item()
        pred    = "healthy" if prob > 0.5 else "disease"
        conf    = prob if prob > 0.5 else 1 - prob
        correct = "✓" if pred == true_bin else "✗"
        color   = "green" if pred == true_bin else "red"

        cam        = run_gradcam(alex, tensor, "alexnet")
        blend, heat= make_overlay(cam, display)

        axes[i][0].imshow(display)
        axes[i][1].imshow(heat)
        axes[i][2].imshow(blend)

        label = f"True: {true_bin} | Pred: {pred} ({conf*100:.0f}%) {correct}"
        axes[i][0].set_ylabel(label, fontsize=8, rotation=0,
                              labelpad=160, color=color, va="center")
        for ax in axes[i]: ax.axis("off")

    plt.tight_layout()
    out = os.path.join(OUT_BIN, "gradcam_summary_test_images.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary grid saved to: {out}")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_binary()
    run_multiclass()
    make_summary_grid()
    print("\n=== All Done ===")
