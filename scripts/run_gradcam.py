#!/usr/local/bin/python3.10
"""
Grad-CAM visualization script.

Crops the 10 leaf images from preliminary_data.png and runs Grad-CAM
using the trained AlexNet and GoogLeNet binary (healthy vs disease) models.

Run from the repo root:
    python3.10 scripts/run_gradcam.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from collections import OrderedDict

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.expanduser("~/AI_DL_Project/AI-and-Deep-Learning-Group-8--6165")
GRID   = os.path.join(BASE, "healthy_disease/output/images/preliminary_data.png")
OUT    = os.path.join(BASE, "healthy_disease/output/images/gradcam")
os.makedirs(OUT, exist_ok=True)

# Best split model (80% train / 10% val / 10% test)
ALEXNET_PT   = os.path.join(BASE, "healthy_disease/models/alexnet_model_test-size_.2.pt")
GOOGLENET_PT = os.path.join(BASE, "healthy_disease/models/googlenet_model_test-size_.2.pt")

sys.path.insert(0, os.path.join(BASE, "scripts"))
from gradcam import GradCAM, get_target_layer

# ── image preprocessing ────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def preprocess(pil_img):
    """Return (tensor [1,3,224,224], display numpy [224,224,3] uint8)."""
    rgb = pil_img.convert("RGB")
    display = np.array(rgb.resize((224, 224)))
    tensor  = transform(rgb).unsqueeze(0)
    return tensor, display


# ── model builders ─────────────────────────────────────────────────────────────

def load_alexnet(path):
    model = models.alexnet(weights=None)
    model.classifier[-1] = nn.Linear(4096, 1)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def load_googlenet(path):
    model = models.googlenet(weights=None, aux_logits=False)
    model.fc = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(1024, 500)),
        ("relu", nn.ReLU()),
        ("fc2", nn.Linear(500, 1)),
    ]))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ── prediction ──────────────────────────────────────────────────────────────────

def predict(model, tensor):
    with torch.no_grad():
        logit = model(tensor).squeeze()
    prob  = torch.sigmoid(logit).item()
    label = "Healthy" if prob > 0.5 else "Disease"
    conf  = prob if prob > 0.5 else 1 - prob
    return label, conf


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

def run_gradcam(model, tensor, arch):
    target = get_target_layer(model, arch)
    gcam   = GradCAM(model, target)
    t      = tensor.clone().requires_grad_(True)
    cam, _ = gcam.generate(t)
    gcam.remove_hooks()
    return cam


def overlay(cam, display_np, alpha=0.45):
    import cv2
    cam_r = cv2.resize(cam, (224, 224))
    heat  = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blend = (alpha * heat + (1 - alpha) * display_np).astype(np.uint8)
    return blend, heat


# ── crop the 10-leaf grid ──────────────────────────────────────────────────────

def crop_leaves(grid_path):
    """
    preliminary_data.png is a 5-column × 2-row grid (1000×500 px).
    Returns list of 10 PIL images, left-to-right, top-to-bottom.
    """
    grid = Image.open(grid_path).convert("RGB")
    W, H = grid.size   # 1000, 500
    cols, rows = 5, 2
    cw, ch = W // cols, H // rows

    leaves = []
    for r in range(rows):
        for c in range(cols):
            box  = (c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)
            leaves.append(grid.crop(box))
    return leaves


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading models…")
    alexnet   = load_alexnet(ALEXNET_PT)
    googlenet = load_googlenet(GOOGLENET_PT)

    print("Cropping leaves from preliminary_data.png…")
    leaves = crop_leaves(GRID)

    for i, leaf in enumerate(leaves):
        print(f"\n── Leaf {i+1}/10 ──")
        tensor, display = preprocess(leaf)

        for model, arch, name in [
            (alexnet,   "alexnet",   "AlexNet"),
            (googlenet, "googlenet", "GoogLeNet"),
        ]:
            label, conf = predict(model, tensor)
            cam         = run_gradcam(model, tensor, arch)
            blend, heat = overlay(cam, display)

            print(f"  {name}: {label} ({conf*100:.1f}%)")

            # ── save figure ──
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
            fig.suptitle(f"Leaf {i+1} | {name} | Pred: {label} ({conf*100:.1f}%)",
                         fontsize=11, fontweight="bold")

            axes[0].imshow(display);  axes[0].set_title("Original");    axes[0].axis("off")
            axes[1].imshow(heat);     axes[1].set_title("Grad-CAM");     axes[1].axis("off")
            axes[2].imshow(blend);    axes[2].set_title("Overlay");      axes[2].axis("off")

            plt.tight_layout()
            fname = os.path.join(OUT, f"leaf{i+1:02d}_{arch}.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved → {fname}")

    # ── combined summary grid (AlexNet only) ──────────────────────────────────
    print("\nGenerating combined summary grid…")
    fig, axes = plt.subplots(10, 3, figsize=(10, 34))
    fig.suptitle("Grad-CAM Summary — AlexNet (Healthy vs Disease)", fontsize=13, fontweight="bold")

    col_titles = ["Original", "Grad-CAM", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")

    for i, leaf in enumerate(leaves):
        tensor, display = preprocess(leaf)
        label, conf     = predict(alexnet, tensor)
        cam             = run_gradcam(alexnet, tensor, "alexnet")
        blend, heat     = overlay(cam, display)

        axes[i][0].imshow(display);  axes[i][0].set_ylabel(f"Leaf {i+1}\n{label} {conf*100:.0f}%",
                                                             fontsize=8, rotation=0, labelpad=55)
        axes[i][1].imshow(heat)
        axes[i][2].imshow(blend)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    summary_path = os.path.join(OUT, "gradcam_summary_alexnet.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary saved → {summary_path}")
    print("\nDone! All results in:", OUT)


if __name__ == "__main__":
    main()
