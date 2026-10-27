"""
Smoking Detection — YOLOv26s Model Evaluation
Generates classification report + confusion matrix from validation set.
Run this after training in the same Kaggle notebook.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIG — adjust paths if needed
# ─────────────────────────────────────────
WEIGHTS_PATH   = "runs/detect/train/weights/best.pt"
DATA_YAML      = "cigarettes-reality-2-trnm2-1/data.yaml"   # adjust to your dataset path
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
IMG_SIZE       = 800
OUTPUT_DIR     = "evaluation_results"
CLASS_NAMES    = ["cigarette"]   # update if your dataset has more classes

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────
print(f"Loading model from: {WEIGHTS_PATH}")
model = YOLO(WEIGHTS_PATH)

# ─────────────────────────────────────────
# 2. RUN VALIDATION
# ─────────────────────────────────────────
print("\nRunning validation on test/val set...")
val_results = model.val(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    verbose=True
)

# ─────────────────────────────────────────
# 3. EXTRACT METRICS FROM VAL RESULTS
# ─────────────────────────────────────────
metrics = val_results.results_dict

print("\n" + "="*55)
print("   YOLO VALIDATION METRICS")
print("="*55)

map50    = metrics.get("metrics/mAP50(B)",    0)
map5095  = metrics.get("metrics/mAP50-95(B)", 0)
precision = metrics.get("metrics/precision(B)", 0)
recall    = metrics.get("metrics/recall(B)",    0)
f1        = 2 * (precision * recall) / (precision + recall + 1e-9)

print(f"  mAP@50         : {map50:.4f}")
print(f"  mAP@50-95      : {map5095:.4f}")
print(f"  Precision      : {precision:.4f}")
print(f"  Recall         : {recall:.4f}")
print(f"  F1 Score       : {f1:.4f}")
print("="*55)

# ─────────────────────────────────────────
# 4. BUILD CLASSIFICATION REPORT
#    (image-level: predicted vs ground truth)
# ─────────────────────────────────────────
print("\nBuilding image-level classification report...")

import yaml
from glob import glob

with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

# Resolve val image path
val_path = data_cfg.get("val", "")
if not os.path.isabs(val_path):
    base = os.path.dirname(DATA_YAML)
    val_path = os.path.join(base, val_path)

image_files = (
    glob(os.path.join(val_path, "*.jpg"))
    + glob(os.path.join(val_path, "*.jpeg"))
    + glob(os.path.join(val_path, "*.png"))
)

if not image_files:
    print(f"  [WARN] No images found at: {val_path}")
    print("  Skipping image-level classification report.")
else:
    print(f"  Found {len(image_files)} validation images.")

    y_true = []
    y_pred = []

    label_dir = val_path.replace("images", "labels")

    for img_path in image_files:
        # Ground truth: does this image have at least one object?
        label_file = os.path.join(
            label_dir,
            Path(img_path).stem + ".txt"
        )
        has_gt = os.path.exists(label_file) and os.path.getsize(label_file) > 0
        gt_label = 1 if has_gt else 0

        # Prediction: does YOLO detect at least one object?
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        has_pred = len(results[0].boxes) > 0
        pred_label = 1 if has_pred else 0

        y_true.append(gt_label)
        y_pred.append(pred_label)

    target_names = ["No cigarette", "Cigarette"]

    print("\n" + "="*55)
    print("   CLASSIFICATION REPORT (Image Level)")
    print("="*55)
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)

    # Save report to text file
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("YOLO Smoking Detection — Classification Report\n")
        f.write("="*55 + "\n\n")
        f.write(f"Model     : {WEIGHTS_PATH}\n")
        f.write(f"Conf thr  : {CONF_THRESHOLD}\n")
        f.write(f"IoU thr   : {IOU_THRESHOLD}\n")
        f.write(f"Images    : {len(image_files)}\n\n")
        f.write(report)
        f.write(f"\nAccuracy  : {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"F1 (macro): {f1_score(y_true, y_pred, average='macro'):.4f}\n")
    print(f"\n  Report saved → {report_path}")

    # ─────────────────────────────────────────
    # 5. CONFUSION MATRIX
    # ─────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Smoking Detection — Confusion Matrix", fontsize=15, fontweight="bold", y=1.01)

    # --- Plot 1: Raw counts ---
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        linewidths=0.5,
        ax=axes[0],
        cbar=True,
        annot_kws={"size": 16, "weight": "bold"}
    )
    axes[0].set_title("Counts", fontsize=13)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)

    # --- Plot 2: Normalised (%) ---
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        linewidths=0.5,
        ax=axes[1],
        vmin=0, vmax=1,
        cbar=True,
        annot_kws={"size": 14}
    )
    axes[1].set_title("Normalised (%)", fontsize=13)
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Confusion matrix saved → {cm_path}")

    # ─────────────────────────────────────────
    # 6. METRICS SUMMARY BAR CHART
    # ─────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1s  = f1_score(y_true, y_pred, zero_division=0)

    metrics_names  = ["Accuracy", "Precision", "Recall", "F1 Score", "mAP@50", "mAP@50-95"]
    metrics_values = [acc, prec, rec, f1s, map50, map5095]
    colors = ["#3266ad", "#5b9bd5", "#70ad47", "#ffc000", "#ed7d31", "#a9d18e"]

    fig2, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor="white", width=0.55)

    for bar, val in zip(bars, metrics_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Evaluation Metrics", fontsize=14, fontweight="bold")
    ax.axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="0.9 threshold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    bar_path = os.path.join(OUTPUT_DIR, "metrics_summary.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Metrics chart saved → {bar_path}")

    # ─────────────────────────────────────────
    # 7. SAVE FULL JSON SUMMARY
    # ─────────────────────────────────────────
    summary = {
        "model": WEIGHTS_PATH,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "total_images": len(image_files),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1s, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map5095, 4),
        "confusion_matrix": cm.tolist()
    }

    json_path = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON summary saved  → {json_path}")

print("\n✅ Evaluation complete. All outputs saved to:", OUTPUT_DIR)
