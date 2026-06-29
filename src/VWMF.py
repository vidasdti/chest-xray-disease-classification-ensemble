# =====================================================
# VALIDATION-INFORMED WEIGHTED SOFT FUSION (Q1-SAFE)
# ConvNeXt Tiny + ResNet50
# BP + VP -> Pneumonia (Evaluation Only)
# =====================================================

import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch import nn

from collections import defaultdict
from tqdm import tqdm

from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import *
from dataset import create_dataloaders

# =====================================================
# Config
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Merge map (Evaluation space)
MERGE_MAP = {
    0: 0,  # BP -> Pneumonia
    1: 0,  # VP -> Pneumonia
    2: 1,  # COVID
    3: 2,  # Normal
    4: 3   # TB
}
MERGED_CLASS_NAMES = ["Pneumonia", "COVID", "Normal", "TB"]


_, val_loader, test_loader = create_dataloaders()

print(f"Validation samples : {len(val_loader.dataset)}")
print(f"Test samples       : {len(test_loader.dataset)}")

# =====================================================
# Load Models
# =====================================================
def load_convnext():
    m = models.convnext_tiny(weights=None)
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f, len(CLASSES))
    )
    m.load_state_dict(torch.load(
        MODEL_DIR / "convnext_advanced.pt",
        map_location=device, weights_only=True
    ))
    return m.to(device).eval()

def load_resnet():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, len(CLASSES))
    )
    m.load_state_dict(torch.load(
        MODEL_DIR / "resnet50_advanced.pt",
        map_location=device, weights_only=True
    ))
    return m.to(device).eval()

convnext = load_convnext()
resnet   = load_resnet()

# =====================================================
# Step 1: Per-class accuracy on VALIDATION
# =====================================================
def per_class_acc(model, loader):
    correct = defaultdict(int)
    total   = defaultdict(int)

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            for t, p in zip(yb.cpu(), preds.cpu()):
                total[int(t)] += 1
                if t == p:
                    correct[int(t)] += 1

    return {c: correct[c] / max(1, total[c]) for c in range(len(CLASSES))}

acc_c = per_class_acc(convnext, val_loader)
acc_r = per_class_acc(resnet, val_loader)

# =====================================================
# Compute fusion weights
# =====================================================
weights = {}
for c in range(len(CLASSES)):
    s = acc_c[c] + acc_r[c]
    weights[c] = {
        "convnext": acc_c[c] / s,
        "resnet":   acc_r[c] / s
    }

print("\n🔧 Learned Fusion Weights (Validation)")
for i, c in enumerate(CLASSES):
    print(f"{c}: ConvNeXt={weights[i]['convnext']:.3f}, "
          f"ResNet={weights[i]['resnet']:.3f}")

# =====================================================
# Step 2: Test-time Fusion + MERGE
# =====================================================
y_true, y_pred = [], []

with torch.no_grad():
    for xb, yb in tqdm(test_loader, desc="FUSION TEST"):
        xb = xb.to(device)

        p_c = torch.softmax(convnext(xb), dim=1)
        p_r = torch.softmax(resnet(xb), dim=1)

        fused = torch.zeros_like(p_c)

        for c in range(len(CLASSES)):
            fused[:, c] = (
                weights[c]["convnext"] * p_c[:, c] +
                weights[c]["resnet"]   * p_r[:, c]
            )

        preds = fused.argmax(1).cpu().numpy()

        #  MERGE BP + VP -> Pneumonia
        for gt, pr in zip(yb.numpy(), preds):
            y_true.append(MERGE_MAP[gt])
            y_pred.append(MERGE_MAP[pr])

# =====================================================
# Results (Merged Evaluation)
# =====================================================
cm = confusion_matrix(y_true, y_pred)

print("\n FUSION RESULTS")
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Macro F1 : {f1_score(y_true, y_pred, average='macro'):.4f}")

print(classification_report(
    y_true,
    y_pred,
    target_names=MERGED_CLASS_NAMES,
    digits=4
))

print(cm)