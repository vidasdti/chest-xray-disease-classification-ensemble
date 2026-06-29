# =====================================================
# ResNet50 Advanced Training (Clean Version)
# MixUp + Label Smoothing + Cosine + EMA
# =====================================================

import random
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from copy import deepcopy

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from tqdm import tqdm

from config import *
from dataset import create_dataloaders

# =====================================================
# Reproducibility
# =====================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# EMA
# =====================================================

class EMA:

    def __init__(self, model, decay=0.999):

        self.decay = decay
        self.shadow = deepcopy(model.state_dict())
        self.backup = None

    @torch.no_grad()
    def update(self, model):

        current = model.state_dict()

        for k in self.shadow:
            if self.shadow[k].dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(
                    current[k],
                    alpha=1 - self.decay)
            else:
                self.shadow[k] = current[k].clone()

    def apply_shadow(self, model):
        self.backup = deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)

# =====================================================
# Data
# =====================================================

train_loader, val_loader, test_loader = create_dataloaders()

print(f"Train : {len(train_loader.dataset)}")
print(f"Val   : {len(val_loader.dataset)}")
print(f"Test  : {len(test_loader.dataset)}")

NUM_CLASSES = len(CLASSES)

# =====================================================
# MixUp
# =====================================================

def mixup(images, labels, alpha=0.4):

    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0)).to(images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam

# =====================================================
# Model
# =====================================================

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Freeze backbone

for p in model.parameters():
    p.requires_grad = False

# Fine-tune layer4

for p in model.layer4.parameters():
    p.requires_grad = True

in_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features,NUM_CLASSES))
model = model.to(device)

ema = EMA(model)

# =====================================================
# Loss / Optimizer
# =====================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

optimizer = torch.optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters(),
    ),

    lr=LEARNING_RATE,

    weight_decay=1e-4,
)

scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# =====================================================
# Training Settings
# =====================================================

GRAD_CLIP = 2.0

best_f1 = 0.0
patience = 0

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet50_advanced.pt"
print("\nTraining ResNet50...\n")

# =====================================================
# Training
# =====================================================

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0

    preds = []
    targets = []

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1:02d}",
        ncols=100,
    )

    for x, y in pbar:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x, y1, y2, lam = mixup(x, y)
        optimizer.zero_grad()
        out = model(x)
        loss = (lam * criterion(out, y1) + (1 - lam) * criterion(out, y2))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        ema.update(model)
        running_loss += loss.item() * x.size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(y.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = accuracy_score(targets, preds)
    train_f1 = f1_score(targets, preds, average="macro")

    # =================================================
    # Validation (EMA)
    # =================================================

    ema.apply_shadow(model)
    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            loss = criterion(out, y)
            val_loss += (loss.item() * x.size(0))
            val_preds.extend(out.argmax(1).cpu().numpy())
            val_targets.extend(y.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average="macro")
    scheduler.step()

    print(
        f"\nEpoch {epoch+1:02d}"
        f" | Train Loss {train_loss:.4f}"
        f" | Train Acc {train_acc:.4f}"
        f" | Train F1 {train_f1:.4f}"
        f" | Val Loss {val_loss:.4f}"
        f" | Val Acc {val_acc:.4f}"
        f" | Val F1 {val_f1:.4f}"
    )

    # =================================================
    # Save Best Model
    # =================================================

    if val_f1 > best_f1:

        best_f1 = val_f1
        patience = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(" Best model saved.")

    else:
        patience += 1
        print(f"No improvement "
            f"({patience}/{EARLY_STOPPING_PATIENCE})")

    ema.restore(model)

    if patience >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping triggered.")

        break

# =====================================================
# Load Best Model
# =====================================================

print("\nLoading best model...\n")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
# =====================================================
# Test
# =====================================================

test_loss = 0

preds = []
targets = []

with torch.no_grad():

    for x, y in tqdm(test_loader, desc="Testing", ncols=100):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        test_loss += (loss.item() * x.size(0))
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(y.cpu().numpy())

test_loss /= len(test_loader.dataset)

test_acc = accuracy_score(targets, preds)
test_f1 = f1_score(targets, preds, average="macro")

report = classification_report(
    targets,
    preds,
    target_names=CLASSES,
    digits=4,
)

cm = confusion_matrix(targets,preds)

# =====================================================
# Results
# =====================================================

print("\n========================================")
print("          TEST RESULTS")
print("========================================")

print(f"Loss      : {test_loss:.4f}")
print(f"Accuracy  : {test_acc:.4f}")
print(f"Macro F1  : {test_f1:.4f}")

print("\nClassification Report\n")
print(report)

print("\nConfusion Matrix\n")
print(cm)

# =====================================================
# Save Report
# =====================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

report_file = RESULT_DIR / "resnet50_report.txt"

with open(
    report_file,
    "w",
    encoding="utf-8",
) as f:

    f.write("========== ResNet50 ==========\n\n")

    f.write(f"Loss      : {test_loss:.4f}\n")
    f.write(f"Accuracy  : {test_acc:.4f}\n")
    f.write(f"Macro F1  : {test_f1:.4f}\n\n")

    f.write("Classification Report\n\n")
    f.write(report)

    f.write("\n\nConfusion Matrix\n\n")
    f.write(np.array2string(cm))

print(f"\nResults saved to:\n{report_file}")

print("\nDone.")