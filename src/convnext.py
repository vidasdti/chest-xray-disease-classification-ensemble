# =====================================================
# ConvNeXt Tiny Transfer Learning
# =====================================================

import torch
import random
import numpy as np
import sys
import os

from torchvision import models
from torch import nn
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *
from dataset import create_dataloaders
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)  

# =====================================================
# Reproducibility
# =====================================================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Load Data (ONLY THIS LINE)
# =====================================================
train_loader, val_loader, test_loader = create_dataloaders()

CLASS_NAMES = CLASSES
NUM_CLASSES = len(CLASSES)

# =====================================================
# Model
# =====================================================
model = models.convnext_tiny(weights="IMAGENET1K_V1")

# freeze backbone
for name, param in model.named_parameters():
    if "features.3" not in name:
        param.requires_grad = False

in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

# =====================================================
# Training setup
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "convnext_tiny.pt"

best_f1 = 0
patience = 0

# =====================================================
# Training loop
# =====================================================
for epoch in range(NUM_EPOCHS):

    model.train()
    preds, targets = [], []

    for x, y in tqdm(train_loader):

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(y.cpu().numpy())

    train_f1 = f1_score(targets, preds, average="macro")

    # -------- validation --------
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    val_f1 = f1_score(targets, preds, average="macro")

    print(f"Epoch {epoch} | Train {train_f1:.4f} | Val {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        patience = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience += 1
        if patience >= EARLY_STOPPING_PATIENCE:
            print("Early stopping")
            break

# =====================================================
# Test
# =====================================================
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():

    for x, y in tqdm(test_loader, desc="Testing"):

        x = x.to(device)
        y = y.to(device)

        out = model(x)

        all_preds.extend(out.argmax(1).cpu().numpy())
        all_targets.extend(y.cpu().numpy())

acc = accuracy_score(all_targets, all_preds)
macro_f1 = f1_score(all_targets, all_preds, average="macro")

print("\n========================================")
print(f"Test Accuracy : {acc:.4f}")
print(f"Test Macro-F1 : {macro_f1:.4f}")
print("========================================\n")

print("Classification Report")
print(classification_report(
    all_targets,
    all_preds,
    target_names=CLASSES,
    digits=4
))

print("Confusion Matrix")
print(confusion_matrix(all_targets, all_preds))