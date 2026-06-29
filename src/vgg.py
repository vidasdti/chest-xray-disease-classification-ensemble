import torch
import numpy as np
import random
from torch import nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
from config import *
from dataset import create_dataloaders



random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = create_dataloaders()

print(f"Train samples : {len(train_loader.dataset)}")
print(f"Validation samples : {len(val_loader.dataset)}")
print(f"Test samples : {len(test_loader.dataset)}")

# =====================================================
# Model: VGG16 (Transfer Learning)
# =====================================================
model = models.vgg16(
    weights=models.VGG16_Weights.IMAGENET1K_V1
)

# Freeze convolutional backbone
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier = nn.Sequential(
    nn.Linear(25088, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, len(CLASSES))
)

model.to(device)

# =====================================================
# Training setup
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

best_val_f1 = 0.0
patience_counter = 0

# =====================================================
# Training loop
# =====================================================
print("\n Training VGG16 (Transfer Learning)...")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_preds, train_targets = [], []

    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_preds.extend(logits.argmax(1).cpu().numpy())
        train_targets.extend(yb.cpu().numpy())

    train_f1 = f1_score(train_targets, train_preds, average="macro")

    # ---------------- Validation ----------------
    model.eval()
    val_preds, val_targets = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            val_preds.extend(logits.argmax(1).cpu().numpy())
            val_targets.extend(yb.cpu().numpy())

    val_f1 = f1_score(val_targets, val_preds, average="macro")

    print(
        f"Epoch [{epoch:02d}] | "
        f"Train F1: {train_f1:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        best_model_path = MODEL_DIR / "vgg16_tl.pt"
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(" Early stopping triggered.")
            break

# =====================================================
# Test evaluation
# =====================================================
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_preds, all_targets = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_targets.extend(yb.cpu().numpy())

acc = accuracy_score(all_targets, all_preds)
macro_f1 = f1_score(all_targets, all_preds, average="macro")

print("\n========================================")
print(f" Test Accuracy : {acc:.4f}")
print(f" Test Macro-F1 : {macro_f1:.4f}")
print("========================================\n")

print(" Classification Report")
print(classification_report(
    all_targets,
    all_preds,
    target_names=CLASSES,
    digits=4
))

cm = confusion_matrix(all_targets, all_preds)
print("\n Confusion Matrix")
print(cm)