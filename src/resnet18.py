import random
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch import nn
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# DataLoaders
# =====================================================
train_loader, val_loader, test_loader = create_dataloaders()

print(f"Train : {len(train_loader.dataset)}")
print(f"Val   : {len(val_loader.dataset)}")
print(f"Test  : {len(test_loader.dataset)}")

# =====================================================
# Model
# =====================================================
weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# Freeze all layers except layer4 and fc
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

model = model.to(device)

# =====================================================
# Loss & Optimizer
# =====================================================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
)

# =====================================================
# Training
# =====================================================
best_val_f1 = 0.0
patience_counter = 0

MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "resnet18.pt"

print("\n Training ResNet18 Transfer Learning...")

for epoch in range(1, NUM_EPOCHS + 1):

    # ---------------- Train ----------------
    model.train()

    train_preds = []
    train_targets = []

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_preds.extend(outputs.argmax(1).cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

    train_f1 = f1_score(
        train_targets,
        train_preds,
        average="macro",
    )

    # ---------------- Validation ----------------
    model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            val_preds.extend(outputs.argmax(1).cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_f1 = f1_score(
        val_targets,
        val_preds,
        average="macro",
    )

    print(
        f"Epoch [{epoch:02d}] | "
        f"Train F1: {train_f1:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

    else:

        patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print(" Early stopping triggered.")
            break

# =====================================================
# Test
# =====================================================
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model.eval()

all_preds = []
all_targets = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

# =====================================================
# Metrics
# =====================================================
acc = accuracy_score(all_targets, all_preds)

macro_f1 = f1_score(
    all_targets,
    all_preds,
    average="macro",
)

print("\n========================================")
print(f" Test Accuracy : {acc:.4f}")
print(f" Test Macro-F1 : {macro_f1:.4f}")
print("========================================\n")

print(" Classification Report")
print(
    classification_report(
        all_targets,
        all_preds,
        target_names=CLASSES,
        digits=4,
    )
)

print(" Confusion Matrix")
print(
    confusion_matrix(
        all_targets,
        all_preds,
    )
)