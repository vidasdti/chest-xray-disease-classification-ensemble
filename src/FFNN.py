import random
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from tqdm import tqdm
import torch.nn.functional as F
from config import *
from dataset import ChestDataset, eval_transform


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
# Dataset
# =====================================================
train_dataset = ChestDataset(
    roots=[
        PREPROCESSED_DATA_DIR / TRAIN_PHASE,
        AUGMENTED_DIR / TRAIN_PHASE,
    ],
    transform=eval_transform,
)

val_dataset = ChestDataset(
    roots=[PREPROCESSED_DATA_DIR / VALID_PHASE],
    transform=eval_transform,
)

test_dataset = ChestDataset(
    roots=[PREPROCESSED_DATA_DIR / TEST_PHASE],
    transform=eval_transform,
)

print(f"Train : {len(train_dataset)}")
print(f"Val   : {len(val_dataset)}")
print(f"Test  : {len(test_dataset)}")

# =====================================================
# DataLoaders for feature extraction
# =====================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)


# =====================================================
# MLP Classifier
# =====================================================
class MLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(

            nn.Flatten(),

            nn.Linear(3 * 128 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


NUM_CLASSES = len(CLASSES)

model = MLP(NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# =====================================================
# Training
# =====================================================
best_val_f1 = 0.0
patience_counter = 0

MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "FFNN.pt"

print("\n Training FFNN...\n")

for epoch in range(1, NUM_EPOCHS + 1):


    model.train()

    running_loss = 0.0

    train_preds = []
    train_targets = []

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)
        xb = F.interpolate(
            xb,
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )
        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

        train_preds.extend(logits.argmax(1).cpu().numpy())
        train_targets.extend(yb.cpu().numpy())

    train_loss = running_loss / len(train_loader.dataset)

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

        for xb, yb in val_loader:

            xb = xb.to(device)
            yb = yb.to(device)
            xb = F.interpolate(
                xb,
                size=(128, 128),
                mode="bilinear",
                align_corners=False
            )
            logits = model(xb)

            val_preds.extend(logits.argmax(1).cpu().numpy())
            val_targets.extend(yb.cpu().numpy())

    val_f1 = f1_score(
        val_targets,
        val_preds,
        average="macro",
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {train_loss:.4f} | "
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
            print("\n Early stopping triggered.")
            break

# =====================================================
# Test
# =====================================================
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

model.eval()

all_preds = []
all_targets = []

with torch.no_grad():

    for xb, yb in test_loader:

        xb = xb.to(device)
        yb = yb.to(device)
        xb = F.interpolate(
            xb,
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )
        logits = model(xb)

        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_targets.extend(yb.cpu().numpy())

overall_acc = accuracy_score(all_targets, all_preds)
macro_f1 = f1_score(all_targets, all_preds, average="macro")

print("\n========================================")
print(f"Test Accuracy : {overall_acc:.4f}")
print(f"Test Macro-F1 : {macro_f1:.4f}")
print("========================================\n")

print("Classification Report\n")

print(classification_report(all_targets, all_preds,target_names=CLASSES,digits=4))
print("Confusion Matrix\n")
print(confusion_matrix(all_targets, all_preds))