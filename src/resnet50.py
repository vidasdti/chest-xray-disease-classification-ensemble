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

MIXUP_ALPHA = 0.4


def mixup(images, labels, alpha=MIXUP_ALPHA):

    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)

    index = torch.randperm(images.size(0)).to(images.device)

    mixed_images = (
        lam * images +
        (1.0 - lam) * images[index]
    )

    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


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

        for shadow_param, param in zip(
            self.shadow.values(),
            model.state_dict().values()
        ):

            if shadow_param.dtype.is_floating_point:
                shadow_param.mul_(self.decay).add_(
                    param,
                    alpha=1-self.decay
                )
            else:
                shadow_param.copy_(param)

    def apply_shadow(self, model):

        self.backup = deepcopy(model.state_dict())

        model.load_state_dict(self.shadow)

    def restore(self, model):

        if self.backup is not None:

            model.load_state_dict(self.backup)

            self.backup = None


# =====================================================
# Model
# =====================================================

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(
    weights=weights
)

# Freeze everything

for param in model.parameters():
    param.requires_grad = False

# Fine-tune layer4

for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier

in_features = model.fc.in_features

model.fc = nn.Sequential(

    nn.Dropout(0.5),

    nn.Linear(
        in_features,
        NUM_CLASSES,
    )

)

model = model.to(device)

ema = EMA(model, decay=0.999)

# =====================================================
# Loss
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

scheduler = CosineAnnealingLR(

    optimizer,

    T_max=NUM_EPOCHS,

)

# =====================================================
# Training settings
# =====================================================

GRAD_CLIP = 2.0

best_val_f1 = 0.0

patience_counter = 0

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = MODEL_DIR / "best_resnet50.pt"

print("\n Training ResNet50...\n")

# =====================================================
# Training
# =====================================================

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()

    running_loss = 0.0

    train_preds = []
    train_targets = []

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch:02d}",
        ncols=100,
    )

    for images, labels in pbar:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images, labels_a, labels_b, lam = mixup(
            images,
            labels,
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = (
            lam * criterion(outputs, labels_a)
            + (1.0 - lam) * criterion(outputs, labels_b)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP,
        )

        optimizer.step()

        ema.update(model)

        running_loss += loss.item() * images.size(0)

        train_preds.extend(
            outputs.argmax(1).detach().cpu().numpy()
        )

        train_targets.extend(
            labels.detach().cpu().numpy()
        )

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = running_loss / len(train_loader.dataset)

    train_acc = accuracy_score(
        train_targets,
        train_preds,
    )

    train_f1 = f1_score(
        train_targets,
        train_preds,
        average="macro",
    )

    # =================================================
    # Validation (EMA Weights)
    # =================================================

    ema.apply_shadow(model)

    model.eval()

    val_loss = 0.0

    val_preds = []
    val_targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            val_loss += (
                loss.item() * images.size(0)
            )

            val_preds.extend(
                outputs.argmax(1).cpu().numpy()
            )

            val_targets.extend(
                labels.cpu().numpy()
            )

    val_loss /= len(val_loader.dataset)

    val_acc = accuracy_score(
        val_targets,
        val_preds,
    )

    val_f1 = f1_score(
        val_targets,
        val_preds,
        average="macro",
    )

    scheduler.step()

    print(
        f"\nEpoch {epoch:02d}"
        f" | Train Loss {train_loss:.4f}"
        f" | Train Acc {train_acc:.4f}"
        f" | Train F1 {train_f1:.4f}"
        f" | Val Loss {val_loss:.4f}"
        f" | Val Acc {val_acc:.4f}"
        f" | Val F1 {val_f1:.4f}"
    )

    # ===============================================
    # Early Stopping
    # ===============================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

        print(" Best model saved.")

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/{EARLY_STOPPING_PATIENCE})"
        )

    ema.restore(model)

    if patience_counter >= EARLY_STOPPING_PATIENCE:

        print("\n Early stopping triggered.")

        break
# =====================================================
# Load Best Model
# =====================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

model.eval()

# =====================================================
# Test Evaluation
# =====================================================

test_loss = 0.0

all_preds = []
all_targets = []

with torch.no_grad():

    for images, labels in tqdm(
        test_loader,
        desc="Testing",
        ncols=100,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        test_loss += (
            loss.item() * images.size(0)
        )

        all_preds.extend(
            outputs.argmax(1).cpu().numpy()
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

test_loss /= len(test_loader.dataset)

test_acc = accuracy_score(
    all_targets,
    all_preds,
)

test_f1 = f1_score(
    all_targets,
    all_preds,
    average="macro",
)

report = classification_report(
    all_targets,
    all_preds,
    target_names=CLASSES,
    digits=4,
)

cm = confusion_matrix(
    all_targets,
    all_preds,
)

# =====================================================
# Print Results
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
# Save Results
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

