# =====================================================
# ConvNeXt Tiny Advanced Training (Clean GOD MODE)
# MixUp + Label Smoothing + Cosine + EMA (fixed)
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
from torch.amp import autocast, GradScaler
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
# EMA (FIXED VERSION)
# =====================================================
class EMA:

    def __init__(self, model, decay=EMA_DECAY):

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
                    alpha=1 - self.decay,
                )
            else:
                self.shadow[k] = current[k].clone()

    def apply_shadow(self, model):

        self.backup = deepcopy(model.state_dict())

        model.load_state_dict(self.shadow)

    def restore(self, model):

        model.load_state_dict(self.backup)

# =====================================================
# Paths / Classes
# =====================================================

train_loader, val_loader, test_loader = create_dataloaders()

print(f"Train : {len(train_loader.dataset)}")
print(f"Val   : {len(val_loader.dataset)}")
print(f"Test  : {len(test_loader.dataset)}")

NUM_CLASSES = len(CLASSES)


# =====================================================
# MixUp
# =====================================================
def mixup(images, labels, alpha=MIXUP_ALPHA):

    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)

    index = torch.randperm(images.size(0)).to(images.device)

    mixed = lam * images + (1 - lam) * images[index]

    return mixed, labels, labels[index], lam

# =====================================================
# Model
# =====================================================
weights = models.ConvNeXt_Tiny_Weights.DEFAULT

model = models.convnext_tiny(weights=weights)

for p in model.parameters():
    p.requires_grad = False

for p in model.features[7].parameters():
    p.requires_grad = True

in_features = model.classifier[2].in_features

model.classifier[2] = nn.Sequential(

    nn.Dropout(0.5),

    nn.Linear(
        in_features,
        NUM_CLASSES,
    )
)

model = model.to(device)

ema = EMA(model)

# =====================================================
# Loss / Optim / Scheduler
# =====================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# =====================================================
# Training
# =====================================================
best_f1 = 0
patience = 0
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
MODEL_PATH = MODEL_DIR / "convnext_advanced.pt"
scaler = GradScaler(device="cuda")

for epoch in range(30):

    model.train()
    preds, targets = [], []

    for x, y in tqdm(train_loader):

        x, y = x.to(device), y.to(device)
        x, y1, y2, lam = mixup(x, y)

        optimizer.zero_grad()

        with autocast(device_type="cuda"):

            out = model(x)

            loss = (
                lam * criterion(out, y1)
                + (1 - lam) * criterion(out, y2)
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(optimizer)
        scaler.update()

    train_f1 = f1_score(targets, preds, average="macro")

    # ===== Validation =====
    ema.apply_shadow(model)
    model.eval()

    vp, vt = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            vp.extend(out.argmax(1).cpu().numpy())
            vt.extend(y.cpu().numpy())

    val_f1 = f1_score(vt, vp, average="macro")
    ema.restore(model)

    scheduler.step()

    print(f"Epoch {epoch} | Train {train_f1:.4f} | Val {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )
        patience = 0
    else:
        patience += 1
        if patience > EARLY_STOPPING_PATIENCE:
            break

# =====================================================
# Test
# =====================================================
model.load_state_dict(torch.load( MODEL_PATH, map_location=device, weights_only=True))
model.eval()

preds, targets = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        with torch.no_grad():
            preds.extend(out.argmax(1).detach().cpu().numpy())
        targets.extend(y.cpu().numpy())

print("Accuracy:", accuracy_score(targets, preds))
print("F1:", f1_score(targets, preds, average="macro"))
print(classification_report(targets, preds, target_names=CLASS_NAMES))

