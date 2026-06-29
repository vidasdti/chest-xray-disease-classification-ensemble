import torch
import numpy as np
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import ChestDataset, eval_transform

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from tqdm import tqdm
from sklearn.decomposition import PCA
from config import *

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

TRAIN_DIR = PREPROCESSED_DATA_DIR / TRAIN_PHASE
TEST_DIR = PREPROCESSED_DATA_DIR / TEST_PHASE
VAL_DIR = PREPROCESSED_DATA_DIR / VALID_PHASE
AUG_DIR = AUGMENTED_DIR / TRAIN_PHASE

CLASS_NAMES = CLASSES
NUM_CLASSES = len(CLASS_NAMES)

train_dataset = ChestDataset(roots=[TRAIN_DIR, AUG_DIR], transform=eval_transform)
test_dataset = ChestDataset(roots=[TEST_DIR], transform=eval_transform)
val_dataset = ChestDataset(roots=[VAL_DIR], transform=eval_transform)

print(f"Train images (with AUG): {len(train_dataset)}")
print(f"Val images:              {len(val_dataset)}")
print(f"Test images:             {len(test_dataset)}")

# =====================================================
#  Feature Extractor
# =====================================================
def extract_features(dataset, desc="extract"):

    features = []
    labels = []

    for image, label in tqdm(dataset, desc=desc, ncols=100):

        feature = image.numpy().reshape(-1)

        features.append(feature.astype(np.float32))
        labels.append(label)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )
# =====================================================
# Feature extraction
# =====================================================
print("Extracting train features...")
X_train, y_train = extract_features(train_dataset, "train")

print("Extracting validation features...")
X_val, y_val = extract_features(val_dataset, "validation")

print("Extracting test features...")
X_test, y_test = extract_features(test_dataset, "test")

# =====================================================
# PCA 
# =====================================================

print("Applying PCA...")

pca = PCA(n_components=300, random_state=RANDOM_SEED)

X_train = pca.fit_transform(X_train)
X_val   = pca.transform(X_val)
X_test  = pca.transform(X_test)

print("PCA done. New shape:", X_train.shape)

# =====================================================
# Decision Tree
# =====================================================
print(" Searching best Decision Tree (Validation)...")

candidate_depths = [5, 10, 15, 20, 30, None]

best_model = None
best_depth = None
best_val_f1 = -1

for depth in candidate_depths:

    dt = DecisionTreeClassifier(
        max_depth=depth,
        random_state=RANDOM_SEED
    )

    dt.fit(X_train, y_train)

    val_pred = dt.predict(X_val)

    val_f1 = f1_score(y_val, val_pred, average="macro")

    print(f"Depth={depth} | Val F1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model = dt
        best_depth = depth

print("\n Best depth:", best_depth)
print(" Best Val F1:", best_val_f1)

# =====================================================
# Evaluation
# =====================================================
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n========================================")
print(f" Accuracy : {acc:.4f}")
print(f" Macro-F1 : {macro_f1:.4f}")
print("========================================\n")

# =====================================================
# Per-class metrics
# =====================================================
precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    y_pred,
    labels=range(NUM_CLASSES)
)

print(" Per-class Metrics")
print("-" * 70)
print(f"{'Class':<8} {'P':>8} {'R':>8} {'F1':>8} {'Support':>10}")
print("-" * 70)

for i, cls in enumerate(CLASS_NAMES):
    print(f"{cls:<8} {precision[i]:>8.3f} {recall[i]:>8.3f} {f1[i]:>8.3f} {support[i]:>10}")

# =====================================================
# Per-class accuracy (one-vs-rest)
# =====================================================
print("\n Per-class Accuracy (OvR)")
print("-" * 40)

for i, cls in enumerate(CLASS_NAMES):
    y_true_bin = (y_test == i).astype(int)
    y_pred_bin = (y_pred == i).astype(int)
    acc_i = accuracy_score(y_true_bin, y_pred_bin)
    print(f"{cls:<8}: {acc_i:.4f}")

# =====================================================
# Confusion matrix
# =====================================================
print("\n Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# Classification report
# =====================================================
print("\n Classification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
))