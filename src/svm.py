import random
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torchvision import models
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
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
train_loader, _, test_loader = create_dataloaders()

print(f"Train samples : {len(train_loader.dataset)}")
print(f"Test samples  : {len(test_loader.dataset)}")

# =====================================================
# CNN Feature Extractor (ResNet18)
# =====================================================
cnn = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)
cnn.fc = torch.nn.Identity()   # remove classifier
cnn = cnn.to(device)
cnn.eval()

def extract_features(loader, model, desc):

    features = []
    labels = []

    model.eval()

    with torch.no_grad():

        for images, target in tqdm(loader, desc=desc, ncols=100):

            images = images.to(device)

            feats = model(images)

            features.append(feats.cpu().numpy())

            labels.append(target.numpy())

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)

    return X, y


print("Extracting train features...")
X_train, y_train = extract_features(
    train_loader,
    cnn,
    "Train feature extraction"
)

print("Extracting test features...")
X_test, y_test = extract_features(
    test_loader,
    cnn,
    "Test feature extraction"
)


# =====================================================
# Train SVM
# =====================================================
print(" Training SVM (RBF kernel)...")

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    decision_function_shape="ovr"
)

svm.fit(X_train, y_train)

# =====================================================
# Evaluation on TEST set
# =====================================================
y_pred = svm.predict(X_test)

overall_acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n========================================")
print(f" Overall Test Accuracy : {overall_acc:.4f}")
print(f" Overall Macro-F1      : {macro_f1:.4f}")
print("========================================\n")

# =====================================================
# Per-class Precision / Recall / F1 / Support
# =====================================================
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=range(len(CLASSES))
)

print(" Per-class Metrics")
print("-" * 75)
print(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 75)

for i, cls in enumerate(CLASSES):
    print(f"{cls:<8} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")

print("-" * 75)

# =====================================================
# Per-class Accuracy (One-vs-Rest)
# =====================================================
print("\n Per-class Accuracy (One-vs-Rest)")
print("-" * 50)

for i, cls in enumerate(CLASSES):
    y_true_bin = (y_test == i).astype(int)
    y_pred_bin = (y_pred == i).astype(int)
    acc_i = accuracy_score(y_true_bin, y_pred_bin)
    print(f"{cls:<8}: {acc_i:.4f}")

# =====================================================
# Confusion Matrix
# =====================================================
print("\n Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# Classification Report (sklearn standard)
# =====================================================
report = classification_report(
    y_test,
    y_pred,
    target_names=CLASSES,
    digits=4
)

print("\n Classification Report")
print(report)