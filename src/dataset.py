from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import *


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
])


class ChestDataset(Dataset):

    def __init__(self, roots, transform):

        self.samples = []
        self.transform = transform

        self.label_map = {c: i for i, c in enumerate(CLASSES)}

        for root in roots:
            for cls in CLASSES:

                class_dir = root / cls
                if not class_dir.exists():
                    continue

                for image_path in sorted(class_dir.iterdir()):  

                    if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                        self.samples.append(
                            (image_path, self.label_map[cls])
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return image, label

def create_dataloaders():

    train_dir = PREPROCESSED_DATA_DIR / TRAIN_PHASE
    val_dir = PREPROCESSED_DATA_DIR / VALID_PHASE
    test_dir = PREPROCESSED_DATA_DIR / TEST_PHASE
    aug_dir = AUGMENTED_DIR / TRAIN_PHASE

    train_loader = DataLoader(
        ChestDataset([train_dir, aug_dir], train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        ChestDataset([val_dir], eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        ChestDataset([test_dir], eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader