from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image, ImageFile
from torchvision import transforms

from config import (
    PREPROCESSED_DATA_DIR,
    AUGMENTED_DIR,
    CLASSES,
    TRAIN_PHASE,
    IMAGE_SIZE,
    NUM_AUGMENTATIONS,
    ROTATION_DEGREES,
    BRIGHTNESS,
    CONTRAST,
    RANDOM_RESIZED_CROP_SCALE,
    OUTPUT_IMAGE_FORMAT,
    VALID_IMAGE_EXTENSIONS,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==========================================================
# Data augmentation pipeline
# ==========================================================

def get_xray_augmentation() -> transforms.Compose:
    """
    Create the data augmentation pipeline for chest X-ray images.
    """

    return transforms.Compose([
        transforms.RandomRotation(
            degrees=ROTATION_DEGREES,
        ),
        transforms.ColorJitter(
            brightness=BRIGHTNESS,
            contrast=CONTRAST,
        ),
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=RANDOM_RESIZED_CROP_SCALE,
        ),
    ])


AUGMENTATION = get_xray_augmentation()


# ==========================================================
# Utility functions
# ==========================================================

def ensure_directory(path: Path) -> None:
    """
    Create the output directory if it does not already exist.
    """

    path.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Image augmentation
# ==========================================================

def process_images() -> None:
    """
    Generate augmented images for all training images.
    """

    for class_name in CLASSES:

        input_folder = PREPROCESSED_DATA_DIR / TRAIN_PHASE / class_name
        output_folder = AUGMENTED_DIR / TRAIN_PHASE / class_name

        ensure_directory(output_folder)

        if not input_folder.exists():
            print(f"[WARNING] Folder not found: {input_folder}")
            continue

        image_files = sorted(
            file
            for file in input_folder.iterdir()
            if file.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )

        if not image_files:
            print(f"[WARNING] No images found for class '{class_name}'.")
            continue

        print(
            f"Processing {class_name} "
            f"({len(image_files)} original images)"
        )

        for image_path in image_files:

            try:

                image = Image.open(image_path).convert("RGB")

                for aug_idx in range(NUM_AUGMENTATIONS):

                    augmented_image = AUGMENTATION(image)

                    output_path = (
                        output_folder /
                        f"{image_path.stem}_aug{aug_idx + 1}.png"
                    )

                    augmented_image.save(
                        output_path,
                        format=OUTPUT_IMAGE_FORMAT,
                    )

            except Exception as error:

                print(f"[ERROR] {image_path.name}: {error}")

    print(
        "\nAugmentation completed successfully.\n"
        f"Generated {NUM_AUGMENTATIONS} augmented images "
        "for every original image."
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    process_images()