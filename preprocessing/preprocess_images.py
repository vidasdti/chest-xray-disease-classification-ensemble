import cv2
import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    RAW_DATASET_DIR,
    PREPROCESSED_DATA_DIR,
    IMAGE_SHAPE,
    CLASS_NAMES,
    CLASS_MAPPING,
    DATASET_PHASES,
    USE_CLAHE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    VALID_IMAGE_EXTENSIONS,
    OUTPUT_IMAGE_FORMAT,
)

# ==========================================================
# Image preprocessing
# ==========================================================

def pad_to_center(
    image: Image.Image,
    size: tuple[int, int] = IMAGE_SHAPE,
) -> Image.Image:
    """
    Resize an image while preserving its aspect ratio and
    pad it to the target size.
    """

    image.thumbnail(size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", size, (0, 0, 0))

    left = (size[0] - image.width) // 2
    top = (size[1] - image.height) // 2

    canvas.paste(image, (left, top))

    return canvas


def apply_clahe(image: Image.Image) -> Image.Image:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance the local contrast of chest X-ray images.
    """

    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )

    enhanced = clahe.apply(gray)

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2RGB,
    )

    return Image.fromarray(enhanced)


# ==========================================================
# Dataset processing
# ==========================================================

def process_class(
    phase: str,
    class_name: str,
) -> None:
    """
    Process all images belonging to a specific class
    within a dataset split.
    """

    input_dir = RAW_DATASET_DIR / phase / class_name
    short_class_name = CLASS_MAPPING[class_name]
    output_dir = PREPROCESSED_DATA_DIR / phase / short_class_name

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[WARNING] Directory not found: {input_dir}")
        return

    image_files = sorted(
        file
        for file in input_dir.iterdir()
        if file.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"[WARNING] No images found in {input_dir}")
        return

    print(
        f"Processing {phase}/{class_name} "
        f"({len(image_files)} images)..."
    )

    for index, image_path in enumerate(image_files):

        try:

            image = Image.open(image_path).convert("RGB")

            image = pad_to_center(image)

            if USE_CLAHE:
                image = apply_clahe(image)

            output_path = output_dir / f"image{index}.png"

            image.save(
                output_path,
                format=OUTPUT_IMAGE_FORMAT,
            )

        except Exception as error:

            print(
                f"[ERROR] "
                f"{image_path.name}: {error}"
            )

    print(f"Completed: {phase}/{class_name}")


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    """
    Run preprocessing for all dataset splits.
    """

    for phase in DATASET_PHASES:

        print(f"\n{'=' * 20}")
        print(f"{phase.upper()} SET")
        print(f"{'=' * 20}")

        for class_name in CLASS_NAMES:
            process_class(
                phase=phase,
                class_name=class_name,
            )

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()