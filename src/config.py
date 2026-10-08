from pathlib import Path

# ==========================================================
# Project directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATASET_DIR = PROJECT_ROOT / "Lung Disease Dataset"
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "data"
AUGMENTED_DIR = PROJECT_ROOT / "AUG"
FEATURE_DIR = PROJECT_ROOT / "features"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

# ==========================================================
# Dataset classes
# ==========================================================

CLASS_NAMES = [
    "BacterialPneumonia",
    "CoronaVirusDisease",
    "Normal",
    "Tuberculosis",
    "ViralPneumonia",
]

# Short class names (used after preprocessing)

CLASSES = ["BP","VP","CV","NO","TB"]

CLASS_MAPPING = dict(zip(CLASS_NAMES, CLASSES))
# ==========================================================
# Dataset splits
# ==========================================================

TRAIN_PHASE = "train"

VALID_PHASE = "val"

TEST_PHASE = "test"

DATASET_PHASES = [
    TRAIN_PHASE,
    VALID_PHASE,
    TEST_PHASE,
]

# ==========================================================
# Image preprocessing
# ==========================================================

# Target image size for CNN input

IMAGE_SIZE = 224

# Size used for padding operations

IMAGE_SHAPE = (224, 224)

# Apply CLAHE during preprocessing

USE_CLAHE = True

CLAHE_CLIP_LIMIT = 2.0

CLAHE_TILE_GRID_SIZE = (8, 8)

# Supported image extensions

VALID_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)

# Output image format

OUTPUT_IMAGE_FORMAT = "PNG"

# ==========================================================
# Data augmentation
# ==========================================================

NUM_AUGMENTATIONS = 5

ROTATION_DEGREES = 5

BRIGHTNESS = 0.08

CONTRAST = 0.08

RANDOM_RESIZED_CROP_SCALE = (0.92, 1.0)

# ==========================================================
# Reproducibility
# ==========================================================

RANDOM_SEED = 42

# ==========================================================
# Training
# ==========================================================

BATCH_SIZE = 16
NUM_EPOCHS = 40

LEARNING_RATE = 1e-3

EARLY_STOPPING_PATIENCE = 7

NUM_WORKERS = 0

PIN_MEMORY = True

MIXUP_ALPHA = 0.4

EMA_DECAY = 0.999

GRAD_CLIP = 2.0

# ==========================================================
# Image normalization
# ==========================================================

NORMALIZATION_MEAN = (
    0.485,
    0.456,
    0.406,
)

NORMALIZATION_STD = (
    0.229,
    0.224,
    0.225,
)

