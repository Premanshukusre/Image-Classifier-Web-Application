from pathlib import Path
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]


# ============================================================
# DATASET PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    print("=" * 70)
    print("LOADING APPLE DISEASE DATASET")
    print("=" * 70)

    print(f"\nDataset directory: {DATASET_DIR}")
    print(f"Train directory  : {TRAIN_DIR}")
    print(f"Validation dir   : {VAL_DIR}")
    print(f"Test directory   : {TEST_DIR}")

    # Check directories
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}"
        )

    if not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {VAL_DIR}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Normalize pixel values
    # --------------------------------------------------------

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(
        lambda images, labels: (
            normalization_layer(images),
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    val_ds = val_ds.map(
        lambda images, labels: (
            normalization_layer(images),
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    test_ds = test_ds.map(
        lambda images, labels: (
            normalization_layer(images),
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # --------------------------------------------------------
    # Improve input pipeline performance
    # --------------------------------------------------------

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 70)

    print("\nClasses:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"  {index}: {class_name}")

    print(f"\nImage size : {IMAGE_SIZE}")
    print(f"Batch size : {BATCH_SIZE}")

    return train_ds, val_ds, test_ds


# ============================================================
# TEST DATA LOADER
# ============================================================

if __name__ == "__main__":

    train_ds, val_ds, test_ds = load_datasets()

    print("\n" + "=" * 70)
    print("DATASET TEST")
    print("=" * 70)

    for images, labels in train_ds.take(1):

        print(f"\nImage batch shape : {images.shape}")
        print(f"Label batch shape : {labels.shape}")

        print(
            f"Pixel value range : "
            f"{images.numpy().min():.4f}"
            f" - "
            f"{images.numpy().max():.4f}"
        )

    print("\nDataset loader test completed successfully.")