import os
import random
import shutil

# ============================================================
# DATASET LOCATION
# ============================================================

DATASET_PATH = r"F:\DL\AppleDiseaseProject"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")


# ============================================================
# SETTINGS
# ============================================================

TEST_RATIO = 0.10

# Fixed seed makes the split reproducible
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# IMAGE FILE TYPES
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)


# ============================================================
# CHECK IF TEST FOLDER ALREADY EXISTS
# ============================================================

if os.path.exists(TEST_PATH):

    print("ERROR:")
    print("The 'test' folder already exists.")
    print("The script will not continue to avoid overwriting anything.")

    exit()


# ============================================================
# CREATE TEST FOLDER
# ============================================================

os.makedirs(TEST_PATH)

print("=" * 70)
print("CREATING TEST DATASET")
print("=" * 70)

print(f"\nTest ratio: {TEST_RATIO * 100:.0f}%")
print(f"Random seed: {RANDOM_SEED}")


# ============================================================
# GET CLASS FOLDERS
# ============================================================

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]


total_moved = 0


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_name in class_names:

    print("\n" + "-" * 70)
    print(f"Processing: {class_name}")
    print("-" * 70)

    train_class_path = os.path.join(
        TRAIN_PATH,
        class_name
    )

    test_class_path = os.path.join(
        TEST_PATH,
        class_name
    )

    # Create corresponding test class folder
    os.makedirs(test_class_path)

    # --------------------------------------------------------
    # Get images
    # --------------------------------------------------------

    images = []

    for filename in os.listdir(train_class_path):

        if filename.lower().endswith(IMAGE_EXTENSIONS):

            images.append(filename)

    # --------------------------------------------------------
    # Calculate number of test images
    # --------------------------------------------------------

    number_of_test_images = round(
        len(images) * TEST_RATIO
    )

    # --------------------------------------------------------
    # Randomly select test images
    # --------------------------------------------------------

    selected_images = random.sample(
        images,
        number_of_test_images
    )

    # --------------------------------------------------------
    # Move selected images
    # --------------------------------------------------------

    for filename in selected_images:

        source = os.path.join(
            train_class_path,
            filename
        )

        destination = os.path.join(
            test_class_path,
            filename
        )

        shutil.move(source, destination)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    remaining_train = len(images) - number_of_test_images

    print(f"Original train images : {len(images)}")
    print(f"Moved to test         : {number_of_test_images}")
    print(f"Remaining train       : {remaining_train}")

    total_moved += number_of_test_images


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("TEST DATASET CREATED")
print("=" * 70)

print(f"\nTotal images moved to test: {total_moved}")

print("\nYour dataset now contains:")

print("\nTRAIN")
print("Images remaining after test split")

print("\nVAL")
print("630 images")

print("\nTEST")
print(f"{total_moved} images")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)