import os
import hashlib
import shutil

# ============================================================
# DATASET PATHS
# ============================================================

DATASET_PATH = r"F:\DL\AppleDiseaseProject"

TRAIN_HEALTHY_PATH = os.path.join(
    DATASET_PATH,
    "train",
    "Apple___healthy"
)

TEST_HEALTHY_PATH = os.path.join(
    DATASET_PATH,
    "test",
    "Apple___healthy"
)


# ============================================================
# THE DUPLICATE TEST IMAGE
# ============================================================

duplicate_test_image = os.path.join(
    TEST_HEALTHY_PATH,
    "c21cf428-bfc3-4710-b5d2-69d1c0e94748___RS_HL 6268.JPG"
)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)


# ============================================================
# HASH FUNCTION
# ============================================================

def calculate_hash(file_path):

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


# ============================================================
# GET HASHES OF ALL TEST IMAGES
# ============================================================

print("=" * 70)
print("REPLACING TRAIN ↔ TEST DUPLICATE")
print("=" * 70)


print("\nChecking current test images...")

test_hashes = set()

for filename in os.listdir(TEST_HEALTHY_PATH):

    if filename.lower().endswith(IMAGE_EXTENSIONS):

        file_path = os.path.join(
            TEST_HEALTHY_PATH,
            filename
        )

        test_hashes.add(
            calculate_hash(file_path)
        )


# ============================================================
# CHECK DUPLICATE EXISTS
# ============================================================

if not os.path.exists(duplicate_test_image):

    print("\nERROR:")
    print("The duplicate test image was not found.")

    print("\nExpected:")
    print(duplicate_test_image)

    exit()


# ============================================================
# FIND A SAFE REPLACEMENT
# ============================================================

print("\nSearching for a replacement image...")


replacement_image = None

for filename in os.listdir(TRAIN_HEALTHY_PATH):

    if not filename.lower().endswith(IMAGE_EXTENSIONS):

        continue

    train_image = os.path.join(
        TRAIN_HEALTHY_PATH,
        filename
    )

    train_hash = calculate_hash(train_image)

    # Make sure this image is NOT already in TEST
    if train_hash not in test_hashes:

        replacement_image = train_image

        break


# ============================================================
# CHECK REPLACEMENT
# ============================================================

if replacement_image is None:

    print("\nERROR:")
    print("Could not find a safe replacement image.")

    exit()


print("\nReplacement image found:")

print(replacement_image)


# ============================================================
# DELETE DUPLICATE FROM TEST
# ============================================================

print("\nRemoving duplicate from TEST...")

os.remove(duplicate_test_image)

print("Duplicate removed successfully.")


# ============================================================
# MOVE REPLACEMENT INTO TEST
# ============================================================

replacement_filename = os.path.basename(
    replacement_image
)

destination = os.path.join(
    TEST_HEALTHY_PATH,
    replacement_filename
)


print("\nMoving replacement into TEST...")

shutil.move(
    replacement_image,
    destination
)

print("Replacement moved successfully.")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("REPLACEMENT COMPLETE")
print("=" * 70)

print("\nRemoved from TEST:")
print(duplicate_test_image)

print("\nAdded to TEST:")
print(destination)

print("\nTraining healthy image count decreased by 1.")
print("Test healthy image count stayed the same.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)