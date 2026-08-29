import os
import hashlib

# ============================================================
# DATASET LOCATION
# ============================================================

DATASET_PATH = r"F:\DL\AppleDiseaseProject"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")


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
# CALCULATE FILE HASH
# ============================================================

def calculate_hash(file_path):

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


# ============================================================
# GET ALL IMAGE FILES
# ============================================================

def get_image_files(folder_path):

    image_files = []

    for root, folders, files in os.walk(folder_path):

        for filename in files:

            if filename.lower().endswith(IMAGE_EXTENSIONS):

                full_path = os.path.join(root, filename)

                image_files.append(full_path)

    return image_files


# ============================================================
# CREATE HASH DICTIONARY
# ============================================================

def create_hash_dictionary(image_files):

    hashes = {}

    for image_path in image_files:

        file_hash = calculate_hash(image_path)

        hashes[file_hash] = image_path

    return hashes


# ============================================================
# COUNT IMAGES
# ============================================================

def count_images(folder_path):

    image_files = get_image_files(folder_path)

    return len(image_files)


# ============================================================
# START VERIFICATION
# ============================================================

print("=" * 70)
print("FINAL DATASET VERIFICATION")
print("=" * 70)


# ============================================================
# COUNT IMAGES
# ============================================================

train_images = get_image_files(TRAIN_PATH)
val_images = get_image_files(VAL_PATH)
test_images = get_image_files(TEST_PATH)


print("\nIMAGE COUNTS")
print("-" * 70)

print(f"Training images   : {len(train_images)}")
print(f"Validation images : {len(val_images)}")
print(f"Test images       : {len(test_images)}")

total_images = (
    len(train_images)
    + len(val_images)
    + len(test_images)
)

print(f"Total images      : {total_images}")


# ============================================================
# CREATE HASHES
# ============================================================

print("\nCreating image fingerprints...")
print("This may take a little while.")


train_hashes = create_hash_dictionary(train_images)

print("TRAIN fingerprints complete.")

val_hashes = create_hash_dictionary(val_images)

print("VAL fingerprints complete.")

test_hashes = create_hash_dictionary(test_images)

print("TEST fingerprints complete.")


# ============================================================
# FIND TRAIN ↔ VAL DUPLICATES
# ============================================================

train_val_duplicates = set(train_hashes.keys()).intersection(
    set(val_hashes.keys())
)


# ============================================================
# FIND TRAIN ↔ TEST DUPLICATES
# ============================================================

train_test_duplicates = set(train_hashes.keys()).intersection(
    set(test_hashes.keys())
)


# ============================================================
# FIND VAL ↔ TEST DUPLICATES
# ============================================================

val_test_duplicates = set(val_hashes.keys()).intersection(
    set(test_hashes.keys())
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK RESULTS")
print("=" * 70)

print(
    f"\nTRAIN ↔ VAL duplicates  : "
    f"{len(train_val_duplicates)}"
)

print(
    f"TRAIN ↔ TEST duplicates : "
    f"{len(train_test_duplicates)}"
)

print(
    f"VAL ↔ TEST duplicates   : "
    f"{len(val_test_duplicates)}"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)

total_duplicates = (
    len(train_val_duplicates)
    + len(train_test_duplicates)
    + len(val_test_duplicates)
)

if total_duplicates == 0:

    print("✅ DATASET VERIFICATION PASSED")

    print("\nNo exact duplicate images were found")
    print("between TRAIN, VAL, and TEST.")

else:

    print("⚠️ DUPLICATES FOUND")

    print("\nPlease review the duplicate counts above.")


print("=" * 70)