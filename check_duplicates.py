import os
import hashlib

# Location of your dataset
DATASET_PATH = r"F:\DL\AppleDiseaseProject"

# Dataset folders
train_path = os.path.join(DATASET_PATH, "train")
val_path = os.path.join(DATASET_PATH, "val")


def calculate_hash(file_path):
    """
    Calculate an MD5 hash for a file.
    Files with the same content will have the same hash.
    """

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_image_files(folder_path):
    """
    Get all image files inside a folder and its subfolders.
    """

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    image_files = []

    for root, folders, files in os.walk(folder_path):

        for filename in files:

            if filename.lower().endswith(image_extensions):

                full_path = os.path.join(root, filename)

                image_files.append(full_path)

    return image_files


print("=" * 70)
print("DUPLICATE IMAGE CHECK")
print("=" * 70)


# ---------------------------------------------------------
# STEP 1: Get all training images
# ---------------------------------------------------------

print("\nScanning training images...")

train_images = get_image_files(train_path)

print(f"Training images found: {len(train_images)}")


# ---------------------------------------------------------
# STEP 2: Calculate hashes for training images
# ---------------------------------------------------------

print("\nCalculating hashes for training images...")

train_hashes = {}

for image_path in train_images:

    file_hash = calculate_hash(image_path)

    train_hashes[file_hash] = image_path


print("Training image hashing complete.")


# ---------------------------------------------------------
# STEP 3: Get all validation images
# ---------------------------------------------------------

print("\nScanning validation images...")

val_images = get_image_files(val_path)

print(f"Validation images found: {len(val_images)}")


# ---------------------------------------------------------
# STEP 4: Compare validation images with training images
# ---------------------------------------------------------

print("\nChecking for duplicates between TRAIN and VAL...")

duplicates = []

for image_path in val_images:

    file_hash = calculate_hash(image_path)

    if file_hash in train_hashes:

        duplicates.append(
            (
                train_hashes[file_hash],
                image_path
            )
        )


# ---------------------------------------------------------
# STEP 5: Display results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"\nTraining images     : {len(train_images)}")
print(f"Validation images   : {len(val_images)}")
print(f"Duplicates found    : {len(duplicates)}")


if len(duplicates) == 0:

    print("\n✅ GREAT!")
    print("No exact duplicate images were found between TRAIN and VAL.")

else:

    print("\n⚠️ DUPLICATES FOUND!")

    print("\nDuplicate pairs:")

    for train_image, val_image in duplicates:

        print("\nTRAIN:")
        print(train_image)

        print("VAL:")
        print(val_image)


print("\n" + "=" * 70)
print("DUPLICATE CHECK COMPLETE")
print("=" * 70)