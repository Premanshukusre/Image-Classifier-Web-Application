import os
import hashlib

# Location of your dataset
DATASET_PATH = r"F:\DL\AppleDiseaseProject"

train_path = os.path.join(DATASET_PATH, "train")
val_path = os.path.join(DATASET_PATH, "val")


def calculate_hash(file_path):
    """
    Create a fingerprint (hash) for an image file.
    Identical files will have identical hashes.
    """

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_image_files(folder_path):

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    image_files = []

    for root, folders, files in os.walk(folder_path):

        for filename in files:

            if filename.lower().endswith(image_extensions):

                full_path = os.path.join(root, filename)

                image_files.append(full_path)

    return image_files


print("=" * 70)
print("DUPLICATE CLEANING")
print("=" * 70)


# ---------------------------------------------------------
# STEP 1: Get training images
# ---------------------------------------------------------

print("\nScanning training images...")

train_images = get_image_files(train_path)

print(f"Training images found: {len(train_images)}")


# ---------------------------------------------------------
# STEP 2: Create hash list for training images
# ---------------------------------------------------------

print("\nCreating training image fingerprints...")

train_hashes = {}

for image_path in train_images:

    file_hash = calculate_hash(image_path)

    train_hashes[file_hash] = image_path


# ---------------------------------------------------------
# STEP 3: Get validation images
# ---------------------------------------------------------

print("\nScanning validation images...")

val_images = get_image_files(val_path)

print(f"Validation images found: {len(val_images)}")


# ---------------------------------------------------------
# STEP 4: Find duplicates
# ---------------------------------------------------------

duplicates = []

print("\nSearching for duplicates...")

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
# STEP 5: Show duplicates
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATES FOUND")
print("=" * 70)

print(f"\nNumber of duplicates: {len(duplicates)}")


if len(duplicates) == 0:

    print("\nNo duplicates found.")
    print("Nothing needs to be deleted.")

else:

    for number, (train_image, val_image) in enumerate(duplicates, start=1):

        print(f"\nDuplicate #{number}")

        print("TRAIN:")
        print(train_image)

        print("VAL:")
        print(val_image)


    # -----------------------------------------------------
    # STEP 6: Ask for confirmation
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    answer = input(
        "\nDo you want to DELETE ONLY the duplicate copies from VAL? "
        "(yes/no): "
    )

    if answer.lower() == "yes":

        print("\nDeleting duplicate validation images...")

        deleted = 0

        for train_image, val_image in duplicates:

            try:

                os.remove(val_image)

                print(f"\nDeleted:")
                print(val_image)

                deleted += 1

            except Exception as error:

                print(f"\nCould not delete:")
                print(val_image)

                print(f"Error: {error}")


        print("\n" + "=" * 70)
        print("CLEANING COMPLETE")
        print("=" * 70)

        print(f"\nImages deleted from VAL: {deleted}")

    else:

        print("\nNo files were deleted.")
        print("Dataset remains unchanged.")


print("\nDone.")