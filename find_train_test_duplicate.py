import os
import hashlib

DATASET_PATH = r"F:\DL\AppleDiseaseProject"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def calculate_hash(file_path):

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_image_files(folder_path):

    image_files = []

    for root, folders, files in os.walk(folder_path):

        for filename in files:

            if filename.lower().endswith(IMAGE_EXTENSIONS):

                full_path = os.path.join(root, filename)

                image_files.append(full_path)

    return image_files


print("=" * 70)
print("FINDING TRAIN ↔ TEST DUPLICATE")
print("=" * 70)


print("\nScanning TRAIN...")

train_images = get_image_files(TRAIN_PATH)

train_hashes = {}

for image_path in train_images:

    file_hash = calculate_hash(image_path)

    train_hashes[file_hash] = image_path


print(f"Training images checked: {len(train_images)}")


print("\nScanning TEST...")

test_images = get_image_files(TEST_PATH)

duplicates = []

for image_path in test_images:

    file_hash = calculate_hash(image_path)

    if file_hash in train_hashes:

        duplicates.append(
            (
                train_hashes[file_hash],
                image_path
            )
        )


print(f"Test images checked: {len(test_images)}")


print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"\nDuplicates found: {len(duplicates)}")


for number, (train_image, test_image) in enumerate(
    duplicates,
    start=1
):

    print(f"\nDuplicate #{number}")

    print("\nTRAIN:")
    print(train_image)

    print("\nTEST:")
    print(test_image)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)