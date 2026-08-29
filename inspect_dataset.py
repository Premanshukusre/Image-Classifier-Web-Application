import os
from PIL import Image
from collections import Counter

# Location of your dataset
DATASET_PATH = r"F:\DL\AppleDiseaseProject"

# The 4 Apple classes
classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy"
]

# Dataset folders
folders = ["train", "val"]


# Check each dataset folder
for folder in folders:

    print("\n" + "=" * 70)
    print(f"{folder.upper()} DATASET")
    print("=" * 70)

    for class_name in classes:

        folder_path = os.path.join(
            DATASET_PATH,
            folder,
            class_name
        )

        # Store image sizes and formats
        image_sizes = Counter()
        image_formats = Counter()

        total = 0
        valid = 0
        corrupted = 0

        if not os.path.exists(folder_path):
            print(f"\nERROR: Folder not found:")
            print(folder_path)
            continue

        # Check every file
        for filename in os.listdir(folder_path):

            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):

                total += 1

                try:
                    # Open image
                    with Image.open(file_path) as img:

                        # Check that image is valid
                        img.verify()

                    # Open again to get image information
                    with Image.open(file_path) as img:

                        width, height = img.size

                        image_sizes[(width, height)] += 1

                        image_formats[img.format] += 1

                    valid += 1

                except Exception:
                    corrupted += 1

        # Display results
        print(f"\nClass: {class_name}")
        print(f"Total files : {total}")
        print(f"Valid images: {valid}")
        print(f"Corrupted   : {corrupted}")

        print("\nImage dimensions:")

        for size, count in image_sizes.items():
            print(f"  {size[0]} x {size[1]} : {count} images")

        print("\nImage formats:")

        for image_format, count in image_formats.items():
            print(f"  {image_format} : {count} images")


print("\n" + "=" * 70)
print("IMAGE INSPECTION COMPLETE")
print("=" * 70)