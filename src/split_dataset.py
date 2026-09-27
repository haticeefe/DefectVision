from pathlib import Path
from collections import Counter
import shutil

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

RANDOM_SEED = 42


def get_class_name(image_path):
    filename = image_path.stem.lower()
    normalized = filename.replace(" ", "_")

    for class_name in CLASS_NAMES:
        normalized_class = class_name.replace("-", "_")

        if normalized.startswith(normalized_class):
            return class_name

    return image_path.parent.name.lower()


def find_images():
    images = []

    for path in RAW_DIR.rglob("*"):

        if path.suffix.lower() in IMAGE_EXTENSIONS:

            class_name = get_class_name(path)

            if class_name in CLASS_NAMES:
                images.append((path, class_name))

    return images


def create_directories():

    for split in ["train", "val", "test"]:

        for class_name in CLASS_NAMES:

            directory = PROCESSED_DIR / split / class_name
            directory.mkdir(parents=True, exist_ok=True)


def copy_images(image_list, split_name):

    for image_path, class_name in image_list:

        destination = (
            PROCESSED_DIR
            / split_name
            / class_name
            / image_path.name
        )

        shutil.copy2(image_path, destination)


def main():

    images = find_images()

    print(f"\nToplam görüntü: {len(images)}")

    paths = [item[0] for item in images]
    labels = [item[1] for item in images]

    # %70 Train - %30 geçici
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    # Kalan %30'u iki eşit parçaya böl:
    # %15 Validation
    # %15 Test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_labels,
    )

    train_data = list(zip(train_paths, train_labels))
    val_data = list(zip(val_paths, val_labels))
    test_data = list(zip(test_paths, test_labels))

    create_directories()

    copy_images(train_data, "train")
    copy_images(val_data, "val")
    copy_images(test_data, "test")

    print("\n===== DATASET SPLIT =====")

    print(f"\nTrain: {len(train_data)}")
    print(Counter(train_labels))

    print(f"\nValidation: {len(val_data)}")
    print(Counter(val_labels))

    print(f"\nTest: {len(test_data)}")
    print(Counter(test_labels))

    print("\n✅ Dataset başarıyla ayrıldı.")


if __name__ == "__main__":
    main()