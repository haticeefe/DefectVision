from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def get_class_name(image_path):
    filename = image_path.stem.lower()
    normalized = filename.replace(" ", "_")

    for class_name in CLASS_NAMES:
        normalized_class = class_name.replace("-", "_")

        if normalized.startswith(normalized_class):
            return class_name

    return image_path.parent.name.lower()


def main():

    images_by_class = defaultdict(list)

    for path in DATA_DIR.rglob("*"):

        if path.suffix.lower() in IMAGE_EXTENSIONS:

            class_name = get_class_name(path)

            if class_name in CLASS_NAMES:
                images_by_class[class_name].append(path)

    fig, axes = plt.subplots(
        len(CLASS_NAMES),
        3,
        figsize=(9, 15)
    )

    for row, class_name in enumerate(CLASS_NAMES):

        samples = images_by_class[class_name][:3]

        for col, image_path in enumerate(samples):

            image = Image.open(image_path)

            axes[row, col].imshow(image)
            axes[row, col].axis("off")

            if col == 0:
                axes[row, col].set_title(class_name)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()