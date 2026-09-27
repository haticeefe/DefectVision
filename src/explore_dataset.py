from pathlib import Path
from collections import Counter
from PIL import Image


# Projenin ana klasörü
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset klasörü
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Desteklenen görsel uzantıları
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(data_dir):
    """data/raw altında bulunan tüm görselleri bulur."""
    images = [
        path
        for path in data_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return images


def get_class_name(image_path):
    """
    NEU datasetinde dosya isimleri genellikle:
    crazing_1.jpg
    inclusion_1.jpg
    patches_1.jpg
    ...

    şeklindedir.
    """
    filename = image_path.stem.lower()

    class_names = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]

    # Farklı dosya isimlendirmelerine karşı biraz toleranslı olalım
    normalized = filename.replace(" ", "_")

    for class_name in class_names:
        normalized_class = class_name.replace("-", "_")

        if normalized.startswith(normalized_class):
            return class_name

    # Eğer klasörler sınıf isimlerine göre ayrılmışsa:
    return image_path.parent.name.lower()


def main():

    images = find_images(DATA_DIR)

    print("\n===== DEFECTVISION DATASET ANALYSIS =====\n")

    print(f"Dataset yolu: {DATA_DIR}")
    print(f"Toplam bulunan görüntü: {len(images)}")

    if len(images) == 0:
        print("\nHiç görüntü bulunamadı.")
        print("Datasetin data/raw klasörünün içinde olduğundan emin ol.")
        return

    # Sınıfları say
    classes = [get_class_name(image) for image in images]
    class_counts = Counter(classes)

    print("\n--- Sınıf Dağılımı ---")

    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count}")

    # İlk görseli incele
    sample_image_path = images[0]

    with Image.open(sample_image_path) as image:

        print("\n--- Örnek Görüntü ---")
        print(f"Dosya: {sample_image_path.name}")
        print(f"Boyut: {image.size}")
        print(f"Renk modu: {image.mode}")

    print("\n Dataset başarıyla okundu.")


if __name__ == "__main__":
    main()