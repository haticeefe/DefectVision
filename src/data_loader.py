from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BATCH_SIZE = 32
IMAGE_SIZE = 224


def get_dataloaders():

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        PROCESSED_DIR / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        PROCESSED_DIR / "val",
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        PROCESSED_DIR / "test",
        transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )


if __name__ == "__main__":

    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print("\n===== DATALOADER CHECK =====")

    print(f"\nSınıflar: {classes}")

    images, labels = next(iter(train_loader))

    print(f"\nBatch görüntü boyutu: {images.shape}")
    print(f"Batch label boyutu: {labels.shape}")

    print(f"\nTrain batch sayısı: {len(train_loader)}")
    print(f"Validation batch sayısı: {len(val_loader)}")
    print(f"Test batch sayısı: {len(test_loader)}")

    print("\n✅ DataLoader başarıyla oluşturuldu.")