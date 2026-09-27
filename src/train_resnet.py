from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders
from resnet_model import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet18_best.pth"

EPOCHS = 10
LEARNING_RATE = 0.001


def train_one_epoch(model, dataloader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predictions = torch.max(outputs, 1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, 1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def main():

    print("\n===== RESNET18 TRANSFER LEARNING =====\n")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Kullanılan cihaz: {device}\n")

    train_loader, val_loader, _, classes = get_dataloaders()

    print(f"Sınıflar: {classes}")
    print(f"Sınıf sayısı: {len(classes)}\n")

    model = create_resnet18(
        num_classes=len(classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Sadece eğitilebilir parametreleri optimizer'a veriyoruz
    optimizer = optim.Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Accuracy:   {val_accuracy * 100:.2f}%"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                },
                MODEL_PATH
            )

            print("✅ Yeni en iyi ResNet18 modeli kaydedildi.")

        print("-" * 50)

    print("\n🎉 ResNet18 eğitimi tamamlandı.")

    print(
        f"En iyi model:\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()