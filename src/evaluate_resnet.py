from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from data_loader import get_dataloaders
from resnet_model import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_best.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def main():

    print("\n===== RESNET18 TEST EVALUATION =====\n")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Kullanılan cihaz: {device}")

    _, _, test_loader, _ = get_dataloaders()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    classes = checkpoint["classes"]

    print(f"Model epoch: {checkpoint['epoch']}")
    print(
        f"Validation Accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )
    print(f"Sınıflar: {classes}\n")

    model = create_resnet18(
        num_classes=len(classes)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []

    test_loss = 0.0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            total += labels.size(0)

            _, predictions = torch.max(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    test_loss /= total

    test_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print("===== TEST RESULTS =====")

    print(
        f"\nTest Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy * 100:.2f}%"
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=4
    )

    print("\n===== CLASSIFICATION REPORT =====\n")
    print(report)

    report_path = (
        RESULTS_DIR
        / "resnet18_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

        file.write(
            f"\nTest Accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

        file.write(
            f"\nTest Loss: {test_loss:.4f}"
        )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d"
    )

    plt.title(
        "DefectVision - ResNet18 Confusion Matrix"
    )

    plt.tight_layout()

    confusion_matrix_path = (
        RESULTS_DIR
        / "resnet18_confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\n✅ Classification report kaydedildi:\n"
        f"{report_path}"
    )

    print(
        f"\n✅ Confusion matrix kaydedildi:\n"
        f"{confusion_matrix_path}"
    )


if __name__ == "__main__":
    main()