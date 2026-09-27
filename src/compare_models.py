from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def main():

    models = [
        "Custom CNN",
        "ResNet18"
    ]

    accuracies = [
        95.93,
        97.04
    ]

    f1_scores = [
        96.00,
        97.04
    ]

    x = range(len(models))

    plt.figure(figsize=(8, 5))

    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        accuracies,
        width,
        label="Test Accuracy"
    )

    plt.bar(
        [i + width / 2 for i in x],
        f1_scores,
        width,
        label="Weighted F1"
    )

    plt.xticks(
        list(x),
        models
    )

    plt.ylim(90, 100)

    plt.ylabel("Score (%)")

    plt.title(
        "DefectVision - Model Performance Comparison"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\n✅ Model karşılaştırma grafiği kaydedildi:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()