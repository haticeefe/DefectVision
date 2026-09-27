from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from resnet_model import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_best.pth"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.hook = target_layer.register_forward_hook(
            self.forward_hook
        )

    def forward_hook(self, module, inputs, output):

        self.activations = output

        # Bu layer'ın gradientlerini kaydet
        output.register_hook(
            self.save_gradient
        )

    def save_gradient(self, gradient):

        self.gradients = gradient

    def generate(self, input_tensor, class_index=None):

        self.model.zero_grad()

        # Frozen backbone'dan gradient geçebilmesi için
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        if class_index is None:
            class_index = output.argmax(dim=1).item()

        confidence = probabilities[
            0,
            class_index
        ].item()

        score = output[
            0,
            class_index
        ]

        score.backward()

        # Her feature map için ortalama gradient
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * self.activations
        ).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze()

        # 0 - 1 arasına normalize
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return (
            cam.detach().cpu(),
            class_index,
            confidence
        )


def main():

    print("\n===== DEFECTVISION GRAD-CAM =====\n")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Kullanılan cihaz: {device}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    classes = checkpoint["classes"]

    model = create_resnet18(
        num_classes=len(classes)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ResNet18'in son convolution katmanı
    target_layer = model.layer4[-1].conv2

    gradcam = GradCAM(
        model,
        target_layer
    )

    # Örnek olarak scratches sınıfından
    # bir test görüntüsü seç
    image_candidates = list(
        (TEST_DIR / "scratches").glob("*")
    )

    if not image_candidates:

        print("❌ Test görüntüsü bulunamadı.")
        return

    image_path = image_candidates[0]

    print(f"Görüntü: {image_path.name}")

    original_image = Image.open(
        image_path
    ).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(device)

    cam, predicted_index, confidence = gradcam.generate(
        input_tensor
    )

    predicted_class = classes[
        predicted_index
    ]

    print(
        f"Tahmin: {predicted_class}"
    )

    print(
        f"Güven: {confidence * 100:.2f}%"
    )

    # Grad-CAM'i 224x224'e getir
    cam_image = Image.fromarray(
        np.uint8(
            cam.numpy() * 255
        )
    )

    cam_image = cam_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    cam_array = (
        np.array(cam_image)
        / 255.0
    )

    # Orijinal görüntü
    resized_original = original_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    original_array = (
        np.array(resized_original)
        / 255.0
    )

    # Heatmap oluştur
    heatmap = plt.cm.jet(
        cam_array
    )[:, :, :3]

    # Orijinal + heatmap
    overlay = (
        0.60 * original_array
        + 0.40 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    # ------------------------------------------------
    # Görselleştirme
    # ------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 5)
    )

    axes[0].imshow(
        original_array
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")

    axes[1].imshow(
        cam_array,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM"
    )

    axes[1].axis("off")

    axes[2].imshow(
        overlay
    )

    axes[2].set_title(
        f"{predicted_class}\n"
        f"Confidence: {confidence * 100:.2f}%"
    )

    axes[2].axis("off")

    plt.suptitle(
        "DefectVision - Grad-CAM Explainability"
    )

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "resnet18_gradcam.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\n✅ Grad-CAM kaydedildi:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()