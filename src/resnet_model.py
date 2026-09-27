import torch.nn as nn
from torchvision import models


def create_resnet18(num_classes=6):

    # ImageNet üzerinde önceden eğitilmiş ResNet18
    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    # Önceden öğrenilmiş convolution katmanlarını dondur
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Son sınıflandırma katmanını bizim 6 sınıfımıza göre değiştir
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            num_features,
            num_classes
        )
    )

    return model


if __name__ == "__main__":

    model = create_resnet18(
        num_classes=6
    )

    print(model)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\n===== RESNET18 CHECK =====")

    print(
        f"Toplam parametre: "
        f"{total_params:,}"
    )

    print(
        f"Eğitilebilir parametre: "
        f"{trainable_params:,}"
    )

    print(
        "\n✅ ResNet18 başarıyla oluşturuldu."
    )