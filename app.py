from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


# =========================================================
# PROJECT SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_best.pth"

IMAGE_SIZE = 224


# Modeldeki sınıf isimlerini kullanıcıya daha düzgün gösterelim
DISPLAY_NAMES = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "rolled-in_scale": "Rolled-in Scale",
    "scratches": "Scratches"
}


# =========================================================
# STREAMLIT PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="DefectVision",
    page_icon="🔍",
    layout="wide"
)


# =========================================================
# GRAD-CAM
# =========================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.forward_hook
        )

    def forward_hook(self, module, inputs, output):

        self.activations = output

        output.register_hook(
            self.save_gradient
        )

    def save_gradient(self, gradient):

        self.gradients = gradient

    def generate(
        self,
        input_tensor,
        class_index
    ):

        self.model.zero_grad()

        input_tensor.requires_grad_(True)

        output = self.model(
            input_tensor
        )

        score = output[
            0,
            class_index
        ]

        score.backward()

        # Feature maplerin önem ağırlıklarını hesapla
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights
            * self.activations
        ).sum(dim=1)

        cam = torch.relu(
            cam
        )

        cam = cam.squeeze()

        # 0 - 1 arasına normalize et
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return (
            cam
            .detach()
            .cpu()
            .numpy()
        )

    def remove_hooks(self):

        self.forward_handle.remove()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    classes = checkpoint[
        "classes"
    ]

    # Model yapısını oluşturuyoruz.
    # Ağırlıklarımız zaten kendi checkpoint dosyamızdan gelecek.
    model = models.resnet18(
        weights=None
    )

    num_features = (
        model.fc.in_features
    )

    model.fc = nn.Sequential(

        nn.Dropout(0.3),

        nn.Linear(
            num_features,
            len(classes)
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        classes,
        device
    )


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# HEADER
# =========================================================

st.title(
    "DefectVision"
)

st.subheader(
    "AI-Powered Industrial Surface Defect Detection"
)

st.write(
    """
    Upload a steel surface image and **DefectVision**
    will classify the detected surface defect using a
    **ResNet18 Transfer Learning model**.

    **Grad-CAM** is used to visualize which regions of
    the image influenced the model's prediction.
    """
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(

    "Upload a surface image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp"
    ]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    model, classes, device = (
        load_model()
    )

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(
        device
    )

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = (
            torch.softmax(
                outputs,
                dim=1
            )
        )

        confidence, predicted_index = (
            torch.max(
                probabilities,
                dim=1
            )
        )

    predicted_index = (
        predicted_index.item()
    )

    confidence = (
        confidence.item()
    )

    predicted_class = classes[
        predicted_index
    ]

    display_class = DISPLAY_NAMES.get(
        predicted_class,
        predicted_class
    )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.divider()

    st.subheader(
        "Prediction Result"
    )

    result_col1, result_col2 = (
        st.columns(2)
    )

    with result_col1:

        st.metric(
            label="Predicted Defect",
            value=display_class
        )

    with result_col2:

        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.2f}%"
        )


    # Confidence yorumu

    if confidence >= 0.90:

        st.success(
            "High-confidence prediction."
        )

    elif confidence >= 0.70:

        st.info(
            "Moderate-confidence prediction."
        )

    else:

        st.warning(
            "Low-confidence prediction. "
            "The result should be interpreted carefully."
        )


    # =====================================================
    # GRAD-CAM
    # =====================================================

    gradcam = GradCAM(
        model,
        model.layer4[-1].conv2
    )

    # Gradient gerektiği için classification'daki
    # input tensor'dan ayrı bir tensor oluşturuyoruz.
    grad_input = transform(
        original_image
    ).unsqueeze(0).to(
        device
    )

    cam = gradcam.generate(
        grad_input,
        predicted_index
    )

    gradcam.remove_hooks()


    # -----------------------------------------------------
    # HEATMAP
    # -----------------------------------------------------

    cam_image = Image.fromarray(
        np.uint8(
            cam * 255
        )
    )

    cam_image = cam_image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    cam_array = (
        np.array(
            cam_image
        )
        / 255.0
    )


    # -----------------------------------------------------
    # ORIGINAL IMAGE
    # -----------------------------------------------------

    resized_original = (
        original_image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )
    )

    original_array = (
        np.array(
            resized_original
        )
        / 255.0
    )


    # -----------------------------------------------------
    # COLOR HEATMAP
    # -----------------------------------------------------

    heatmap = plt.cm.jet(
        cam_array
    )[:, :, :3]


    # -----------------------------------------------------
    # OVERLAY
    # -----------------------------------------------------

    overlay = (
        0.60 * original_array
        + 0.40 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )


    # =====================================================
    # GRAD-CAM VISUALIZATION
    # =====================================================

    st.divider()

    st.subheader(
        "Grad-CAM Explainability"
    )

    st.write(
        """
        Warmer regions indicate areas that had a
        stronger influence on the model's prediction.
        """
    )

    image_col1, image_col2, image_col3 = (
        st.columns(3)
    )


    with image_col1:

        st.image(
            resized_original,
            caption="Original Image",
            use_container_width=True
        )


    with image_col2:

        st.image(
            heatmap,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )


    with image_col3:

        st.image(
            overlay,
            caption=(
                f"{display_class} "
                f"({confidence * 100:.2f}%)"
            ),
            use_container_width=True
        )


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    st.divider()

    st.subheader(
        "Class Probabilities"
    )

    probability_data = []

    for index, class_name in enumerate(
        classes
    ):

        probability = (
            probabilities[
                0,
                index
            ].item()
            * 100
        )

        probability_data.append(
            {
                "Defect Type": (
                    DISPLAY_NAMES.get(
                        class_name,
                        class_name
                    )
                ),
                "Probability": probability
            }
        )


    probability_df = pd.DataFrame(
        probability_data
    )

    probability_df = (
        probability_df.sort_values(
            by="Probability",
            ascending=False
        )
    )


    # Grafik
    st.bar_chart(
        probability_df.set_index(
            "Defect Type"
        )
    )


    # Tablo
    probability_table = (
        probability_df.copy()
    )

    probability_table[
        "Probability"
    ] = probability_table[
        "Probability"
    ].map(
        lambda value:
        f"{value:.2f}%"
    )

    st.dataframe(
        probability_table,
        hide_index=True,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DefectVision • PyTorch • ResNet18 Transfer Learning • Grad-CAM"
)