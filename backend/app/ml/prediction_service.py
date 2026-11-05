
import cv2
import numpy as np
import torch
from pathlib import Path

from app.ml.model_loader import model_loader


# Must match the normalization used during model training
MODEL_INPUT_SIZE = (224, 224)

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
)



CLASS_MAPPING = {
    0: "AK",
    1: "BCC",
    2: "BKL",
    3: "DF",
    4: "MEL",
    5: "NV",
    7: "VASC"
}



def preprocess_image(image_path):
    """
    Preprocess one uploaded dermoscopic image.

    Returns:
        rgb_float: RGB image in range 0-1 for visualization.
        tensor: Normalized image tensor of shape (1, 3, 224, 224).
    """

    image_path = str(image_path)

    # Read image using OpenCV
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError("Unable to read the uploaded image.")

    # Convert BGR to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input dimensions
    rgb = cv2.resize(
        rgb,
        MODEL_INPUT_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Convert pixel values to 0-1
    rgb_float = rgb.astype(np.float32) / 255.0

    # ImageNet normalization
    normalized = (rgb_float - IMAGENET_MEAN) / IMAGENET_STD

    # Convert HWC to CHW and add batch dimension
    tensor = torch.from_numpy(
        normalized.transpose(2, 0, 1)
    ).float().unsqueeze(0)

    return rgb_float, tensor


def predict_image(image_path):
    """
    Generate lesion classification prediction.

    Returns:
        Predicted class, predicted label,
        and probabilities for all 8 classes.
    """

    rgb_float, tensor = preprocess_image(image_path)

    # Generate predictions
    results = model_loader.predict(tensor)

    predicted_class = results["predicted_class"]
    probabilities = results["probabilities"]

    # Format class probabilities
    class_probabilities = {
        CLASS_MAPPING[index]: round(
            float(probabilities[index]) * 100, 2
        )
        for index in CLASS_MAPPING
    }

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_MAPPING[predicted_class],
        "confidence": class_probabilities[
            CLASS_MAPPING[predicted_class]
        ],
        "class_probabilities": class_probabilities
    }
