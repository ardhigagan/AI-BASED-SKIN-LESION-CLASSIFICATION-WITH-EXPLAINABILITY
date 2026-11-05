
import torch
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from app.ml.model_loader import model_loader


class GradCAMService:
    def __init__(self):
        self.device = model_loader.device
        self.model = model_loader.model

        # Grad-CAM requires gradients through the model
        for param in self.model.parameters():
            param.requires_grad_(True)

        # Avoid DenseNet's in-place ReLU issue with Grad-CAM
        def forward(x):
            features = self.model.features(x)
            output = F.relu(features, inplace=False)
            output = F.adaptive_avg_pool2d(output, (1, 1))
            output = torch.flatten(output, 1)
            return self.model.classifier(output)

        self.model.forward = forward
        self.model.eval()

        # Last convolutional feature layer
        self.target_layers = [self.model.features[-1]]

        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers
        )

        print("Grad-CAM service initialized successfully")

    def generate(self, tensor, rgb_float, target_class):
        tensor = tensor.to(self.device)

        # Generate Grad-CAM for the predicted class
        input_tensor = tensor.clone().requires_grad_(True)

        heatmap = self.cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(int(target_class))]
        )[0]

        # Overlay heatmap on original RGB image
        overlay = show_cam_on_image(
            rgb_float,
            heatmap,
            use_rgb=True
        )

        return {
            "heatmap": heatmap,
            "overlay": overlay
        }


gradcam_service = GradCAMService()
