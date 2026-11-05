
import torch

from app.config import Config
from pipeline.model import build_model


class ModelLoader:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self._load_model()

    def _load_model(self):

        model = build_model(
            architecture="densenet121",
            num_classes=8,
            freeze_base=False
        )

        checkpoint = torch.load(
            Config.MODEL_PATH,
            map_location=self.device,
            weights_only=False
        )

        # Support checkpoints saved as state_dict or wrapped dictionaries
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]

        # Handle DataParallel checkpoints
        checkpoint = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint.items()
        }

        model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        print(f"Model loaded successfully on {self.device}")

        return model

    def predict(self, tensor):

        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        probabilities = torch.softmax(output, dim=1)

        predicted_class = probabilities.argmax(dim=1).item()

        return {
            "predicted_class": predicted_class,
            "probabilities": probabilities[0].cpu().tolist()
        }


model_loader = ModelLoader()
