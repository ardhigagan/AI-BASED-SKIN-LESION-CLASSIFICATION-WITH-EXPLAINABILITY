"""
gradcam.py

Grad-CAM explainability: draws a heatmap on top of a lesion image showing
WHICH PARTS OF THE IMAGE the model focused on when it made its prediction.
Red/yellow = areas that influenced the prediction most. A trustworthy model
should be looking at the lesion itself, not the skin around it, hair, or the
edges of the photo.

Works with all three architectures (resnet50, efficientnet_b0, densenet121).

Colab usage. First run once:   !pip install grad-cam -q
dataset.py, model.py and evaluate.py must be in /content/ too, plus the .pt file.

    # 1) Example heatmaps from the test set: some correct, some wrong predictions
    !python gradcam.py finetuned_4blocks_efficientnet_b0.pt

    # 2) Only melanoma images (correctly caught vs missed) - best set for the report
    !python gradcam.py finetuned_4blocks_efficientnet_b0.pt --melanoma-only --n 6

    # 3) One specific image
    !python gradcam.py finetuned_4blocks_efficientnet_b0.pt --image /content/test_data/test_data/test_dataset/12.jpg

Results are saved as PNG files in /content/gradcam_results/.

The GradCamExplainer class below is written so the Flask backend can reuse it
later: explainer.explain(tensor, rgb) returns the heatmap overlay and the prediction.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/content")

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import LABEL_TO_IDX, IDX_TO_LABEL, MELANOMA_IDX, MODEL_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
from model import build_model
from evaluate import (
    load_test_data,
    predict_all,
    infer_architecture,
    resolve_model_path,
    KNOWN_ARCHITECTURES,
    BATCH_SIZE,
    NUM_WORKERS,
)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("/content/gradcam_results")


# ---------------------------------------------------------------------------
# IMAGE PREPARATION (exactly the same steps dataset.py uses for training/testing)
# ---------------------------------------------------------------------------

def prepare_image(path):
    """
    Returns (rgb_float, tensor):
      rgb_float - the 224x224 image as floats in 0-1, used to draw the heatmap on
      tensor    - the same image resized + ImageNet-normalized, shape (1, 3, 224, 224), fed to the model
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    rgb_float = rgb.astype(np.float32) / 255.0

    normalized = (rgb_float - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return rgb_float, tensor


# ---------------------------------------------------------------------------
# MODEL SETUP
# ---------------------------------------------------------------------------

def get_target_layers(model, arch):
    """The last convolutional stage of each network - where Grad-CAM reads what the model 'looked at'."""
    if arch == "resnet50":
        return [model.layer4[-1]]
    if arch in ("efficientnet_b0", "densenet121"):
        return [model.features[-1]]
    raise ValueError(f"Unknown architecture '{arch}'.")


def _patch_densenet_forward(model):
    # torchvision's DenseNet applies an in-place ReLU right after the layer Grad-CAM
    # listens to, which can break gradient capture. Same computation, not in-place.
    def forward(x):
        features = model.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return model.classifier(out)

    model.forward = forward


def load_model_for_gradcam(model_path, arch, device):
    # freeze_base=False on purpose: Grad-CAM needs gradients to flow through the whole
    # network, and a frozen backbone (requires_grad=False) would give it nothing to read.
    # The saved weights overwrite the values either way, so predictions are unchanged.
    model = build_model(architecture=arch, num_classes=len(LABEL_TO_IDX), freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(True)
    if arch == "densenet121":
        _patch_densenet_forward(model)
    return model


class GradCamExplainer:
    """Loads one trained model and produces Grad-CAM heatmaps for images."""

    def __init__(self, model_path, arch, device):
        self.arch = arch
        self.device = device
        self.model = load_model_for_gradcam(model_path, arch, device)
        self.cam = GradCAM(model=self.model, target_layers=get_target_layers(self.model, arch))

    def explain(self, tensor, rgb_float, class_idx=None):
        """
        tensor:    (1, 3, 224, 224) from prepare_image()
        rgb_float: (224, 224, 3) floats 0-1 from prepare_image()
        class_idx: which class to explain; default = the class the model predicted

        Returns a dict: overlay (uint8 RGB image with heatmap), heat (224x224 values 0-1),
        pred (predicted class index), probs (confidence for all classes).
        """
        tensor = tensor.to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        target = pred if class_idx is None else class_idx

        input_tensor = tensor.clone().requires_grad_(True)
        heat = self.cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target)])[0]
        overlay = show_cam_on_image(rgb_float, heat, use_rgb=True)
        return {"overlay": overlay, "heat": heat, "pred": pred, "probs": probs}

    def close(self):
        try:
            self.cam.activations_and_grads.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SELECTING EXAMPLES + DRAWING (no deep-learning code in here)
# ---------------------------------------------------------------------------

def pick_indices(y_true, y_pred, n, melanoma_only=False, seed=0):
    """Random (but repeatable) choice of n correctly-predicted and n wrongly-predicted images."""
    rng = random.Random(seed)
    candidates = [i for i, t in enumerate(y_true) if not melanoma_only or t == MELANOMA_IDX]
    correct = [i for i in candidates if y_true[i] == y_pred[i]]
    wrong = [i for i in candidates if y_true[i] != y_pred[i]]
    return rng.sample(correct, min(n, len(correct))), rng.sample(wrong, min(n, len(wrong)))


def save_grid(samples, title, path):
    """One row per image: the original on the left, the Grad-CAM heatmap on the right."""
    if not samples:
        return
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(6.4, 3.2 * n), squeeze=False)
    for row, s in enumerate(samples):
        axes[row, 0].imshow(s["rgb"])
        axes[row, 0].set_title(f"Image {s['id']}", fontsize=9)
        axes[row, 1].imshow(s["overlay"])
        true_part = f"True {s['true']} | " if s["true"] else ""
        color = "#1e7a3c" if s["correct"] is True else ("#c0392b" if s["correct"] is False else "black")
        axes[row, 1].set_title(f"{true_part}Pred {s['pred']} ({s['conf']:.0%})", fontsize=9, color=color)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def make_sample(image_id, true_label, result, rgb):
    pred = result["pred"]
    return {
        "id": image_id,
        "true": true_label,
        "pred": IDX_TO_LABEL[pred],
        "conf": float(result["probs"][pred]),
        "rgb": rgb,
        "overlay": result["overlay"],
        "correct": None if true_label is None else (IDX_TO_LABEL[pred] == true_label),
    }


# ---------------------------------------------------------------------------
# MODES
# ---------------------------------------------------------------------------

def run_single_image(explainer, image_path, true_label, out_dir):
    rgb, tensor = prepare_image(image_path)
    result = explainer.explain(tensor, rgb)

    print(f"\nImage: {image_path}")
    print("Top predictions:")
    for idx in np.argsort(result["probs"])[::-1][:3]:
        print(f"  {IDX_TO_LABEL[int(idx)]:5} {result['probs'][idx]:.1%}")

    stem = Path(image_path).stem
    sample = make_sample(stem, true_label.upper() if true_label else None, result, rgb)
    save_grid([sample], f"Grad-CAM - {stem}", out_dir / f"gradcam_{stem}.png")


def run_test_set(explainer, model_name, n, melanoma_only, seed, out_dir):
    ds = load_test_data()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    y_true = [LABEL_TO_IDX[label] for label in ds.labels]
    y_pred, _ = predict_all(explainer.model, loader, explainer.device)

    correct_idx, wrong_idx = pick_indices(y_true, y_pred, n, melanoma_only, seed)
    scope = "melanoma images only" if melanoma_only else "all classes"
    print(f"\nExplaining {len(correct_idx)} correct and {len(wrong_idx)} wrong predictions ({scope}).")

    tag = "melanoma_" if melanoma_only else ""
    for group, indices in (("correct", correct_idx), ("wrong", wrong_idx)):
        samples = []
        for i in indices:
            rgb, tensor = prepare_image(ds._resolve_image_path(ds.image_ids[i]))
            result = explainer.explain(tensor, rgb)
            samples.append(make_sample(ds.image_ids[i], ds.labels[i], result, rgb))

        if group == "correct":
            title = f"{model_name}: correct predictions" + (" (melanoma caught)" if melanoma_only else "")
        else:
            title = f"{model_name}: wrong predictions" + (" (melanoma missed)" if melanoma_only else "")
        save_grid(samples, title, out_dir / f"gradcam_{model_name}_{tag}{group}.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM heatmaps for a trained model.")
    parser.add_argument("model", help="trained model .pt file (uploaded to /content/)")
    parser.add_argument("--arch", choices=KNOWN_ARCHITECTURES, default=None,
                        help="architecture, if it can't be guessed from the filename")
    parser.add_argument("--image", default=None, help="explain this one image instead of test-set examples")
    parser.add_argument("--true-label", default=None, help="with --image: the real diagnosis, e.g. MEL")
    parser.add_argument("--n", type=int, default=4, help="how many correct and how many wrong examples (default 4 each)")
    parser.add_argument("--melanoma-only", action="store_true", help="only use melanoma images from the test set")
    parser.add_argument("--seed", type=int, default=0, help="changes which random examples are picked")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="folder for the PNG files")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    if model_path is None:
        raise SystemExit(f"Model file not found: {args.model}  (upload it to /content/ first)")
    arch = args.arch or infer_architecture(model_path.name)
    if arch is None:
        raise SystemExit("Can't tell the architecture from the filename. Add: --arch resnet50 (or efficientnet_b0 / densenet121)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | model: {model_path.name} | architecture: {arch}")

    explainer = GradCamExplainer(model_path, arch, device)
    out_dir = Path(args.out)
    try:
        if args.image:
            run_single_image(explainer, args.image, args.true_label, out_dir)
        else:
            run_test_set(explainer, model_path.stem, args.n, args.melanoma_only, args.seed, out_dir)
    finally:
        explainer.close()


if __name__ == "__main__":
    main()
