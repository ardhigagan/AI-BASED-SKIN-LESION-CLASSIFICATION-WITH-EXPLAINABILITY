"""
model.py

Builds the CNN classifier using transfer learning, as described in the PRD:
takes a pretrained ResNet50, EfficientNet-B0, or DenseNet121 (already trained on millions of
general images) and replaces its final layer so it predicts our 8 lesion
classes instead.

Usage:
    from model import build_model

    model = build_model(architecture="resnet50", num_classes=8, freeze_base=True)
"""

import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

NUM_CLASSES = 8  # AK, BCC, BKL, DF, MEL, NV, SCC, VASC — matches dataset.py's LABEL_TO_IDX


def build_model(architecture: str = "resnet50", num_classes: int = NUM_CLASSES, freeze_base: bool = True):
    """
    Builds a pretrained model with a new final layer for our lesion classes.

    architecture: "resnet50", "efficientnet_b0", or "densenet121"
    num_classes:  number of lesion types to predict (8 for this dataset)
    freeze_base:  if True, freezes all the pretrained layers so only the new
                  final layer trains (this is the Day 4 baseline approach).
                  Set to False later (Day 6) when you want to fine-tune the
                  last few layers too.

    Returns a ready-to-train PyTorch model.
    """

    if architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the final classification layer (originally outputs 1000
        # ImageNet classes) with one that outputs our 8 lesion classes.
        # This new layer is created AFTER freezing, so it's trainable by default.
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif architecture == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False

        # DenseNet's final layer is called "classifier" (a single Linear layer)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Expected 'resnet50', 'efficientnet_b0', or 'densenet121'."
        )

    return model


def unfreeze_last_layers(model, architecture: str = "resnet50", num_blocks: int = 1):
    """
    Unfreezes the last `num_blocks` blocks of the pretrained backbone, for
    fine-tuning (Day 6). Call this AFTER build_model(..., freeze_base=True)
    and after training the baseline, then use a smaller learning rate.

    architecture: must match what build_model() was called with
    num_blocks:   how many of the final blocks to unfreeze (start with 1)
    """

    if architecture == "resnet50":
        # ResNet50's blocks, in order, are layer1 -> layer2 -> layer3 -> layer4.
        # Unfreezing from the end (layer4 first) is the standard approach.
        blocks = [model.layer4, model.layer3, model.layer2, model.layer1]

    elif architecture == "efficientnet_b0":
        # EfficientNet's feature blocks are in model.features (a Sequential).
        blocks = list(reversed(list(model.features)))

    elif architecture == "densenet121":
        # DenseNet121 has 4 "dense blocks" (denseblock1..4), each separated by
        # a small "transition" layer. We group each dense block with the
        # transition layer that comes right BEFORE it, so one "block" here
        # means one dense block + its entry transition. This mirrors
        # ResNet50's 4-stage structure, so FINE_TUNE_BLOCKS=4 means fully
        # unfrozen (except the very first stem layers, which stay frozen).
        f = model.features
        blocks = [
            [f.norm5, f.denseblock4],          # last block (+ final normalization)
            [f.transition3, f.denseblock3],
            [f.transition2, f.denseblock2],
            [f.transition1, f.denseblock1],    # earliest block
        ]

    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Expected 'resnet50', 'efficientnet_b0', or 'densenet121'."
        )

    for block in blocks[:num_blocks]:
        # DenseNet blocks are lists of modules; ResNet/EfficientNet blocks are single modules.
        modules = block if isinstance(block, list) else [block]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    return model


def count_trainable_params(model) -> int:
    """Returns how many parameters will actually be updated during training —
    useful to sanity-check that freezing/unfreezing worked as expected."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick manual check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sanity check: for each architecture, show how many parameters are
    # trainable when frozen, then after unfreezing 1..4 blocks. The number
    # should grow with each extra block — if it doesn't, something's wrong.
    for arch in ["resnet50", "efficientnet_b0", "densenet121"]:
        print(f"\n=== {arch} ===")
        for n_blocks in [0, 1, 2, 3, 4]:
            model = build_model(architecture=arch, num_classes=NUM_CLASSES, freeze_base=True)
            if n_blocks > 0:
                unfreeze_last_layers(model, architecture=arch, num_blocks=n_blocks)
            print(f"  unfrozen blocks={n_blocks}: trainable params = {count_trainable_params(model):,}")
