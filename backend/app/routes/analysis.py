
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import cloudinary
import cloudinary.uploader

from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

from app.ml.prediction_service import predict_image, preprocess_image
from app.ml.gradcam_service import gradcam_service
from app.services.knowledge_service import knowledge_service


analysis_bp = Blueprint("analysis", __name__)

MODEL_NAME = "DenseNet121"
MODEL_VERSION = "Fine-tuned DenseNet121"
DISPLAYED_CLASSES = 7

MEDICAL_DISCLAIMER = (
    "This application is intended for educational and research purposes "
    "only. Its predictions are not medical diagnoses and must not be used "
    "to make treatment decisions. AI predictions may be incorrect. "
    "Consult a qualified dermatologist for professional evaluation. "
    "A suspicious, changing, bleeding, painful, or non-healing lesion "
    "should be assessed by a healthcare professional."
)


def configure_cloudinary():

    cloudinary.config(
        cloud_name=current_app.config["CLOUDINARY_CLOUD_NAME"],
        api_key=current_app.config["CLOUDINARY_API_KEY"],
        api_secret=current_app.config["CLOUDINARY_API_SECRET"],
        secure=True
    )


def upload_to_cloudinary(image_path, analysis_id, image_type):

    result = cloudinary.uploader.upload(
        str(image_path),
        folder=f"skin-lesion-analysis/{analysis_id}",
        public_id=image_type,
        resource_type="image",
        overwrite=True
    )

    return result["secure_url"]


# HEALTH CHECK

@analysis_bp.route("/health", methods=["GET"])
def health_check():

    return jsonify({
        "status": "healthy",
        "message": "Skin Lesion Analysis API is running"
    }), 200


# ANALYZE IMAGE

@analysis_bp.route("/analyze", methods=["POST"])
def analyze_image():

    if "image" not in request.files:
        return jsonify({
            "status": "failed",
            "error": "No image file provided"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "status": "failed",
            "error": "No image selected"
        }), 400

    filename = secure_filename(file.filename)

    extension = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    allowed_extensions = current_app.config["ALLOWED_EXTENSIONS"]

    if extension not in allowed_extensions:
        return jsonify({
            "status": "failed",
            "error": "Unsupported image format. Use JPG, JPEG, or PNG."
        }), 400

    analysis_id = str(uuid.uuid4())

    try:

        # Configure Cloudinary
        configure_cloudinary()

        # Save upload temporarily
        upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
        upload_dir.mkdir(parents=True, exist_ok=True)

        upload_path = upload_dir / f"{analysis_id}.{extension}"
        file.save(upload_path)

        # Prediction
        prediction = predict_image(upload_path)

        # Educational information
        lesion_information = knowledge_service.get_lesion_information(
            prediction["predicted_label"]
        )

        # Preprocessing
        rgb_float, tensor = preprocess_image(upload_path)

        # Grad-CAM
        cam_results = gradcam_service.generate(
            tensor=tensor,
            rgb_float=rgb_float,
            target_class=prediction["predicted_class"]
        )

        # Temporary generated directory
        output_dir = (
            Path(current_app.config["GENERATED_FOLDER"])
            / analysis_id
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Original image
        original = (rgb_float * 255).astype(np.uint8)
        original_path = output_dir / "original.png"

        cv2.imwrite(
            str(original_path),
            cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        )

        # Heatmap
        heatmap = (cam_results["heatmap"] * 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap_path = output_dir / "heatmap.png"

        cv2.imwrite(
            str(heatmap_path),
            heatmap_color
        )

        # Overlay
        overlay_path = output_dir / "overlay.png"

        cv2.imwrite(
            str(overlay_path),
            cv2.cvtColor(cam_results["overlay"], cv2.COLOR_RGB2BGR)
        )

        # Cloudinary uploads
        original_url = upload_to_cloudinary(
            original_path, analysis_id, "original"
        )

        heatmap_url = upload_to_cloudinary(
            heatmap_path, analysis_id, "heatmap"
        )

        overlay_url = upload_to_cloudinary(
            overlay_path, analysis_id, "overlay"
        )

        # Report
        report = {
            "analysis_id": analysis_id,
            "status": "completed",
            "analyzed_at": datetime.now(timezone.utc).isoformat(),

            "model": {
                "name": MODEL_NAME,
                "version": MODEL_VERSION,
                "architecture": "DenseNet121",
                "num_classes": DISPLAYED_CLASSES
            },

            "prediction": {
                "predicted_class": prediction["predicted_class"],
                "predicted_label": prediction["predicted_label"],
                "confidence": prediction["confidence"],
                "class_probabilities": prediction["class_probabilities"]
            },

            "visualization": {
                "original_image": original_url,
                "gradcam_heatmap": heatmap_url,
                "gradcam_overlay": overlay_url
            },

            "educational_information": lesion_information,
            "medical_disclaimer": MEDICAL_DISCLAIMER
        }

        return jsonify(report), 200

    except Exception:

        current_app.logger.exception("Image analysis failed")

        return jsonify({
            "status": "failed",
            "analysis_id": analysis_id,
            "error": "Image analysis failed",
            "message": "An unexpected error occurred during analysis."
        }), 500
