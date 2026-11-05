
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from flask import (
    Blueprint,
    jsonify,
    request,
    current_app,
    send_from_directory
)
from werkzeug.utils import secure_filename

from app.ml.prediction_service import predict_image, preprocess_image
from app.ml.gradcam_service import gradcam_service
from app.services.knowledge_service import knowledge_service


analysis_bp = Blueprint("analysis", __name__)


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

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


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@analysis_bp.route("/health", methods=["GET"])
def health_check():

    return jsonify({
        "status": "healthy",
        "message": "Skin Lesion Analysis API is running"
    }), 200


# --------------------------------------------------
# ANALYZE IMAGE
# --------------------------------------------------

@analysis_bp.route("/analyze", methods=["POST"])
def analyze_image():

    # STEP 1: Validate upload

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

        # STEP 2: Save uploaded image

        upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
        upload_dir.mkdir(parents=True, exist_ok=True)

        upload_path = upload_dir / f"{analysis_id}.{extension}"

        file.save(upload_path)

        # STEP 3: Prediction

        prediction = predict_image(upload_path)

        # STEP 4: Educational information

        lesion_information = knowledge_service.get_lesion_information(
            prediction["predicted_label"]
        )

        # STEP 5: Image preprocessing

        rgb_float, tensor = preprocess_image(upload_path)

        # STEP 6: Grad-CAM

        cam_results = gradcam_service.generate(
            tensor=tensor,
            rgb_float=rgb_float,
            target_class=prediction["predicted_class"]
        )

        # STEP 7: Create generated directory

        output_dir = (
            Path(current_app.config["GENERATED_FOLDER"])
            / analysis_id
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # STEP 8: Save original image

        original = (rgb_float * 255).astype(np.uint8)

        cv2.imwrite(
            str(output_dir / "original.png"),
            cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        )

        # STEP 9: Save heatmap

        heatmap = (
            cam_results["heatmap"] * 255
        ).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        cv2.imwrite(
            str(output_dir / "heatmap.png"),
            heatmap_color
        )

        # STEP 10: Save overlay

        overlay = cam_results["overlay"]

        cv2.imwrite(
            str(output_dir / "overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        # STEP 11: Image URLs

        image_urls = {
            "original": f"/api/generated/{analysis_id}/original.png",
            "heatmap": f"/api/generated/{analysis_id}/heatmap.png",
            "overlay": f"/api/generated/{analysis_id}/overlay.png"
        }

        # STEP 12: Timestamp

        analyzed_at = datetime.now(timezone.utc).isoformat()

        # STEP 13: Final report

        report = {

            "analysis_id": analysis_id,

            "status": "completed",

            "analyzed_at": analyzed_at,

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
                "original_image": image_urls["original"],
                "gradcam_heatmap": image_urls["heatmap"],
                "gradcam_overlay": image_urls["overlay"]
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


# --------------------------------------------------
# SERVE GENERATED IMAGES
# --------------------------------------------------

@analysis_bp.route(
    "/generated/<analysis_id>/<filename>",
    methods=["GET"]
)
def serve_generated_image(analysis_id, filename):

    allowed_files = {
        "original.png",
        "heatmap.png",
        "overlay.png"
    }

    if filename not in allowed_files:
        return jsonify({
            "error": "Invalid image filename"
        }), 404

    output_dir = (
        Path(current_app.config["GENERATED_FOLDER"])
        / analysis_id
    )

    return send_from_directory(
        output_dir,
        filename
    )
