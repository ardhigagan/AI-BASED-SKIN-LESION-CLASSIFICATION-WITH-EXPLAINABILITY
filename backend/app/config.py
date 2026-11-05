
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load backend/.env
load_dotenv(BASE_DIR / ".env")


class Config:

    SECRET_KEY = os.getenv(
        "SECRET_KEY",
        "development-secret-key"
    )

    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

    UPLOAD_FOLDER = BASE_DIR / "uploads"

    GENERATED_FOLDER = BASE_DIR / "generated"

    MODEL_PATH = (
        BASE_DIR
        / "models"
        / "finetuned_2blocks_densenet121.pt"
    )

    KNOWLEDGE_PATH = (
        BASE_DIR
        / "knowledge"
        / "lesion_information.md"
    )

    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

    # Cloudinary credentials
    CLOUDINARY_CLOUD_NAME = os.getenv(
        "CLOUDINARY_CLOUD_NAME"
    )

    CLOUDINARY_API_KEY = os.getenv(
        "CLOUDINARY_API_KEY"
    )

    CLOUDINARY_API_SECRET = os.getenv(
        "CLOUDINARY_API_SECRET"
    )
