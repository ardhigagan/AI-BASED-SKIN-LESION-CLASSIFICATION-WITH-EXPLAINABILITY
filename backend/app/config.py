
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Config:

    SECRET_KEY = "development-secret-key"

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
