
from flask import Flask
from flask_cors import CORS

from app.config import Config


def create_app():

    app = Flask(__name__)

    app.config.from_object(Config)

    CORS(app)

    # Create required directories
    app.config["UPLOAD_FOLDER"].mkdir(
        parents=True, exist_ok=True
    )

    app.config["GENERATED_FOLDER"].mkdir(
        parents=True, exist_ok=True
    )

    # Register routes
    from app.routes.analysis import analysis_bp

    app.register_blueprint(
        analysis_bp,
        url_prefix="/api"
    )

    return app
