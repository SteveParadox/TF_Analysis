import os
import logging
from flask import Flask

def create_app(config_object=None):
    """Application factory for creating Flask app instances."""
    app = Flask(__name__)

    # Load default config or user-defined
    app.config.from_object(config_object)

    # Initialize logging
    configure_logging(app)

    # Register blueprints if modularized
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app

def configure_logging(app):
    """Attach rotating log file handler."""
    log_level = app.config.get("LOG_LEVEL", logging.INFO)
    log_file = app.config.get("LOG_FILE", "logs/app.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handler = logging.FileHandler(log_file)
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )
    handler.setFormatter(formatter)

    app.logger.setLevel(log_level)
    app.logger.addHandler(handler)
