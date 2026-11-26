"""Central configuration. All values come from environment variables (see ../.env.example)."""
import os

SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")

# Google Drive folders that hold uploaded images, one per class
DRIVE_FOLDER_IDS = {
    "eczema": os.environ.get("DRIVE_FOLDER_ECZEMA", ""),
    "healthy": os.environ.get("DRIVE_FOLDER_HEALTHY", ""),
}

CLIENT_DATA_DIR = os.environ.get("CLIENT_DATA_DIR", "client_data")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_DIR, "eczema_classifier_latest.pth"))
VALIDATION_DATA_PATH = os.environ.get("VALIDATION_DATA_PATH", "validation_data")
