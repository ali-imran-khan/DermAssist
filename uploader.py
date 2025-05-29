from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

upload_blueprint = Blueprint("upload", __name__)

# Load service account credentials
SERVICE_ACCOUNT_FILE = 'dermassist-461211-b86ef1cc3728.json'
SCOPES = ['https://www.googleapis.com/auth/drive']
FOLDER_ID = '1YCXJgKy4xoBhmrmm3m5MlQ7p0yBSGj7S'

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive_service = build('drive', 'v3', credentials=credentials)

@upload_blueprint.route('/upload', methods=['POST'])
def upload_image():
    try:
        image = request.files['image']
        label = request.form.get('label')

        if not image or not label:
            return jsonify({'error': 'Image or label missing'}), 400

        filename = secure_filename(f"{label}_{image.filename}")
        filepath = os.path.join("temp", filename)
        os.makedirs("temp", exist_ok=True)
        image.save(filepath)

        file_metadata = {
            'name': filename,
            'parents': [FOLDER_ID]
        }
        media = MediaFileUpload(filepath, mimetype='image/jpeg')

        uploaded_file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        os.remove(filepath)

        return jsonify({'message': 'Uploaded successfully', 'file_id': uploaded_file.get('id')}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
