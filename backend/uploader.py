from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from google.cloud import firestore
import io
import datetime
import numpy as np
import cv2
from PIL import Image
import base64
import hashlib
import os
from single_test import classify_image
import config



federated_blueprint = Blueprint("federated", __name__)

SERVICE_ACCOUNT_FILE = config.SERVICE_ACCOUNT_FILE
SCOPES = ['https://www.googleapis.com/auth/drive']

MIN_SKIN_THRESHOLD = 20
BLUR_THRESHOLD = 100
MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 200
MIN_CONTRAST = 30
MAX_QUALITY_ISSUES = 1
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Folder IDs on Google Drive
FOLDER_IDS = config.DRIVE_FOLDER_IDS

# Setup Google Drive and Firestore
try:
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)
    firestore_client = firestore.Client.from_service_account_json(SERVICE_ACCOUNT_FILE)
    print("[INFO] Google services initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize Google services: {e}")
    drive_service = None
    firestore_client = None

def safe_print(message):
    """Safe printing function to avoid encoding issues"""
    try:
        print(message)
    except:
        print("[LOG] Message encoding issue")

def calculate_image_hash(image_data):
    """SHA-256 of the raw image bytes, used for duplicate detection."""
    try:
        return hashlib.sha256(image_data).hexdigest()
    except Exception as e:
        safe_print(f"[ERROR] Hash calculation failed: {e}")
        return None

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for Firestore compatibility"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def detect_skin_pixels_simple(image_array):
    try:
        if image_array is None or len(image_array.shape) != 3:
            return 0, None

        hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image_array, cv2.COLOR_BGR2YCrCb)

        lower_hsv = np.array([0, 10, 30])
        upper_hsv = np.array([25, 255, 255])
        lower_ycrcb = np.array([0, 120, 70])
        upper_ycrcb = np.array([255, 180, 150])

        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # Check for large skin-like region
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_skin_region = any(cv2.contourArea(c) > 5000 for c in contours)
        if not large_skin_region:
            return 0, skin_mask

        # dominant colour filter
        try:
            resized = cv2.resize(image_array, (100, 100))
            pixels = resized.reshape((-1, 3)).astype(np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, _ = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            _, counts = np.unique(labels, return_counts=True)
            prominent_clusters = sum(count > (0.05 * len(pixels)) for count in counts)

            if prominent_clusters > 3:
                return 0, skin_mask  # Too many dominant colors (likely cloth)
        except Exception as sub_e:
            safe_print(f"[WARN] Color analysis failed: {sub_e}")

        total_pixels = skin_mask.shape[0] * skin_mask.shape[1]
        skin_pixels = cv2.countNonZero(skin_mask)
        skin_percentage = (skin_pixels / float(total_pixels)) * 100 if total_pixels > 0 else 0

        return skin_percentage, skin_mask

    except Exception as e:
        safe_print(f"[ERROR] Skin detection failed: {e}")
        return 0, None


def analyze_image_quality_simple(image_array):
    """Brightness / contrast / blur checks."""
    try:
        if image_array is None or len(image_array.shape) != 3:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Calculate blur metric using Laplacian
        blur_metric = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate basic statistics
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # Firestore can't store numpy bools
        quality_info = {
            'blur_metric': float(blur_metric),
            'brightness': brightness,
            'contrast': contrast,
            'is_blurry': bool(blur_metric < BLUR_THRESHOLD),
            'is_too_dark': bool(brightness < MIN_BRIGHTNESS),
            'is_too_bright': bool(brightness > MAX_BRIGHTNESS),
            'low_contrast': bool(contrast < MIN_CONTRAST)
        }
        
        return quality_info
        
    except Exception as e:
        safe_print(f"[ERROR] Quality analysis failed: {e}")
        return None

def check_duplicate_simple(image_hash, client_id):
    """Check Firestore for an image with the same hash."""
    try:
        if not firestore_client or not image_hash:
            return False
        
        uploads_ref = firestore_client.collection('image_uploads')
        # Use the new filter syntax to avoid the warning
        query = uploads_ref.where(filter=firestore.FieldFilter('client_id', '==', client_id)).where(filter=firestore.FieldFilter('image_hash', '==', image_hash)).limit(1)
        docs = list(query.stream())
        return len(docs) > 0
    except Exception as e:
        safe_print(f"[ERROR] Duplicate check failed: {e}")
        return False

def download_image_from_drive(file_id):
    """Fetch an image from Drive by file id."""
    try:
        if not drive_service:
            return None
        
        request_obj = drive_service.files().get_media(fileId=file_id)
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request_obj)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        file_io.seek(0)
        return file_io.getvalue()
        
    except Exception as e:
        safe_print(f"[ERROR] Failed to download image {file_id}: {e}")
        return None
    

@federated_blueprint.route('/upload', methods=['POST'])
def upload_with_analysis_compatible():
    """Compatible upload function with detailed debugging"""
    try:
        safe_print(f"[DEBUG] Upload request received at {datetime.datetime.now()}")

        if not request.files.get('image'):
            safe_print("[DEBUG] REJECTION: No image file provided")
            return jsonify({'error': 'No image file provided', 'debug': 'missing_image_file'}), 400

        image = request.files['image']
        image_data = image.read()
        if not image_data:
            safe_print("[DEBUG] REJECTION: Empty image data")
            return jsonify({'error': 'Empty image data', 'debug': 'empty_image_data'}), 400

        raw_label = request.form.get('label', '').strip()
        client_id = request.form.get('client_id', '').strip()

        if not client_id:
            safe_print("[DEBUG] REJECTION: No client_id provided")
            return jsonify({'error': 'No client_id provided', 'debug': 'missing_client_id'}), 400

        if not raw_label:
            safe_print("[DEBUG] No label provided — classifying...")
            prediction = classify_image(image_data)

            # Normalize and reformat output
            eczema_score = prediction.get("Eczema", 0)
            healthy_score = prediction.get("Healthy", 0)
            predicted_class = prediction.get("Predicted_Class", "unknown").lower()

            formatted_results = [
                {"label": "eczema", "confidence": float(round(eczema_score / 100 if eczema_score > 1 else eczema_score, 4))},
                {"label": "healthy", "confidence": float(round(healthy_score / 100 if healthy_score > 1 else healthy_score, 4))},
            ]

            label = predicted_class
            safe_print(f"[DEBUG] Predicted label: {label} with confidence {formatted_results}")

        else:
            label = raw_label.lower().replace('\r', '').replace('\n', '').replace(' ', '_')
            formatted_results = [{"label": label, "confidence": 1.0}]

        safe_print(f"[DEBUG] Processing upload - Client: {client_id[:8]}... | Label: {label} | Filename: {image.filename}")

        file_size_mb = len(image_data) / (1024 * 1024)
        safe_print(f"[DEBUG] Image file size: {file_size_mb:.2f} MB")

        if len(image_data) > MAX_FILE_SIZE:
            safe_print(f"[DEBUG] REJECTION: File too large ({file_size_mb:.2f} MB > {MAX_FILE_SIZE/(1024*1024)} MB)")
            return jsonify({'error': 'File too large', 'debug': f'file_size_{file_size_mb:.2f}MB'}), 400

        safe_print("[DEBUG] Calculating image hash...")
        image_hash = calculate_image_hash(image_data)
        if image_hash:
            safe_print(f"[DEBUG] Image hash: {image_hash[:16]}...")
            if check_duplicate_simple(image_hash, client_id):
                safe_print(f"[DEBUG] REJECTION: Duplicate image detected (hash: {image_hash[:16]}...)")
                return jsonify({'error': 'Duplicate image detected', 'debug': f'duplicate_hash_{image_hash[:8]}'}), 400
        else:
            safe_print("[DEBUG] WARNING: Could not calculate image hash")

        safe_print("[DEBUG] Converting image to OpenCV format...")
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            safe_print(f"[DEBUG] REJECTION: Failed to decode image - {str(e)}")
            return jsonify({'error': 'Failed to decode image', 'debug': f'decode_error_{str(e)[:50]}'}), 400

        if cv_image is None:
            safe_print("[DEBUG] REJECTION: Invalid image format - cv2.imdecode returned None")
            return jsonify({'error': 'Invalid image format', 'debug': 'opencv_decode_failed'}), 400

        image_dimensions = f"{cv_image.shape[1]}x{cv_image.shape[0]}"
        safe_print(f"[DEBUG] Image dimensions: {image_dimensions} (WxH)")

        safe_print("[DEBUG] Starting skin detection analysis...")
        skin_percentage, skin_mask = detect_skin_pixels_simple(cv_image)
        safe_print(f"[DEBUG] Skin detection result: {skin_percentage:.1f}%")

        if skin_percentage < MIN_SKIN_THRESHOLD:
            safe_print("[DEBUG] REJECTION: Insufficient skin region — rejecting image")
            return jsonify({
                'message': 'Image rejected: no skin detected',
                'file_id': None,
                'accepted': False,
                'skin_percentage': skin_percentage,
                'quality_score': 0,
                'quality_issues': ['no_skin_detected'],
                'results': formatted_results,
                'debug': {
                    'image_dimensions': image_dimensions,
                    'file_size_mb': file_size_mb,
                    'processing_successful': False
                }
            }), 400


        safe_print("[DEBUG] Starting quality analysis...")
        quality_metrics = analyze_image_quality_simple(cv_image)
        if quality_metrics:
            safe_print(f"[DEBUG] Quality metrics - Blur: {quality_metrics.get('blur_metric', 0):.1f}, Brightness: {quality_metrics.get('brightness', 0):.1f}, Contrast: {quality_metrics.get('contrast', 0):.1f}")
        else:
            safe_print("[DEBUG] WARNING: Quality analysis failed")

        is_skin_image = skin_percentage >= MIN_SKIN_THRESHOLD
        quality_issues = []

        if quality_metrics:
            if quality_metrics.get('is_blurry', False):
                quality_issues.append('blurry')
            if quality_metrics.get('is_too_dark', False):
                quality_issues.append('too_dark')
            if quality_metrics.get('is_too_bright', False):
                quality_issues.append('too_bright')
            if quality_metrics.get('low_contrast', False):
                quality_issues.append('low_contrast')

        safe_print(f"[DEBUG] Quality issues found: {quality_issues} (count: {len(quality_issues)}, max allowed: {MAX_QUALITY_ISSUES})")

        accept_image = is_skin_image and len(quality_issues) <= MAX_QUALITY_ISSUES
        quality_score = 100 - len(quality_issues) * 20 + max(0, (skin_percentage - MIN_SKIN_THRESHOLD) * 2)
        quality_score = max(0, min(100, quality_score))

        if not accept_image:
            return jsonify({
                'message': 'Image uploaded and analyzed successfully',
                'file_id': None,
                'accepted': False,
                'skin_percentage': skin_percentage,
                'quality_score': quality_score,
                'quality_issues': quality_issues,
                'results': formatted_results,
                'debug': {
                    'image_dimensions': image_dimensions,
                    'file_size_mb': file_size_mb,
                    'processing_successful': True
                }
            }), 200

        safe_print("[DEBUG] Starting Google Drive upload...")
        if not drive_service:
            safe_print("[DEBUG] REJECTION: Drive service not available")
            return jsonify({'error': 'Drive service not available', 'debug': 'drive_service_unavailable'}), 500

        try:
            folder_id = FOLDER_IDS[label]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = secure_filename(f"{client_id}_{label}_{timestamp}_{image.filename}")
            image_stream = io.BytesIO(image_data)

            media = MediaIoBaseUpload(image_stream, mimetype=image.content_type, resumable=True)
            file_metadata = {'name': filename, 'parents': [folder_id], 'mimeType': image.content_type}
            uploaded_file = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True 
            ).execute()
            file_id = uploaded_file.get('id')
        except Exception as e:
            safe_print(f"[DEBUG] REJECTION: Drive upload failed - {str(e)}")
            return jsonify({'error': 'Upload to Drive failed', 'debug': f'drive_error_{str(e)[:50]}'}), 500

        if firestore_client:
            try:
                firestore_client.collection('image_uploads').document(file_id).set({
                    'file_id': file_id,
                    'filename': filename,
                    'original_filename': image.filename,
                    'client_id': client_id,
                    'label': label,
                    'image_hash': image_hash,
                    'upload_time': firestore.SERVER_TIMESTAMP,
                    'upload_datetime': datetime.datetime.now().isoformat(),
                    'mime_type': image.content_type,
                    'file_size': len(image_data),
                    'skin_percentage': float(skin_percentage),
                    'quality_metrics': convert_numpy_types(quality_metrics),
                    'quality_score': float(quality_score),
                    'accepted': True,
                    'quality_issues': quality_issues,
                    'height': int(cv_image.shape[0]),
                    'width': int(cv_image.shape[1]),
                    'channels': int(cv_image.shape[2]),
                    'used_in_training': False
                })
            except Exception as e:
                safe_print(f"[DEBUG] WARNING: Firestore write failed - {str(e)}")

        safe_print(f"[UPLOAD SUCCESS] Client: {client_id[:8]}... | Label: {label} | File ID: {file_id}")
        return jsonify({
            'file_id': file_id,
            'results': formatted_results,
            'message': 'Image uploaded and analyzed successfully',
            'accepted': True,
            'skin_percentage': skin_percentage,
            'quality_score': quality_score,
            'quality_issues': quality_issues
        }), 200

    except Exception as e:
        safe_print(f"[DEBUG] CRITICAL ERROR: {str(e)}")
        import traceback
        safe_print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'debug': f'exception_{type(e).__name__}_{str(e)[:50]}'
        }), 500
