from flask import Flask, jsonify, request
from uploader import federated_blueprint  # Make sure this file is uploader.py
from data import ImageDownloader
from global_model import FederatedEczemaLearning
import os
import json


app = Flask(__name__)
app.register_blueprint(federated_blueprint)



@app.route('/download-images', methods=['GET', 'POST'])
def download_images():
    """Download all images from qualifying clients and create ML dataset"""
    try:
        # Handle both GET and POST requests from Flutter app
        if request.method == 'POST':
            data = request.get_json() or {}
            client_id = data.get('client_id')
            file_id = data.get('file_id')
            action = data.get('action')
            
            # Log the request for debugging
            print(f"POST /download-images - client_id: {client_id}, file_id: {file_id}, action: {action}")
            
            if action == 'download_training_images':
                downloader = ImageDownloader()
                # Download images for this specific client
                results = downloader.download_images_by_client(client_ids=[client_id] if client_id else None)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Training images downloaded successfully',
                    'client_id': client_id,
                    'file_id': file_id,
                    'results': results
                })
        
        # Default GET behavior
        downloader = ImageDownloader()
        results = downloader.download_images_by_client()
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in /download-images: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/client-stats')
def get_client_stats():
    """Get statistics about all clients and their image counts"""
    try:
        downloader = ImageDownloader()
        stats = downloader.get_client_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/download-images/<client_id>')
def download_images_for_client(client_id):
    """Download images for a specific client"""
    try:
        downloader = ImageDownloader()
        results = downloader.download_images_by_client(client_ids=[client_id])
        return jsonify(results)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@app.route('/run-fedavg', methods=['POST','GET'])
def run_federated_learning():
    """
    Trigger full federated learning process:
    - Loads all available client folders
    - Runs training
    - Applies FedAvg
    - Saves global model
    """
    try:
        # Handle POST request from Flutter app
        if request.method == 'POST':
            data = request.get_json() or {}
            client_id = data.get('client_id')
            file_id = data.get('file_id')
            action = data.get('action')
            
            # Log the request for debugging
            print(f"POST /run-fedavg - client_id: {client_id}, file_id: {file_id}, action: {action}")
            
            if action == 'start_federated_training':
                # Parameters (can be made dynamic via request data if needed)
                CLIENT_DATA_DIR = "D:/uni/FYP/federation/client_data"
                GLOBAL_MODEL_DIR = "D:/uni/FYP/federation/models"
                VALIDATION_DATA_PATH = "./validation_data"
                INITIAL_MODEL_PATH = "D:/uni/FYP/federation/models/eczema_classifier_latest.pth"

                NUM_ROUNDS = 1
                LOCAL_EPOCHS = 3
                BATCH_SIZE = 8

                fed = FederatedEczemaLearning(
                    client_data_dir=CLIENT_DATA_DIR,
                    global_model_path=GLOBAL_MODEL_DIR,
                    initial_model_path=INITIAL_MODEL_PATH if os.path.exists(INITIAL_MODEL_PATH) else None
                )

                results = fed.run_federated_learning(
                    local_epochs=LOCAL_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data_path=VALIDATION_DATA_PATH if os.path.exists(VALIDATION_DATA_PATH) else None
                )

                return jsonify({
                    'status': 'success',
                    'message': 'Federated learning completed successfully',
                    'client_id': client_id,
                    'file_id': file_id,
                    'results': results
                })
        
        # Default GET behavior - run federated learning
        CLIENT_DATA_DIR = "D:/uni/FYP/federation/client_data"
        GLOBAL_MODEL_DIR = "D:/uni/FYP/federation/models"
        VALIDATION_DATA_PATH = "./validation_data"
        INITIAL_MODEL_PATH = "D:/uni/FYP/federation/models/eczema_classifier_latest.pth"

        NUM_ROUNDS = 1
        LOCAL_EPOCHS = 3
        BATCH_SIZE = 8

        fed = FederatedEczemaLearning(
            client_data_dir=CLIENT_DATA_DIR,
            global_model_path=GLOBAL_MODEL_DIR,
            initial_model_path=INITIAL_MODEL_PATH if os.path.exists(INITIAL_MODEL_PATH) else None
        )

        results = fed.run_federated_learning(
            local_epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data_path=VALIDATION_DATA_PATH if os.path.exists(VALIDATION_DATA_PATH) else None
        )

        return jsonify({
            'status': 'success',
            'message': 'Federated learning completed',
            'results': results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in /run-fedavg: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/download-images-limited/<int:limit>')
def download_images_limited(limit):
    """Download images with a limit on number of clients"""
    try:
        downloader = ImageDownloader()
        results = downloader.download_images_by_client(limit=limit)
        return jsonify(results)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/download-images-custom', methods=['POST'])
def download_images_custom():
    """Download images with custom parameters via POST request"""
    try:
        data = request.get_json() or {}
        client_ids = data.get('client_ids', None)
        limit = data.get('limit', None)
        
        downloader = ImageDownloader()
        results = downloader.download_images_by_client(
            client_ids=client_ids,
            limit=limit
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/dataset-info')
def get_dataset_info():
    """Get information about the current dataset if it exists"""
    try:
        import os
        import json
        
        dataset_info_path = './dataset/dataset_info.json'
        dataset_summary_path = './dataset/dataset_summary.txt'
        
        response = {
            'dataset_exists': False,
            'dataset_info': None,
            'summary_exists': False
        }
        
        if os.path.exists(dataset_info_path):
            response['dataset_exists'] = True
            with open(dataset_info_path, 'r') as f:
                response['dataset_info'] = json.load(f)
        
        if os.path.exists(dataset_summary_path):
            response['summary_exists'] = True
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint to verify services are working"""
    try:
        import datetime
        downloader = ImageDownloader()
        
        # Check if services are initialized
        services_ok = (downloader.drive_service is not None and 
                      downloader.firestore_client is not None)
        
        return jsonify({
            'status': 'healthy' if services_ok else 'unhealthy',
            'services_initialized': services_ok,
            'timestamp': str(datetime.datetime.now())
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)