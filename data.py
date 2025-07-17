import os
import io
import random
import shutil
import json
from collections import defaultdict, Counter
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.cloud import firestore
import datetime

class ImageDownloader:
    def __init__(self):
        # Configuration
        self.SERVICE_ACCOUNT_FILE = 'service_account.json'
        self.SCOPES = ['https://www.googleapis.com/auth/drive.file']
        self.BASE_PATH = 'D:/uni/FYP/federation/client_data'
        self.MODELS_PATH = 'D:/uni/FYP/federation/models'
        self.MIN_IMAGES_PER_CLIENT = 6
        self.MIN_QUALIFYING_CLIENTS =1
        
        # Data split ratios
        self.TRAIN_RATIO = 0.70
        self.TEST_RATIO = 0.15
        self.VALIDATION_RATIO = 0.15
        
        # Set random seed for reproducible splits
        self.RANDOM_SEED = 42
        random.seed(self.RANDOM_SEED)
        
        # Folder IDs from your uploader
        self.FOLDER_IDS = {
            'eczema': '1Mv_PFua73_1SD-VIwrQquK0kwCGZ9AqS',
            'healthy': '1uqcWoboObnjY0BxUe3PU61vjEXs21Tt6'
        }
        
        # Classes are now defined as eczema and healthy (no numeric mapping)
        self.CLASSES = ['eczema', 'healthy']
        
        # Initialize services
        self.drive_service = None
        self.firestore_client = None
        self._initialize_services()
    
    def safe_print(self, message):
        """Safe printing function to avoid encoding issues"""
        try:
            print(message)
        except:
            print("[LOG] Message encoding issue")
    
    def _initialize_services(self):
        """Initialize Google Drive and Firestore services"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES
            )
            self.drive_service = build('drive', 'v3', credentials=credentials)
            self.firestore_client = firestore.Client.from_service_account_json(self.SERVICE_ACCOUNT_FILE)
            self.safe_print("[INFO] Google services initialized successfully")
            return True
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to initialize Google services: {e}")
            return False
    
    def _cleanup_existing_dataset(self):
        """Remove existing client_data folder if it exists"""
        try:
            if os.path.exists(self.BASE_PATH):
                self.safe_print(f"[INFO] Removing existing dataset folder: {self.BASE_PATH}")
                shutil.rmtree(self.BASE_PATH)
                self.safe_print("[INFO] Existing dataset folder removed successfully")
            else:
                self.safe_print("[INFO] No existing dataset folder found")
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to remove existing dataset folder: {e}")
            raise Exception(f"Could not clean up existing dataset: {e}")
    
    def get_client_statistics(self):
        """Get comprehensive client statistics"""
        if not self.firestore_client:
            return {'error': 'Firestore not initialized'}
        
        try:
            client_counts = self._get_client_image_counts()
            
            if not client_counts:
                return {'error': 'No client data found'}
            
            # Calculate statistics
            total_clients = len(client_counts)
            qualifying_clients = self._filter_qualifying_clients(client_counts)
            
            stats = {
                'total_clients': total_clients,
                'qualifying_clients': len(qualifying_clients),
                'min_images_required': self.MIN_IMAGES_PER_CLIENT,
                'clients': []
            }
            
            for client_id, counts in client_counts.items():
                client_info = {
                    'client_id': client_id[:12] + '...',  # Truncate for privacy
                    'full_client_id': client_id,
                    'eczema_images': counts['eczema'],
                    'healthy_images': counts['healthy'],
                    'total_images': counts['total'],
                    'qualifies': counts['total'] >= self.MIN_IMAGES_PER_CLIENT
                }
                stats['clients'].append(client_info)
            
            # Sort by total images (descending)
            stats['clients'].sort(key=lambda x: x['total_images'], reverse=True)
            
            return stats
            
        except Exception as e:
            return {'error': f'Failed to get statistics: {str(e)}'}
    
    def _get_client_image_counts(self):
        """Get image counts per client from Firestore"""
        try:
            self.safe_print("[INFO] Fetching client image counts from Firestore...")
            
            uploads_ref = self.firestore_client.collection('image_uploads')
            client_counts = defaultdict(lambda: defaultdict(int))
            
            for doc in uploads_ref.stream():
                doc_data = doc.to_dict()
                client_id = doc_data.get('client_id', 'unknown')
                label = doc_data.get('label', 'unknown')
                
                if client_id != 'unknown' and label in self.FOLDER_IDS:
                    client_counts[client_id][label] += 1
            
            # Convert to regular dict and calculate totals
            result = {}
            for client_id, labels in client_counts.items():
                total_images = sum(labels.values())
                result[client_id] = {
                    'eczema': labels.get('eczema', 0),
                    'healthy': labels.get('healthy', 0),
                    'total': total_images
                }
            
            return result
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to get client counts: {e}")
            return {}
    
    def _filter_qualifying_clients(self, client_counts):
        """Filter clients that have at least MIN_IMAGES_PER_CLIENT images"""
        qualifying_clients = {}
        
        for client_id, counts in client_counts.items():
            if counts['total'] >= self.MIN_IMAGES_PER_CLIENT:
                qualifying_clients[client_id] = counts
        
        # Sort by total count (highest first)
        sorted_clients = dict(sorted(qualifying_clients.items(), 
                                    key=lambda x: x[1]['total'], reverse=True))
        
        return sorted_clients
    
    def download_images_by_client(self, client_ids=None, limit=None):
        """Download images and create client-based dataset structure"""
        if not self.drive_service or not self.firestore_client:
            return {'error': 'Services not initialized'}
        
        debug_info = {
            'total_clients_found': 0,
            'clients_below_minimum': 0,
            'qualifying_clients': 0,
            'clients_requested_not_found': [],
            'clients_with_no_files': [],
            'clients_processed': 0,
            'skipped_clients': [],
            'failed_downloads': [],
            'successful_downloads': 0
        }
        
        try:
            # Clean up existing dataset folder first
            self.safe_print("[DEBUG] Step 0: Cleaning up existing dataset...")
            self._cleanup_existing_dataset()
            
            # Get client image counts
            self.safe_print("[DEBUG] Step 1: Getting client image counts...")
            client_counts = self._get_client_image_counts()
            if not client_counts:
                return {'error': 'No client data found', 'debug': debug_info}
            
            debug_info['total_clients_found'] = len(client_counts)
            self.safe_print(f"[DEBUG] Found {len(client_counts)} total clients in database")
            
            # Filter qualifying clients
            self.safe_print("[DEBUG] Step 2: Filtering qualifying clients...")
            qualifying_clients = self._filter_qualifying_clients(client_counts)
            
            # Track clients that didn't qualify
            debug_info['clients_below_minimum'] = len(client_counts) - len(qualifying_clients)
            debug_info['qualifying_clients'] = len(qualifying_clients)
            
            self.safe_print(f"[DEBUG] {debug_info['clients_below_minimum']} clients skipped (below {self.MIN_IMAGES_PER_CLIENT} images)")
            self.safe_print(f"[DEBUG] {debug_info['qualifying_clients']} clients qualify")
            
            # Track skipped clients (below minimum)
            for client_id, counts in client_counts.items():
                if counts['total'] < self.MIN_IMAGES_PER_CLIENT:
                    debug_info['skipped_clients'].append({
                        'client_id': client_id[:12] + '...',
                        'reason': 'below_minimum_images',
                        'image_count': counts['total'],
                        'minimum_required': self.MIN_IMAGES_PER_CLIENT
                    })
            
            # If specific client_ids provided, filter to those
            if client_ids:
                self.safe_print(f"[DEBUG] Step 3: Filtering to specific client IDs: {len(client_ids)} requested")
                original_qualifying = set(qualifying_clients.keys())
                requested_clients = set(client_ids)
                
                # Track requested clients not found
                debug_info['clients_requested_not_found'] = [
                    cid[:12] + '...' for cid in requested_clients - original_qualifying
                ]
                
                qualifying_clients = {
                    cid: counts for cid, counts in qualifying_clients.items() 
                    if cid in client_ids
                }
                
                self.safe_print(f"[DEBUG] {len(debug_info['clients_requested_not_found'])} requested clients not found or don't qualify")
                self.safe_print(f"[DEBUG] {len(qualifying_clients)} clients matched from request")
            
            # Apply limit if specified
            if limit:
                self.safe_print(f"[DEBUG] Step 4: Applying limit of {limit} clients")
                original_count = len(qualifying_clients)
                qualifying_clients = dict(list(qualifying_clients.items())[:limit])
                skipped_due_to_limit = original_count - len(qualifying_clients)
                self.safe_print(f"[DEBUG] {skipped_due_to_limit} qualifying clients skipped due to limit")
            
            if not qualifying_clients:
                return {
                    'error': 'No qualifying clients found', 
                    'debug': debug_info
                }
            
            # Get file information
            self.safe_print("[DEBUG] Step 5: Getting file information from Firestore...")
            client_files = self._get_client_files_from_firestore(qualifying_clients.keys())
            
            # Track clients with no files found
            for client_id in qualifying_clients.keys():
                if client_id not in client_files or not any(client_files[client_id].values()):
                    debug_info['clients_with_no_files'].append({
                        'client_id': client_id[:12] + '...',
                        'reason': 'no_files_found_in_drive',
                        'expected_images': qualifying_clients[client_id]['total']
                    })
            
            if not client_files:
                return {
                    'error': 'No file information found', 
                    'debug': debug_info
                }
            
            self.safe_print(f"[DEBUG] {len(client_files)} clients have downloadable files")
            self.safe_print(f"[DEBUG] {len(debug_info['clients_with_no_files'])} clients had no files found")
            
            debug_info['clients_processed'] = len(client_files)
            
            # Create client-based dataset structure
            self.safe_print("[DEBUG] Step 6: Creating client-based dataset structure...")
            client_folder_mapping = self._create_client_dataset_structure(qualifying_clients.keys())
            
            # Download and organize images by client
            self.safe_print("[DEBUG] Step 7: Starting download process...")
            download_stats = self._download_and_organize_images_by_client(
                client_files, qualifying_clients, client_folder_mapping, debug_info
            )
            
            # Create summary files
            self.safe_print("[DEBUG] Step 8: Creating summary files...")
            self._create_summary_files(qualifying_clients, download_stats, client_folder_mapping)
            
            # Calculate totals
            total_downloaded = sum(
                sum(client_stats['healthy'] + client_stats['eczema'] 
                    for client_stats in split_stats.values())
                for split_stats in download_stats.values()
            )
            
            debug_info['successful_downloads'] = total_downloaded
            
            self.safe_print(f"[DEBUG] FINAL SUMMARY:")
            self.safe_print(f"[DEBUG] - Total clients in database: {debug_info['total_clients_found']}")
            self.safe_print(f"[DEBUG] - Clients below minimum: {debug_info['clients_below_minimum']}")
            self.safe_print(f"[DEBUG] - Qualifying clients: {debug_info['qualifying_clients']}")
            self.safe_print(f"[DEBUG] - Clients processed: {debug_info['clients_processed']}")
            self.safe_print(f"[DEBUG] - Successful downloads: {debug_info['successful_downloads']}")
            self.safe_print(f"[DEBUG] - Failed downloads: {len(debug_info['failed_downloads'])}")
            
            return {
                'status': 'success',
                'total_clients_processed': len(qualifying_clients),
                'total_images_downloaded': total_downloaded,
                'dataset_location': os.path.abspath(self.BASE_PATH),
                'debug': debug_info,
                'client_folders': client_folder_mapping,
                'download_stats': download_stats
            }
            
        except Exception as e:
            self.safe_print(f"[DEBUG] EXCEPTION: {str(e)}")
            return {'error': f'Download failed: {str(e)}', 'debug': debug_info}
    
    def _get_client_files_from_firestore(self, target_clients):
        """Get file information for target clients from Firestore"""
        try:
            client_files = defaultdict(lambda: defaultdict(list))
            uploads_ref = self.firestore_client.collection('image_uploads')
            
            for doc in uploads_ref.stream():
                doc_data = doc.to_dict()
                client_id = doc_data.get('client_id')
                
                if client_id in target_clients:
                    file_info = {
                        'file_id': doc_data.get('file_id'),
                        'filename': doc_data.get('filename'),
                        'original_filename': doc_data.get('original_filename'),
                        'upload_time': doc_data.get('upload_datetime'),
                        'skin_percentage': doc_data.get('skin_percentage', 0),
                        'quality_score': doc_data.get('quality_score', 0)
                    }
                    
                    label = doc_data.get('label')
                    if label in self.FOLDER_IDS and file_info['file_id']:
                        client_files[client_id][label].append(file_info)
            
            return dict(client_files)
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to get file information: {e}")
            return {}
    
    def _create_client_dataset_structure(self, client_ids):
        """Create client-based directory structure"""
        client_folder_mapping = {}
        splits = ['train', 'test', 'validate']
        
        for i, client_id in enumerate(client_ids, 1):
            client_folder_name = f"client_{i:03d}"
            client_folder_mapping[client_id] = client_folder_name
            
            for split in splits:
                for class_name in self.CLASSES:
                    dir_path = os.path.join(self.BASE_PATH, client_folder_name, split, class_name)
                    os.makedirs(dir_path, exist_ok=True)
        
        os.makedirs(self.MODELS_PATH, exist_ok=True)
        return client_folder_mapping
    
    def _split_images_by_client(self, client_files):
        """Split each client's images into train/test/validate"""
        client_splits = {}
        
        for client_id, files_by_label in client_files.items():
            client_splits[client_id] = {'train': {}, 'test': {}, 'validate': {}}
            
            for label, files in files_by_label.items():
                if not files:
                    continue
                
                # Shuffle files for random split
                shuffled_files = files.copy()
                random.shuffle(shuffled_files)
                
                total_files = len(shuffled_files)
                train_count = int(total_files * self.TRAIN_RATIO)
                test_count = int(total_files * self.TEST_RATIO)
                
                # Split files
                train_files = shuffled_files[:train_count]
                test_files = shuffled_files[train_count:train_count + test_count]
                validate_files = shuffled_files[train_count + test_count:]
                
                client_splits[client_id]['train'][label] = train_files
                client_splits[client_id]['test'][label] = test_files
                client_splits[client_id]['validate'][label] = validate_files
        
        return client_splits
    
    def _download_and_organize_images_by_client(self, client_files_info, qualifying_clients, 
                                                client_folder_mapping, debug_info):
        """Download images and organize them by client"""
        download_stats = {}
        
        # Split images for each client
        client_splits = self._split_images_by_client(client_files_info)
        
        total_files_to_download = 0
        for client_files in client_files_info.values():
            for files in client_files.values():
                total_files_to_download += len(files)
        
        self.safe_print(f"[DEBUG] Total files to download: {total_files_to_download}")
        
        downloaded_count = 0
        
        for client_idx, (client_id, client_folder) in enumerate(client_folder_mapping.items(), 1):
            if client_id not in client_files_info:
                self.safe_print(f"[DEBUG] WARNING: Client {client_id[:12]}... not in file info, skipping")
                continue
            
            client_splits_data = client_splits[client_id]
            download_stats[client_id] = {
                'train': {'healthy': 0, 'eczema': 0, 'failed': 0},
                'test': {'healthy': 0, 'eczema': 0, 'failed': 0},
                'validate': {'healthy': 0, 'eczema': 0, 'failed': 0}
            }
            
            total_client_files = sum(
                len(files) for split_files in client_splits_data.values() 
                for files in split_files.values()
            )
            
            self.safe_print(f"[DEBUG] Processing client {client_idx}/{len(client_folder_mapping)}: "
                           f"{client_folder} ({client_id[:12]}...) - {total_client_files} files")
            
            # Process each split for this client
            for split_name, split_files in client_splits_data.items():
                for label, files in split_files.items():
                    if not files:
                        continue
                    
                    split_dir = os.path.join(self.BASE_PATH, client_folder, split_name, label)
                    
                    self.safe_print(f"[DEBUG]   {split_name}/{label}: {len(files)} files")
                    
                    for i, file_info in enumerate(files):
                        file_id = file_info['file_id']
                        original_filename = file_info['original_filename'] or f"{label}_{i+1}.jpg"
                        
                        file_extension = os.path.splitext(original_filename)[1] or '.jpg'
                        safe_filename = f"{label}_{i+1:03d}{file_extension}"
                        save_path = os.path.join(split_dir, safe_filename)
                        
                        if self._download_image_from_drive(file_id, save_path):
                            download_stats[client_id][split_name][label] += 1
                            downloaded_count += 1
                            
                            if downloaded_count % 10 == 0:
                                progress = (downloaded_count / total_files_to_download) * 100
                                self.safe_print(f"[DEBUG] Progress: {downloaded_count}/{total_files_to_download} ({progress:.1f}%)")
                        else:
                            download_stats[client_id][split_name]['failed'] += 1
                            
                            debug_info['failed_downloads'].append({
                                'client_id': client_id[:12] + '...',
                                'client_folder': client_folder,
                                'file_id': file_id,
                                'filename': safe_filename,
                                'split': split_name,
                                'label': label
                            })
            
            # Log client completion
            client_total = sum(
                stats['healthy'] + stats['eczema'] 
                for stats in download_stats[client_id].values()
            )
            client_failed = sum(stats['failed'] for stats in download_stats[client_id].values())
            
            self.safe_print(f"[DEBUG]   {client_folder} complete: {client_total} success, {client_failed} failed")
        
        return download_stats
    
    def _download_image_from_drive(self, file_id, save_path):
        """Download a single image from Google Drive"""
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            with open(save_path, 'wb') as f:
                f.write(file_io.getvalue())
            
            return True
            
        except Exception as e:
            self.safe_print(f"[ERROR] Failed to download file {file_id}: {e}")
            return False
    
    def _create_summary_files(self, qualifying_clients, download_stats, client_folder_mapping):
        """Create summary and info files"""
        # Create detailed summary file
        summary_file = os.path.join(self.BASE_PATH, "dataset_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Federated Learning Dataset Summary\n")
            f.write(f"Generated: {datetime.datetime.now()}\n")
            f.write(f"Random Seed: {self.RANDOM_SEED}\n")
            f.write(f"Classes: {', '.join(self.CLASSES)}\n\n")
            
            f.write(f"Data Split Configuration:\n")
            f.write(f"Training: {self.TRAIN_RATIO*100}%\n")
            f.write(f"Test: {self.TEST_RATIO*100}%\n")
            f.write(f"Validation: {self.VALIDATION_RATIO*100}%\n\n")
            
            f.write(f"Client Distribution:\n")
            for client_id, client_folder in client_folder_mapping.items():
                if client_id in qualifying_clients and client_id in download_stats:
                    counts = qualifying_clients[client_id]
                    stats = download_stats[client_id]
                    
                    f.write(f"\n{client_folder} ({client_id}):\n")
                    f.write(f"  Original: {counts['healthy']} healthy, {counts['eczema']} eczema, {counts['total']} total\n")
                    
                    for split_name, split_stats in stats.items():
                        total_split = split_stats['healthy'] + split_stats['eczema']
                        f.write(f"  {split_name}: {total_split} images ({split_stats['healthy']} healthy, {split_stats['eczema']} eczema)\n")
        
        # Create dataset info JSON
        dataset_info = {
            "name": "Federated Eczema Classification Dataset",
            "classes": self.CLASSES,
            "class_to_idx": {class_name: idx for idx, class_name in enumerate(self.CLASSES)},
            "structure": "client_based",
            "clients": {},
            "total_clients": len(client_folder_mapping),
            "created": datetime.datetime.now().isoformat(),
            "random_seed": self.RANDOM_SEED
        }
        
        # Add client information
        for client_id, client_folder in client_folder_mapping.items():
            if client_id in download_stats:
                dataset_info["clients"][client_folder] = {
                    "splits": download_stats[client_id]
                }
        
        info_file = os.path.join(self.BASE_PATH, "dataset_info.json")
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Create client mapping file
        mapping_file = os.path.join(self.BASE_PATH, "client_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump(client_folder_mapping, f, indent=2)