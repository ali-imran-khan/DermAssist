import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm
import os
import copy
from collections import OrderedDict
from PIL import ImageFile
import json
import logging
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FederatedEczemaLearning:
    def __init__(self, 
                 client_data_dir="D:/uni/FYP/federation/client_data",
                 global_model_path="D:/uni/FYP/federation//models",
                 initial_model_path=None,
                 device=None):
        """
        Initialize Federated Learning pipeline for Eczema Detection
        
        Args:
            client_data_dir: Directory containing client folders
            global_model_path: Directory to save global models
            initial_model_path: Path to initial model (optional)
            device: Computing device
        """
        self.client_data_dir = client_data_dir
        self.global_model_path = global_model_path
        self.initial_model_path = initial_model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories
        os.makedirs(self.global_model_path, exist_ok=True)
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize global model
        self.global_model = self._initialize_global_model()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Client data directory: {self.client_data_dir}")
        
    def _initialize_global_model(self):
        """Initialize the global EfficientNet-B3 model"""
        model = create_model('efficientnet_b3', pretrained=True, num_classes=2)
        
        # Load initial model if provided
        if self.initial_model_path and os.path.exists(self.initial_model_path):
            logger.info(f"Loading initial model from {self.initial_model_path}")
            model.load_state_dict(torch.load(self.initial_model_path, map_location=self.device))
        
        model.to(self.device)
        return model
    
    def get_client_list(self):
        """Get list of available client directories"""
        if not os.path.exists(self.client_data_dir):
            logger.error(f"Client data directory {self.client_data_dir} does not exist")
            return []
        
        clients = [d for d in os.listdir(self.client_data_dir) 
                  if os.path.isdir(os.path.join(self.client_data_dir, d))]
        logger.info(f"Found {len(clients)} clients: {clients}")
        return sorted(clients)
    
    def load_client_data(self, client_id, batch_size=8, split='train'):
        """
        Load data for a specific client from train/test/validate split
        
        Args:
            client_id: Client identifier
            batch_size: Batch size for DataLoader
            split: Data split to load ('train', 'test', 'validate')
            
        Returns:
            DataLoader: Client's data loader
            int: Number of samples
        """
        client_path = os.path.join(self.client_data_dir, client_id)
        split_path = os.path.join(client_path, split)
        
        if not os.path.exists(split_path):
            logger.error(f"Client {split} directory {split_path} does not exist")
            return None, 0
        
        try:
            dataset = datasets.ImageFolder(split_path, transform=self.transform)
            shuffle = True if split == 'train' else False
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
            
            logger.info(f"Client {client_id} ({split}): {len(dataset)} samples, {len(dataset.classes)} classes")
            return dataloader, len(dataset)
            
        except Exception as e:
            logger.error(f"Error loading {split} data for client {client_id}: {str(e)}")
            return None, 0
    
    def train_local_model(self, model, dataloader, client_id, local_epochs=3, lr=1e-4):
        """
        Train model locally on client data
        
        Args:
            model: Model to train
            dataloader: Client's data loader
            client_id: Client identifier
            local_epochs: Number of local training epochs
            lr: Learning rate
            
        Returns:
            dict: Updated model state dict
            dict: Training metrics
        """
        model.train()
        # Get all labels from the dataset to compute class weights
        targets = []
        for _, label in dataloader.dataset.samples:
            targets.append(label)

        class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=np.array(targets)
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        logger.info(f"Training local model for client {client_id} - {local_epochs} epochs")
        
        for epoch in range(local_epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_samples = 0
            
            with tqdm(dataloader, desc=f"Client {client_id} - Epoch {epoch+1}/{local_epochs}") as pbar:
                for images, labels in pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate accuracy
                    _, preds = torch.max(outputs, 1)
                    batch_correct = (preds == labels).sum().item()
                    batch_samples = labels.size(0)
                    
                    epoch_loss += loss.item()
                    epoch_correct += batch_correct
                    epoch_samples += batch_samples
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{100 * batch_correct / batch_samples:.2f}%"
                    })
            
            epoch_acc = 100 * epoch_correct / epoch_samples
            logger.info(f"Client {client_id} - Epoch {epoch+1}: Loss={epoch_loss/len(dataloader):.4f}, Acc={epoch_acc:.2f}%")
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        # Calculate final metrics
        avg_loss = total_loss / (local_epochs * len(dataloader))
        avg_acc = 100 * total_correct / total_samples
        
        metrics = {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'samples': total_samples // local_epochs  # Average samples per epoch
        }
        
        logger.info(f"Client {client_id} training completed - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%")
        
        return model.state_dict(), metrics
    
    def evaluate_client_model(self, model, client_id, batch_size=8, split='test'):
        """
        Evaluate model on client's test or validation data
        
        Args:
            model: Model to evaluate
            client_id: Client identifier  
            batch_size: Batch size for evaluation
            split: Data split to evaluate on ('test' or 'validate')
            
        Returns:
            dict: Evaluation metrics
        """
        dataloader, sample_count = self.load_client_data(client_id, batch_size, split)
        
        if dataloader is None or sample_count == 0:
            logger.warning(f"No {split} data available for client {client_id}")
            return None
        
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Evaluating client {client_id} on {split}"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
        
        logger.info(f"Client {client_id} {split} evaluation - Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        return metrics
    
    def aggregate_weights(self, client_weights, sample_counts):
        """
        Perform FedAvg aggregation of client weights
        
        Args:
            client_weights: List of client state dictionaries
            sample_counts: List of sample counts per client
            
        Returns:
            dict: Aggregated model state dictionary
        """
        logger.info("Performing FedAvg aggregation...")
        
        total_samples = sum(sample_counts)
        logger.info(f"Total samples across all clients: {total_samples}")
        
        # Initialize aggregated weights with zeros
        aggregated_weights = OrderedDict()
        
        # Get the structure from the first client
        first_client_weights = client_weights[0]
        for key in first_client_weights.keys():
            aggregated_weights[key] = torch.zeros_like(first_client_weights[key], dtype=torch.float32)

        
        # Weighted averaging
        for client_idx, (weights, sample_count) in enumerate(zip(client_weights, sample_counts)):
            weight_ratio = sample_count / total_samples
            logger.info(f"Client {client_idx}: {sample_count} samples, weight ratio: {weight_ratio:.4f}")
            
            for key in weights.keys():
                aggregated_weights[key] += (weights[key].float() * weight_ratio)

        
        logger.info("FedAvg aggregation completed")
        return aggregated_weights
    
    def save_global_model(self, additional_info=None):
        """
        Save the global model as eczema_classifier_latest.pth
        
        Args:
            additional_info: Additional information to save
        """
        model_filename = "eczema_classifier_latest.pth"
        model_path = os.path.join(self.global_model_path, model_filename)
        
        torch.save(self.global_model.state_dict(), model_path)
        logger.info(f"Global model saved: {model_path}")
        
        # Save additional information
        if additional_info:
            info_filename = "training_info.json"
            info_path = os.path.join(self.global_model_path, info_filename)
            with open(info_path, 'w') as f:
                json.dump(additional_info, f, indent=2)
            logger.info(f"Training info saved: {info_path}")
    
    def evaluate_global_model(self, val_dataloader):
        """
        Evaluate the global model on validation data
        
        Args:
            val_dataloader: Validation data loader
            
        Returns:
            dict: Evaluation metrics
        """
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc="Evaluating global model"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(val_dataloader)
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
        
        logger.info(f"Global model evaluation - Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        return metrics
    
    def run_federated_training(self, clients=None, local_epochs=3, batch_size=8, evaluate_clients=True):
        """
        Run federated training (single iteration)
        
        Args:
            clients: List of client IDs (if None, use all available clients)
            local_epochs: Number of local training epochs
            batch_size: Batch size for training
            evaluate_clients: Whether to evaluate each client on their test data
            
        Returns:
            dict: Training results and metrics
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"STARTING FEDERATED TRAINING")
        logger.info(f"{'='*50}")
        
        if clients is None:
            clients = self.get_client_list()
        
        if not clients:
            logger.error("No clients available for training")
            return None
        
        client_weights = []
        sample_counts = []
        client_metrics = {}
        
        # Train on each client
        for client_id in clients:
            logger.info(f"\n--- Training on Client: {client_id} ---")
            
            # Load client training data
            dataloader, sample_count = self.load_client_data(client_id, batch_size, 'train')
            if dataloader is None or sample_count == 0:
                logger.warning(f"Skipping client {client_id} - no training data available")
                continue
            
            # Create a copy of global model for local training
            local_model = copy.deepcopy(self.global_model)
            
            # Train locally
            updated_weights, train_metrics = self.train_local_model(
                local_model, dataloader, client_id, local_epochs
            )
            
            client_weights.append(updated_weights)
            sample_counts.append(sample_count)
            
            # Initialize client metrics
            client_metrics[client_id] = {
                'train': train_metrics,
                'train_samples': sample_count
            }
            
            # Evaluate on client's test data if requested
            if evaluate_clients:
                # Load the trained model for evaluation
                local_model.load_state_dict(updated_weights)
                
                # Evaluate on test data
                test_metrics = self.evaluate_client_model(local_model, client_id, batch_size, 'test')
                if test_metrics:
                    client_metrics[client_id]['test'] = test_metrics
                
                # Evaluate on validation data if available
                val_metrics = self.evaluate_client_model(local_model, client_id, batch_size, 'validate')
                if val_metrics:
                    client_metrics[client_id]['validate'] = val_metrics
        
        if not client_weights:
            logger.error("No clients completed training successfully")
            return None
        
        # Aggregate weights using FedAvg
        logger.info(f"\n--- Aggregating weights from {len(client_weights)} clients ---")
        aggregated_weights = self.aggregate_weights(client_weights, sample_counts)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_weights)
        
        # Prepare training information
        training_info = {
            'participating_clients': len(client_weights),
            'total_samples': sum(sample_counts),
            'client_metrics': client_metrics,
            'sample_distribution': dict(zip(clients[:len(sample_counts)], sample_counts))
        }
        
        # Save global model
        self.save_global_model(training_info)
        
        logger.info(f"\nFederated training completed successfully!")
        logger.info(f"Participating clients: {len(client_weights)}")
        logger.info(f"Total samples: {sum(sample_counts)}")
        
        return training_info
    
    def run_federated_learning(self, local_epochs=3, batch_size=8, validation_data_path=None, evaluate_clients=True):
        """
        Run complete federated learning process (single training iteration)
        
        Args:
            local_epochs: Number of local training epochs
            batch_size: Batch size for training
            validation_data_path: Path to global validation data (optional)
            evaluate_clients: Whether to evaluate each client on their test/validate data
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING FEDERATED LEARNING")
        logger.info(f"{'='*60}")
        
        # Load global validation data if provided
        val_dataloader = None
        if validation_data_path and os.path.exists(validation_data_path):
            val_dataset = datasets.ImageFolder(validation_data_path, transform=self.transform)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            logger.info(f"Global validation dataset loaded: {len(val_dataset)} samples")
        
        # Run federated training
        training_results = self.run_federated_training(
            local_epochs=local_epochs, 
            batch_size=batch_size,
            evaluate_clients=evaluate_clients
        )
        
        if training_results is None:
            logger.error("Federated training failed")
            return None
        
        # Evaluate global model on global validation data if available
        if val_dataloader:
            val_metrics = self.evaluate_global_model(val_dataloader)
            training_results['global_validation_metrics'] = val_metrics
        
        # Print training summary
        logger.info(f"\n--- Training Summary ---")
        logger.info(f"Clients participated: {training_results['participating_clients']}")
        logger.info(f"Total samples: {training_results['total_samples']}")
        
        # Print client-specific results
        for client_id, metrics in training_results['client_metrics'].items():
            logger.info(f"\nClient {client_id}:")
            logger.info(f"  Training - Accuracy: {metrics['train']['accuracy']:.2f}%, Loss: {metrics['train']['loss']:.4f}")
            if 'test' in metrics:
                logger.info(f"  Test - Accuracy: {metrics['test']['accuracy']:.2f}%, Loss: {metrics['test']['loss']:.4f}")
            if 'validate' in metrics:
                logger.info(f"  Validate - Accuracy: {metrics['validate']['accuracy']:.2f}%, Loss: {metrics['validate']['loss']:.4f}")
        
        if 'global_validation_metrics' in training_results:
            val_acc = training_results['global_validation_metrics']['accuracy']
            logger.info(f"\nGlobal model validation accuracy: {val_acc:.2f}%")
        
        # Save final results
        final_results_path = os.path.join(self.global_model_path, "federated_learning_results.json")
        with open(final_results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FEDERATED LEARNING COMPLETED")
        logger.info(f"Model saved as: eczema_classifier_latest.pth")
        logger.info(f"Results saved to: {final_results_path}")
        logger.info(f"{'='*60}")
        
        return training_results


def main():
    """
    Main function to run federated learning with split data
    Expected client data structure:
    client_data/
    ├── client_001/
    │   ├── train/
    │   │   ├── eczema/
    │   │   └── normal/
    │   ├── test/
    │   │   ├── eczema/
    │   │   └── normal/
    │   └── validate/
    │       ├── eczema/
    │       └── normal/
    ├── client_002/
    │   ├── train/
    │   ├── test/
    │   └── validate/
    └── ...
    """
    # Configuration
    CLIENT_DATA_DIR = "D:/uni/FYP/federation/client_data"  # Directory containing client_001, client_002, etc.
    GLOBAL_MODEL_DIR = "D:/uni/FYP/federation/models"      # Directory to save global models
    VALIDATION_DATA_PATH = "./validation_data"  # Optional global validation data
    INITIAL_MODEL_PATH = "D:/uni/FYP/federation/models/eczema_classifier_latest.pth"  # Your existing model
    
    # Federated Learning Parameters
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 8
    EVALUATE_CLIENTS = True  # Whether to evaluate each client on their test/validate data
    
    # Initialize federated learning
    fed_learning = FederatedEczemaLearning(
        client_data_dir=CLIENT_DATA_DIR,
        global_model_path=GLOBAL_MODEL_DIR,
        initial_model_path=INITIAL_MODEL_PATH if os.path.exists(INITIAL_MODEL_PATH) else None
    )
    
    # Run federated learning
    results = fed_learning.run_federated_learning(
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data_path=VALIDATION_DATA_PATH if os.path.exists(VALIDATION_DATA_PATH) else None,
        evaluate_clients=EVALUATE_CLIENTS
    )
    
    print("\nFederated Learning Process Completed!")
    print(f"Model saved as: eczema_classifier_latest.pth")
    print(f"Check the '{GLOBAL_MODEL_DIR}' directory for saved models and results.")


