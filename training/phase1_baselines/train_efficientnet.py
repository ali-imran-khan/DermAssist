import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from timm import create_model
from sklearn.metrics import classification_report
from collections import Counter
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset directories
train_dir = "data/phase1_subset/Train"
test_dir = "data/phase1_subset/Test"
print("Dataset directories set.")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
print("Data transformations defined.")

# Load datasets
train_dataset = ImageFolder(root=train_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)
print(f"Training dataset loaded with {len(train_dataset)} images.")
print(f"Testing dataset loaded with {len(test_dataset)} images.")

# Print class-to-label mapping
print("Classes:", train_dataset.classes)
print("Class-to-index mapping:", train_dataset.class_to_idx)

# Count the occurrences of each label
all_labels = [label for _, label in train_dataset]
class_counts = Counter(all_labels)
print("Class counts in the training dataset:", class_counts)

# Convert class counts to tensor
class_counts_tensor = torch.tensor([class_counts[i] for i in range(len(train_dataset.classes))], dtype=torch.float)
print("Class counts tensor:", class_counts_tensor)

# Calculate class weights as the inverse of class frequency
class_weights = class_counts_tensor.sum() / class_counts_tensor
print("Class weights calculated:", class_weights)

# Move class weights to the device
class_weights = class_weights.to(device)

# Verify DataLoader output
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
print("DataLoader initialized for training dataset.")
for images, labels in train_loader:
    print("First batch - Batch labels:", labels)
    print("First batch - Label counts:", torch.bincount(labels))
    break

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
print("DataLoaders set for training and testing.")

# Define the EfficientNet model
model = create_model('efficientnet_b0', pretrained=True, num_classes=2)
model = model.to(device)
print("EfficientNet model initialized and moved to device.")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
print("Loss function and optimizer defined.")

# Training loop
num_epochs = 5
print("Starting training loop...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"Epoch {epoch + 1}/{num_epochs} started.")
    
    # Use tqdm to track the batch progress
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Track performance
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total * 100
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("Training complete.")

# Evaluation
print("Starting evaluation...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Use logits for binary classification
        probs = torch.sigmoid(outputs[:, 1])  # Probabilities for the positive class
        predicted = (probs > 0.5).int()       # Threshold at 0.5

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Evaluation complete.")

# Print classification metrics
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Save the model
model_path = "efficientnet_binary_classification.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
