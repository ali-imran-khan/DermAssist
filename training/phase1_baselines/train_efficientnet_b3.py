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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay



# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset directories
train_dir = "data/phase1_subset/Train"
test_dir = "data/phase1_subset/Test"
print("Dataset directories set.")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # EfficientNet-B3 input size
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

# DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
print("DataLoaders set for training and testing.")

# Define the EfficientNet-B3 model
model = create_model('efficientnet_b3', pretrained=True, num_classes=len(train_dataset.classes))
model = model.to(device)
print("EfficientNet-B3 model initialized and moved to device.")

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
all_probs = [] 

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Use logits for binary classification
        probs = torch.softmax(outputs, dim=1)  # Probabilities for each class
        predicted = torch.argmax(probs, dim=1)  # Class with the highest probability

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())  # Store probabilities for AUC

print("Evaluation complete.")

# Print classification metrics
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# AUC and ROC Curve (Assumes binary classification)
if len(train_dataset.classes) == 2:
    y_true = [1 if label == 1 else 0 for label in all_labels]  # Convert labels to binary
    y_probs = [prob[1] for prob in all_probs]  # Get probabilities for the positive class

    auc_score = roc_auc_score(y_true, y_probs)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    print(f"AUC Score: {auc_score:.4f}")

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
else:
    print("AUC and ROC Curve calculations are only applicable for binary classification.")

# Save the model
model_path = "efficientnet_b3_binary_classification.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
