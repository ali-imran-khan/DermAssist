import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from timm import create_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay, classification_report
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset directory (change if needed)
test_dir = "data/phase1_subset/Test"

# Data transformations (Must match training transformations)
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Load test dataset
test_dataset = ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
print(f"Testing dataset loaded with {len(test_dataset)} images.")

# Load the saved model
model_path = "efficientnet_binary_classification.pth"  # Change if needed
model = create_model('efficientnet_b0', pretrained=True, num_classes=len(test_dataset.classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully.")

# Evaluate the model
all_preds = []
all_labels = []
all_probs = []  # To store class probabilities for AUC

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Convert logits to probabilities
        probs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

print("Evaluation complete.")

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# AUC & ROC Curve (For Binary Classification)
if len(test_dataset.classes) == 2:
    y_true = np.array([1 if label == 1 else 0 for label in all_labels])  # Convert to binary labels
    y_probs = np.array([prob[1] for prob in all_probs])  # Get probability of positive class

    auc_score = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)

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
