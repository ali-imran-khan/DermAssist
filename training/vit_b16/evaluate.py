import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ViT-B/16 model
model = create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
model.load_state_dict(torch.load("eczema_classifier_vit_b16_close.pth", map_location=device))
model.to(device)
model.eval()

# Define transform for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load test dataset
test_dir = "data/dermo/Test"
test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Class labels
class_names = test_data.classes

# Metrics
y_true = []
y_pred = []
y_scores = []

print("\nTesting ViT model...\n")
for images, labels in tqdm(test_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)

    probs = torch.softmax(outputs, dim=1)
    _, predicted = torch.max(probs, 1)

    y_true.append(labels.item())
    y_pred.append(predicted.item())
    y_scores.append(probs[0][1].item())  # Probability of class 1

    print(f"Predicted: {class_names[predicted.item()]} | Actual: {class_names[labels.item()]}")

# Convert to NumPy arrays
y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)
y_scores_np = np.array(y_scores)

# Metrics calculation
acc = 100 * (y_pred_np == y_true_np).sum() / len(y_true_np)
precision = precision_score(y_true_np, y_pred_np)
recall = recall_score(y_true_np, y_pred_np)
f1 = f1_score(y_true_np, y_pred_np)
conf_matrix = confusion_matrix(y_true_np, y_pred_np)
auc = roc_auc_score(y_true_np, y_scores_np)

# Display results
print("\ndone")
print(f"Accuracy:  {acc:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_np, y_pred_np, target_names=class_names))

# Confusion Matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
if len(class_names) == 2:
    fpr, tpr, _ = roc_curve(y_true_np, y_scores_np)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
else:
    print("AUC and ROC Curve are applicable only for binary classification.")
