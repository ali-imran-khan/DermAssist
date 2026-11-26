import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

train_dir = "data/dermo/Train"
val_dir = "data/dermo/Validation"

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

# Model setup
model = create_model('efficientnet_b3', pretrained=True, num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(output, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    print(f"\nValidation phase:")
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    print(f"Validation Accuracy: {100 * val_correct / val_total:.2f}%\n")

# Save model
torch.save(model.state_dict(), "eczema_classifier_efficientnet_b3_close_large.pth")
print("Model saved successfully.")
