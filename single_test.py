import torch
from torchvision import transforms
from PIL import Image
from timm import create_model
import io
import numpy as np

# Load model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model('efficientnet_b3', pretrained=False, num_classes=2)
model_path = 'D:/uni/FYP/federation/models/eczema_classifier_latest.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])
classes = ['Eczema', 'Healthy']




def classify_image(image_bytes):
    """Classify image and return label + confidence scores"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    prediction = {
        "Eczema": round(probs[0] * 100, 2),
        "Healthy": round(probs[1] * 100, 2),
        "Predicted_Class": classes[np.argmax(probs)]
    }
    return prediction
