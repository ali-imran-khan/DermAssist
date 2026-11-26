import torch
from torchvision import transforms
from timm import create_model
from PIL import Image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (adjust if needed)
classes = ['Eczema', 'Healthy']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

def predict_image(image_path, model_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Load model
    model = create_model('efficientnet_b3', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    # Display results
    print("Prediction Confidence:")
    for idx, cls in enumerate(classes):
        print(f"{cls}: {probs[idx]*100:.2f}%")
    print(f"\nFinal Prediction: {classes[probs.argmax()]}")

if __name__ == "__main__":
    predict_image("eczema1.png", "eczema_classifier_efficientnet_b3.pth")
