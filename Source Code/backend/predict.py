import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# -------------------------------
# Load class names
# -------------------------------
data_dir = "data/PlantVillage-Dataset/raw/color"
class_names = sorted(os.listdir(data_dir))

treatments = {
    "Apple___Apple_scab": "Use fungicides and remove infected leaves.",
    "Apple___Black_rot": "Prune infected branches and apply fungicide.",
    "Apple___Cedar_apple_rust": "Apply fungicides and remove nearby cedar trees.",
    "Apple___healthy": "No treatment needed. Plant is healthy.",

    "Blueberry___healthy": "No treatment needed.",

    "Cherry_(including_sour)___Powdery_mildew": "Use sulfur-based fungicides and improve airflow.",
    "Cherry_(including_sour)___healthy": "No treatment needed.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant hybrids and fungicides.",
    "Corn_(maize)___Common_rust": "Apply fungicides and plant resistant varieties.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use crop rotation and fungicides.",
    "Corn_(maize)___healthy": "No treatment needed.",

    "Grape___Black_rot": "Remove infected leaves and apply fungicide.",
    "Grape___Esca_(Black_Measles)": "Prune infected vines and avoid water stress.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Use fungicides and remove infected leaves.",
    "Grape___healthy": "No treatment needed.",

    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees and control insect vectors.",

    "Peach___Bacterial_spot": "Use copper sprays and resistant varieties.",
    "Peach___healthy": "No treatment needed.",

    "Pepper,_bell___Bacterial_spot": "Use copper-based sprays and avoid overhead watering.",
    "Pepper,_bell___healthy": "No treatment needed.",

    "Potato___Early_blight": "Use fungicides and crop rotation.",
    "Potato___Late_blight": "Apply fungicides and remove infected plants.",
    "Potato___healthy": "No treatment needed.",

    "Strawberry___Leaf_scorch": "Remove infected leaves and apply fungicides.",
    "Strawberry___healthy": "No treatment needed.",

    "Tomato___Bacterial_spot": "Use copper sprays and avoid leaf wetness.",
    "Tomato___Early_blight": "Use fungicides and remove infected leaves.",
    "Tomato___Late_blight": "Apply fungicides and destroy infected plants.",
    "Tomato___Leaf_Mold": "Improve ventilation and apply fungicides.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves and apply fungicides.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use miticides and increase humidity.",
    "Tomato___Target_Spot": "Apply fungicides and remove infected leaves.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies and remove infected plants.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants and disinfect tools.",
    "Tomato___healthy": "No treatment needed."
}
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 38)

    model.load_state_dict(torch.load("models/model_epoch_5.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------
# Image transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------
# Prediction function
# -------------------------------
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    confidence = confidence.item()

    # Raw class
    raw_class = class_names[predicted.item()]

    # Clean output
    plant, disease = raw_class.split("___")
    disease = disease.replace("_", " ")
    predicted_class = f"{plant} - {disease}"

    # Confidence filter (OOD handling)
    if confidence < 0.5:
        return {
            "prediction": "Not a valid leaf image",
            "confidence": round(confidence * 100, 2),
            "treatment": "N/A"
        }

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "treatment": treatments.get(raw_class, "No treatment data available")
    }