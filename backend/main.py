from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import io
import os

from backend.predict import predict_image

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend UI
@app.get("/")
def home():
    return FileResponse("frontend/index.html")


# Disease treatment info
disease_info = {
    "Tomato___Early_blight": "Use fungicide and remove infected leaves.",
    "Tomato___Late_blight": "Avoid excess moisture and apply copper fungicide.",
    "Potato___Early_blight": "Use certified seeds and crop rotation.",
    "Apple___Black_rot": "Prune infected branches and apply fungicide.",
}

# Prediction route
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = predict_image(image)

    prediction = result.get("prediction", "Unknown")
    confidence = result.get("confidence", 0)

    treatment = disease_info.get(
        prediction,
        "Maintain proper plant care and consult an agricultural expert."
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "treatment": treatment
    }