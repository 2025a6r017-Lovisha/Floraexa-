
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import os
import tensorflow as tf

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load the trained CNN model
MODEL_PATH = "models/plant_disease_efficientnet.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("First layer:", model.layers[0].__class__.__name__)
print("Preprocessing layers:")
for layer in model.layers[:5]:
    print(layer.__class__.__name__, layer.get_config().get("scale", ""))
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# Load class names
CLASS_NAMES_PATH = "models/class_names.txt"

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [
        line.strip().split(": ", 1)[1]
        for line in f
        if ": " in line
    ]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        return HTMLResponse(
            "Please upload a valid image.",
            status_code=400
        )

    try:
        image = Image.open(file.file).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]

        print("Model output shape:", model.output_shape)
        print("Number of classes:", len(class_names))

        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index]) * 100

        if predicted_index < len(class_names):
            disease = class_names[predicted_index]
        else:
            disease = f"Class {predicted_index}"

        print("Predicted index:", predicted_index)
        print("Predicted label:", disease)
        print("Confidence:", round(confidence, 2), "%")

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "disease": disease,
                "confidence": round(confidence, 2)
            }
        )

    except Exception as e:
        print("Prediction error:", e)
        return HTMLResponse(
            f"Could not process the image: {e}",
            status_code=500
        )

    finally:
        file.file.close()
