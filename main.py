import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Configure CORS settings to allow access from specified origins
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "HEAD"],
    allow_headers=["*"],
)

# Load potato model with compile=False
POTATO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "potatoes.h5")
POTATO_MODEL = tf.keras.models.load_model(POTATO_MODEL_PATH, compile=False)

POTATO_CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# Load tomato model with compile=False
TOMATO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "tomatoes.h5")
TOMATO_MODEL = tf.keras.models.load_model(TOMATO_MODEL_PATH, compile=False)

TOMATO_CLASS_NAMES = [
    "Bacterial-spot", "Early-blight", "Healthy", "Late-blight",
    "Leaf-mold", "Mosaic-virus", "Septoria-leaf-spot", "Yellow-leaf-curl-virus"
]

# Endpoint for a ping test to ensure the server is running
@app.head("/ping")
async def ping():
    return "Hello, I am alive"

# Function to read uploaded file as an image and convert to numpy array
def read_file_as_image(data) -> np.ndarray:
    # Open image using PIL
    image = Image.open(BytesIO(data))
    # Resize image to expected dimensions (256x256 pixels)
    image = image.resize((256, 256))
    # Convert image to numpy array
    image_array = np.array(image)
    # Normalize values between 0 and 1
    image_array = image_array.astype(np.float32) / 255.0
    return image_array

# Endpoint to predict potato type based on uploaded image
@app.post("/predict/potato")
async def predict_potato(
    file: UploadFile = File(...)
):
    # Read and preprocess uploaded image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # Make predictions using loaded model
    predictions = POTATO_MODEL.predict(img_batch)

    # Get predicted class and confidence level
    predicted_class = POTATO_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return {
        'class': predicted_class,
        'confidence': confidence
    }

# Endpoint to predict tomato type based on uploaded image
@app.post("/predict/tomato")
async def predict_tomato(
    file: UploadFile = File(...)
):
    # Read and preprocess uploaded image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # Make predictions using loaded model
    predictions = TOMATO_MODEL.predict(img_batch)

    # Get predicted class and confidence level
    predicted_class = TOMATO_CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == "__main__":
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
