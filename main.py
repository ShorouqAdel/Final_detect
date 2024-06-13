from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gdown

app = FastAPI()

# Google Drive file ID (extracted from your link)
file_id = "1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"
file_url = f"https://drive.google.com/uc?id={file_id}"
model_path = "plant_disease_prediction_model.h5"

# Check if the model file exists; if not, download it
if not os.path.exists(model_path):
    print(f"Downloading the model from Google Drive...")
    gdown.download(file_url, model_path, quiet=False)

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError(f"Error loading model: {e}")

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def predict_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = tf.image.resize_with_pad(np.array(image), target_height=224, target_width=224)
        
        input_arr = image / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)
        
        predictions = model.predict(input_arr)
        return np.argmax(predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image.")
    
    try:
        image_data = await file.read()
        predicted_index = predict_image(image_data)
        predicted_class = class_names[predicted_index]
        
        return JSONResponse(content={"prediction": predicted_class})
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Plant Disease Detection API! Use /predict to analyze plant images."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
