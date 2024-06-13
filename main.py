import os
from fastapi import FastAPI
from tensorflow.keras.models import load_model
import gdown

app = FastAPI()

# رابط الملف على Google Drive
google_drive_url = 'https://drive.google.com/uc?id=1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf'
model_path = 'model/model.h5'

# وظيفة لتحميل النموذج من Google Drive
def download_model():
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        gdown.download(google_drive_url, model_path, quiet=False)
        print(f'Model downloaded to {model_path}')
    else:
        print(f'Model already exists at {model_path}')

# تحميل النموذج
download_model()

# تحميل النموذج باستخدام TensorFlow
model = load_model(model_path)

@app.get("/")
def read_root():
    return {"message": "Model is loaded and ready to use"}

# وظيفة لاستخدام النموذج - مثال بسيط
@app.get("/predict")
def predict():
    # هنا يمكنك وضع الشيفرة الخاصة بالتنبؤ باستخدام النموذج
    # على سبيل المثال، استخدام النموذج لتنبؤ بيانات معينة
    # input_data = ...

    # مثال فقط، لاستخدام النموذج لتنبؤ عشوائي
    # result = model.predict(input_data)
    return {"prediction": "This is a placeholder for the actual prediction"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
