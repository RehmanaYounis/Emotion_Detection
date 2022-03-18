import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
endpoint = "http://localhost:8651/v1/models/emotion_model:predict"
model = tf.keras.models.load_model("../saved_model/1")

classes = [  'anger',
             'contempt',
             'disgust',
             'fear',
             'happiness',
             'neutral',
             'sadness',
             'surprise']

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    image = np.stack((image,) * 3, axis=-1)

    return image


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = response.json()["predictions"][0]
    predicted_class = classes[np.argmax(prediction)]
    confidence = (np.max(prediction))
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

    # predictions = model.predict(img_batch)
    #
    # predicted_class = classes[np.argmax(predictions[0])]
    # confidence = np.max(predictions[0])
    # return {
    #     'class': predicted_class,
    #     'confidence': float(confidence)
    # }




if __name__ =="__main__":
    uvicorn.run(app, host='localhost', port=8000)
