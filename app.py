from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import uvicorn

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model("model/saved_model")
MAX_LEN = 100

app = FastAPI(title="Medical Text Classifier")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: TextInput):
    text = input_data.text
    sequences = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN)
    predictions = model.predict(padded)
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    return {
        "input": text,
        "predicted_class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
