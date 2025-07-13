from fastapi import FastAPI
from pydantic import BaseModel
from app.preprocess import preprocess_text
from app.model import predict_category

app = FastAPI(title="Medical Text Classification API")

class MedicalTextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Medical Text Classification API is running."}

@app.post("/predict")
def predict_text(input_data: MedicalTextInput):
    category, confidence = predict_category(input_data.text, preprocess_text)
    return {
        "input_text": input_data.text,
        "predicted_category": category,
        "confidence_score": confidence
    }

