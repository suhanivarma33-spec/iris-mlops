"""FastAPI service for Iris species predictions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import load_model, SPECIES

app = FastAPI(title="Iris Flower Classifier", version="1.0.0", description="Predict Iris species from four measurements.")
_model = None

class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, description="Sepal width in cm")
    petal_length: float = Field(gt=0, description="Petal length in cm")
    petal_width: float = Field(gt=0, description="Petal width in cm")

@app.get("/")
def home():
    return {"message": "Iris Flower Classification API", "docs": "/docs", "species": SPECIES}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: FlowerInput):
    global _model
    try:
        if _model is None:
            _model = load_model()
        measurements = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
        species = SPECIES[int(_model.predict([measurements])[0])]
        probabilities = _model.predict_proba([measurements])[0].tolist()
        return {"predicted_species": species, "probabilities": dict(zip(SPECIES, probabilities))}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
