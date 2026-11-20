"""Load the trained model and make predictions."""
from pathlib import Path
import joblib
import numpy as np
from src.train import MODEL_PATH

SPECIES = ["setosa", "versicolor", "virginica"]


def load_model():
    if not MODEL_PATH.exists():
        from src.train import train_model
        train_model()
    return joblib.load(MODEL_PATH)


def predict_species(measurements, model=None):
    model = model or load_model()
    values = np.asarray(measurements, dtype=float).reshape(1, -1)
    if values.shape[1] != 4:
        raise ValueError("Exactly four flower measurements are required.")
    prediction = int(model.predict(values)[0])
    return SPECIES[prediction]
