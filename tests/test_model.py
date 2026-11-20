"""Basic tests for the project."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_preprocessing import load_dataset, FEATURES
from src.predict import predict_species, SPECIES


def test_dataset_shape_and_columns():
    df = load_dataset()
    assert len(df) == 150
    assert all(name in df.columns for name in FEATURES)
    assert "target" in df.columns


def test_prediction_label():
    # Typical Iris setosa measurements; prediction is checked for valid output.
    label = predict_species([5.1, 3.5, 1.4, 0.2])
    assert label in SPECIES


def test_species_count():
    assert len(SPECIES) == 3
