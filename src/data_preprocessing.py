"""Load and prepare the Iris dataset."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def save_dataset(path: Path = DATA_DIR / "iris.csv") -> Path:
    """Save the bundled sklearn Iris dataset to CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_iris(as_frame=True)
    df = dataset.frame.copy()
    df.columns = FEATURES + ["target"]
    df["species"] = df["target"].map(dict(enumerate(dataset.target_names)))
    df.to_csv(path, index=False)
    return path


def load_dataset(path: Path = DATA_DIR / "iris.csv") -> pd.DataFrame:
    """Load CSV, creating it from sklearn if it is not present."""
    if not path.exists():
        save_dataset(path)
    return pd.read_csv(path)


if __name__ == "__main__":
    output = save_dataset()
    print(f"Dataset saved: {output}")
