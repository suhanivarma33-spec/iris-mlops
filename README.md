
# Iris Flower Classification  using MLOps .
2. Open Terminal > New folder
open the `Irs-MLOps` folder in VS Code

A beginner-friendly project that classifies Iris flowers into Setosa, Versicolor, or Virginica using a Random Forest model and an MLOps toolchain.

## Requirements
- Windows 10/11 (or Linux/macOS), Python 3.11, Git, VS Code
- Docker Desktop (optional, for container execution)

## Setup on Windows (PowerShell)
1. Extract the ZIP a Terminal (PowerShell).
3. Create and activate a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in this terminal and activate again.
4. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Generate the CSV dataset:
   ```powershell
   python src/data_preprocessing.py
   ```
   
6. Train, evaluate, save the model and log the run to MLflow:
   ```powershell
   python src/train.py
   ```
   The trained model is saved under `models/iris_model.joblib`, metrics in `metrics.json`, and MLflow runs locally in `mlruns/`.
7. Run tests:
   ```powershell
   pytest -v
   ```
8. Launch MLflow's local tracking UI (in another terminal):
   ```powershell
   mlflow ui
   ```
   Open http://127.0.0.1:5000.
9. Launch the API:
   ```powershell
   uvicorn app.main:app --reload
   ```
   Open http://127.0.0.1:8000/docs to use the interactive API.

## Example API request
POST `/predict` with JSON:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## Git version control
```powershell
git init
git add .
git commit -m "Initial Iris MLOps project"
```

## DVC data versioning
Initialize DVC locally:
```powershell
dvc init
dvc add data/iris.csv
git add data/iris.csv.dvc .dvc/config .gitignore
git commit -m "Track Iris dataset with DVC"
```
For a remote DVC store, configure a remote directory/cloud storage using the DVC documentation. No remote is configured in this starter project.

## DVC pipeline
The included `dvc.yaml` describes dataset preparation and model training stages. After DVC is initialized, run:
```powershell
dvc repro
```

## Docker
With Docker Desktop installed and running:
```powershell
docker build -t iris-mlops .
docker run -p 8000:8000 iris-mlops
```
Then open http://127.0.0.1:8000/docs.

## GitHub Actions
The workflow in `.github/workflows/ci.yml` installs dependencies, generates data, trains the model, and runs pytest on pushes and pull requests.

## Project layout
```
Iris-MLOps/
├── data/                 # CSV dataset (generated)
├── models/               # serialized model
├── src/                  # preprocessing, training, prediction
├── tests/                # automated tests
├── app/                  # FastAPI application
├── .github/workflows/     # GitHub Actions workflow
├── dvc.yaml              # DVC pipeline
├── Dockerfile
└── requirements.txt
```
# iris-mlops
End-to-end Iris MLOps Project

