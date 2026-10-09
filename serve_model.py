from pathlib import Path

import mlflow
import mlflow.pyfunc
import uvicorn


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MLFLOW TRACKING DATABASE
# ============================================================

MLFLOW_DB = BASE_DIR / "mlflow.db"

mlflow.set_tracking_uri(
    "sqlite:///" + str(MLFLOW_DB).replace("\\", "/")
)


# ============================================================
# MODEL RUN
# ============================================================

RUN_ID = "f1a5da07938b4ca99fba9b635ba0f2ce"

MODEL_URI = f"runs:/{RUN_ID}/house_price_model"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading MLflow model...")
print("Model URI:", MODEL_URI)

model = mlflow.pyfunc.load_model(
    MODEL_URI
)

print("Model loaded successfully!")


# ============================================================
# CREATE MLflow SCORING SERVER
# ============================================================

from mlflow.pyfunc.scoring_server import init

app = init(model)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\nStarting MLflow model server...")
    print("URL: http://127.0.0.1:5001")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5001
    )