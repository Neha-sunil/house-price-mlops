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
# GREEN MODEL
# ============================================================

RUN_ID = "f1a5da07938b4ca99fba9b635ba0f2ce"

MODEL_URI = f"runs:/{RUN_ID}/house_price_model"


print("Loading GREEN deployment model...")
print("Model URI:", MODEL_URI)


model = mlflow.pyfunc.load_model(
    MODEL_URI
)

print("GREEN model loaded successfully!")


# ============================================================
# CREATE API
# ============================================================

from mlflow.pyfunc.scoring_server import init

app = init(model)


# ============================================================
# START GREEN SERVER
# ============================================================

if __name__ == "__main__":

    print("\nStarting GREEN deployment...")
    print("GREEN URL: http://127.0.0.1:5002")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5002
    )