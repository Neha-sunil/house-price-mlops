from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "house_prices.csv"
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# 2. MLflow CONFIGURATION
# ============================================================

mlflow.set_tracking_uri(
    f"sqlite:///{BASE_DIR / 'mlflow.db'}"
)

mlflow.set_experiment("House Price Prediction")


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("price", axis=1)

y = df["price"]


# ============================================================
# 5. DEFINE FEATURES
# ============================================================

numerical_features = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
    "age"
]

categorical_features = [
    "location"
]


# ============================================================
# 6. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 7. DEFINE XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42
)


# ============================================================
# 8. CREATE COMPLETE ML PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nPreprocessing completed")
print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# ============================================================
# 10. TRAIN MODEL WITH MLFLOW TRACKING
# ============================================================

print("\nTraining XGBoost model...")


with mlflow.start_run():

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )

    print("Model training completed!")


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = pipeline.predict(X_test)


    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # Display metrics
    # --------------------------------------------------------

    print("\nModel Evaluation")
    print("----------------")

    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.4f}")


    # ========================================================
    # 11. LOG PARAMETERS TO MLFLOW
    # ========================================================

    mlflow.log_param(
        "model",
        "XGBoost Regressor"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "max_depth",
        3
    )

    mlflow.log_param(
        "learning_rate",
        0.05
    )

    mlflow.log_param(
        "test_size",
        0.20
    )


    # ========================================================
    # 12. LOG METRICS
    # ========================================================

    mlflow.log_metric(
        "mae",
        mae
    )

    mlflow.log_metric(
        "rmse",
        rmse
    )

    mlflow.log_metric(
        "r2",
        r2
    )


    # ========================================================
    # 13. MODEL VALIDATION
    # ========================================================

    R2_THRESHOLD = 0.50

    mlflow.log_param(
        "r2_threshold",
        R2_THRESHOLD
    )


    print("\nModel Validation")
    print("----------------")

    print(
        f"Required R²: {R2_THRESHOLD:.2f}"
    )

    print(
        f"Actual R²  : {r2:.4f}"
    )


    # ========================================================
    # 14. VALIDATION CHECK
    # ========================================================

    if r2 >= R2_THRESHOLD:

        print("VALIDATION PASSED")
        print("Model is approved.")


        # ----------------------------------------------------
        # MLflow validation tag
        # ----------------------------------------------------

        mlflow.set_tag(
            "validation_status",
            "PASSED"
        )


        # ----------------------------------------------------
        # Create models directory
        # ----------------------------------------------------

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Save XGBoost model
        # ----------------------------------------------------

        model_path = (
            MODELS_DIR /
            "xgboost_house_price_model.json"
        )


        # Extract XGBoost model
        xgb_model = pipeline.named_steps["model"]


        xgb_model.save_model(
            model_path
        )


        print(
            f"Model saved to: {model_path}"
        )


        # ====================================================
        # 15. CREATE MLFLOW MODEL SIGNATURE
        # ====================================================

        signature = mlflow.models.infer_signature(
            X_train,
            pipeline.predict(X_train)
        )


        # ====================================================
        # 16. LOG COMPLETE PIPELINE TO MLFLOW
        # ====================================================

        mlflow.sklearn.log_model(
            pipeline,
            name="house_price_model",
            signature=signature,
            input_example=X_train.iloc[[0]],
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBRegressor"
            ]
        )


        print(
            "Complete preprocessing + model pipeline "
            "logged to MLflow."
        )


    else:

        print("VALIDATION FAILED")
        print("Model is rejected.")


        mlflow.set_tag(
            "validation_status",
            "FAILED"
        )


        raise ValueError(
            f"Model validation failed. "
            f"R²={r2:.4f} is below "
            f"required threshold={R2_THRESHOLD:.2f}"
        )


# ============================================================
# 17. COMPLETION MESSAGE
# ============================================================

print("\nMLflow run completed successfully.")