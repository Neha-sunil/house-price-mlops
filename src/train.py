from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "house_prices.csv"


# MLflow configuration
mlflow.set_tracking_uri(
    f"sqlite:///{BASE_DIR / 'mlflow.db'}"
)

mlflow.set_experiment("House Price Prediction")


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# Features and target
X = df.drop("price", axis=1)
y = df["price"]


# Feature types
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


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


print("\nPreprocessing completed")
print("Training shape:", X_train_processed.shape)
print("Testing shape :", X_test_processed.shape)


# Model
model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42
)


# Validation threshold
R2_THRESHOLD = 0.50


# Start MLflow run
with mlflow.start_run():

    print("\nTraining XGBoost model...")
    model.fit(X_train_processed, y_train)
    print("Model training completed!")


    # Predictions
    y_pred = model.predict(X_test_processed)


    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)


    print("\nModel Evaluation")
    print("----------------")
    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.4f}")


    # Log parameters
    mlflow.log_param("model", "XGBRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("test_size", 0.20)


    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)


    # Model validation
    print("\nModel Validation")
    print("----------------")

    print(f"Required R²: {R2_THRESHOLD:.2f}")
    print(f"Actual R²  : {r2:.4f}")


    if r2 >= R2_THRESHOLD:

        print("VALIDATION PASSED")
        print("Model is approved.")

        # Save model
        model_path = BASE_DIR / "models" / "xgboost_house_price_model.json"
        model.save_model(model_path)

        print(f"Model saved to: {model_path}")

        # Log model to MLflow
        mlflow.xgboost.log_model(
            model,
            name="house_price_model"
        )

    else:

        print("VALIDATION FAILED")
        print("Model is NOT approved.")


    print("\nMLflow run completed successfully.")