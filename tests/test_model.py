from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "house_prices.csv"


def build_model():

    df = pd.read_csv(DATA_PATH)

    X = df.drop("price", axis=1)
    y = df["price"]

    numerical_features = [
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "parking",
        "age"
    ]

    categorical_features = ["location"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)

    return df, y_test, predictions


def test_dataset_exists():
    assert DATA_PATH.exists()


def test_dataset_shape():
    df = pd.read_csv(DATA_PATH)

    assert df.shape == (20, 8)


def test_model_r2():
    _, y_test, predictions = build_model()

    r2 = r2_score(y_test, predictions)

    assert r2 >= 0.50