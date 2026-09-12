from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .data import prepare_features


@dataclass
class BenchmarkResult:
    metrics: pd.DataFrame
    predictions: dict[str, np.ndarray]


def _models(random_state: int = 42) -> dict:
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "MLP/ANN": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=800,
                random_state=random_state,
                early_stopping=True,
            ),
        ),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
        )
    except ImportError:
        pass
    return models


def benchmark_regressors(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    test_size: float = 0.30,
    random_state: int = 42,
) -> BenchmarkResult:
    """Benchmark baseline regressors with a reproducible random split."""
    X, y = prepare_features(df, features, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rows = []
    predictions = {}
    for name, model in _models(random_state).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        })

    metrics = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return BenchmarkResult(metrics=metrics, predictions=predictions)
