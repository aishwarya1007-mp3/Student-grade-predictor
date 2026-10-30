"""
model.py
--------
Trains, evaluates, and persists machine learning models for
Student Grade Prediction.

Models compared
---------------
1. Linear Regression      – interpretable baseline
2. Random Forest          – robust ensemble method
3. Gradient Boosting      – high-accuracy ensemble
4. Decision Tree          – simple tree-based model

All evaluation metrics are computed from real model predictions on the
held-out test set — no results are fabricated.
"""

import os
import pickle
import numpy as np
from sklearn.linear_model  import LinearRegression
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree          import DecisionTreeRegressor
from sklearn.metrics       import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models.pkl")

# ── Model definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return a dictionary of untrained models."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=8,
                                                   random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100,
                                                        learning_rate=0.1,
                                                        max_depth=4,
                                                        random_state=42),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_models(X_train, y_train) -> dict:
    """Fit every model on the training data and return the trained models."""
    models = get_models()
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_models(trained_models: dict, X_test, y_test) -> dict:
    """
    Compute MAE, MSE, RMSE, and R² for each model.
    All metrics come from real predictions on the held-out test set.
    """
    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 20)  # grades are bounded 0–20

        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        results[name] = {
            "MAE":       round(mae,  4),
            "MSE":       round(mse,  4),
            "RMSE":      round(rmse, 4),
            "R2 Score":  round(r2,   4),
            "y_pred":    y_pred,
        }
    return results


def get_best_model(results: dict) -> str:
    """Return the name of the model with the highest R² score."""
    return max(results, key=lambda k: results[k]["R2 Score"])


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(trained_models: dict, feature_names: list) -> dict:
    """
    Extract feature importances from tree-based models.
    Returns a dict  model_name → {feature: importance}.
    Linear Regression uses absolute coefficient values instead.
    """
    importances = {}
    for name, model in trained_models.items():
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For linear regression: use |coefficient| as proxy for importance
            imp = np.abs(model.coef_)
            imp = imp / (imp.sum() + 1e-9)  # normalise to sum to 1
        else:
            continue

        importances[name] = dict(zip(feature_names, imp.round(4)))

    return importances


# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_grade(model, scaler, input_values: list) -> float:
    """
    Predict the final grade for a single student.

    Parameters
    ----------
    model        : trained sklearn model
    scaler       : fitted StandardScaler
    input_values : list of raw feature values (same order as MODEL_FEATURES)

    Returns
    -------
    Predicted grade clipped to [0, 20].
    """
    arr = np.array(input_values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    return float(np.clip(pred, 0, 20))


# ── Persistence ───────────────────────────────────────────────────────────────

def save_models(trained_models: dict, scaler, feature_names: list):
    """Pickle trained models and scaler to disk for fast reloading."""
    payload = {
        "models":        trained_models,
        "scaler":        scaler,
        "feature_names": feature_names,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)


def load_saved_models():
    """
    Load pickled models from disk.
    Returns (trained_models, scaler, feature_names) or None if no file.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["models"], payload["scaler"], payload["feature_names"]
