import os
import logging
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Logger
# -----------------------------
logger = logging.getLogger(__name__)

# -----------------------------
# Main callable for Airflow
# -----------------------------
def train_model():

    DATA_DIR = "/opt/airflow/data"
    MODEL_DIR = "/opt/airflow/models"

    os.makedirs(MODEL_DIR, exist_ok=True)

    processed_path = os.path.join(DATA_DIR, "processed_walmart_sales.csv")

    logger.info("📥 Loading processed data from %s", processed_path)
    df = pd.read_csv(processed_path)

    # -----------------------------
    # Features & target
    # -----------------------------
    X = df.drop(columns=["Weekly_Sales"])
    y = df["Weekly_Sales"]

    # -----------------------------
    # Time-based split (80/20)
    # -----------------------------
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test  = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test  = y.iloc[split_index:]

    logger.info("📊 Train size: %d | Test size: %d", len(X_train), len(X_test))

    # -----------------------------
    # Define models
    # -----------------------------
    models = {
        "Linear_Regression": LinearRegression(),
        "Random_Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient_Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )
    }

    # -----------------------------
    # Train & evaluate
    # -----------------------------
    results = []
    trained_models = {}

    for name, model in models.items():
        logger.info("🚀 Training model: %s", name)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        trained_models[name] = model

        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)

        logger.info(
            "📈 %s | RMSE: %.2f | MAE: %.2f | R²: %.3f",
            name, rmse, mae, r2
        )

    # -----------------------------
    # Model comparison (persisted)
    # -----------------------------
    results_df = pd.DataFrame(results).sort_values("RMSE")

    metrics_path = os.path.join(MODEL_DIR, "model_performance_metrics.csv")
    results_df.to_csv(metrics_path, index=False)

    logger.info(
        "📊 MODEL PERFORMANCE COMPARISON saved to %s\n%s",
        metrics_path,
        results_df.to_string(index=False)
    )

    # -----------------------------
    # Save best model (Gradient Boosting)
    # -----------------------------
    best_model = trained_models["Gradient_Boosting"]
    best_model_path = os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")

    joblib.dump(best_model, best_model_path)

    logger.info("🏆 Best model saved to %s", best_model_path)


# -----------------------------
# CLI entry (safe)
# -----------------------------
if __name__ == "__main__":
    train_model()
