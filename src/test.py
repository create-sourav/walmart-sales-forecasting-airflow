import os
import logging
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Logger
# -----------------------------
logger = logging.getLogger(__name__)

# -----------------------------
# Main callable for Airflow
# -----------------------------
def evaluate_model():

    DATA_DIR = "/opt/airflow/data"
    MODEL_DIR = "/opt/airflow/models"

    processed_path = os.path.join(DATA_DIR, "processed_walmart_sales.csv")
    model_path = os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")

    logger.info("📥 Loading processed data from %s", processed_path)
    df = pd.read_csv(processed_path)

    # -----------------------------
    # Features & target
    # -----------------------------
    X = df.drop(columns=["Weekly_Sales"])
    y = df["Weekly_Sales"]

    # -----------------------------
    # Use last 20% as test set
    # -----------------------------
    split_index = int(len(df) * 0.8)

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    logger.info("Test samples: %d", len(X_test))

    # -----------------------------
    # Load trained model
    # -----------------------------
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    # -----------------------------
    # Predict
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # Evaluate
    # -----------------------------
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    logger.info("Model evaluation completed")
    logger.info("RMSE: %.2f", rmse)
    logger.info("MAE : %.2f", mae)
    logger.info("R²  : %.3f", r2)

    # -----------------------------
    # Sample predictions (for logs)
    # -----------------------------
    results = pd.DataFrame({
        "Actual_Weekly_Sales": y_test.values[:10],
        "Predicted_Weekly_Sales": y_pred[:10]
    })

    logger.info("🔍 Sample predictions:\n%s", results.to_string(index=False))


# -----------------------------
# CLI entry (safe)
# -----------------------------
if __name__ == "__main__":
    evaluate_model()
