import os
import logging
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Logger (Airflow-friendly)
# -----------------------------
logger = logging.getLogger(__name__)

# -----------------------------
# Future Forecast Function
# -----------------------------
def forecast_future_sales(n_weeks: int = 4):
    """
    Generates future weekly sales forecasts using recursive prediction.
    Saves results to /opt/airflow/data/future_sales_forecast.csv
    """

    DATA_DIR = "/opt/airflow/data"
    MODEL_DIR = "/opt/airflow/models"

    model_path = os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")
    data_path = os.path.join(DATA_DIR, "processed_walmart_sales.csv")

    # -----------------------------
    # Load model and data
    # -----------------------------
    logger.info("📦 Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("📥 Loading processed data from %s", data_path)
    df = pd.read_csv(data_path)

    # -----------------------------
    # Separate features & target
    # -----------------------------
    target_col = "Weekly_Sales"
    feature_cols = [c for c in df.columns if c != target_col]

    # Use last row as starting point
    last_row = df.iloc[-1:].copy()

    forecasts = []

    # -----------------------------
    # Recursive forecasting
    # -----------------------------
    for step in range(1, n_weeks + 1):
        X_future = last_row[feature_cols]
        y_pred = model.predict(X_future)[0]

        forecasts.append({
            "Week_Ahead": step,
            "Predicted_Weekly_Sales": round(float(y_pred), 2)
        })

        # -----------------------------
        # Update lag & rolling features
        # -----------------------------
        if "Weekly_Sales_lag_2" in last_row.columns:
            last_row["Weekly_Sales_lag_2"] = last_row["Weekly_Sales_lag_1"]

        if "Weekly_Sales_lag_1" in last_row.columns:
            last_row["Weekly_Sales_lag_1"] = y_pred

        if "Rolling_mean_4" in last_row.columns:
            last_row["Rolling_mean_4"] = y_pred

        if "Rolling_mean_12" in last_row.columns:
            last_row["Rolling_mean_12"] = y_pred

        last_row[target_col] = y_pred

        logger.info(
            "📈 Week %d forecast: %.2f",
            step,
            y_pred
        )

    # -----------------------------
    # Save forecast output
    # -----------------------------
    forecast_df = pd.DataFrame(forecasts)
    output_path = os.path.join(DATA_DIR, "future_sales_forecast.csv")
    forecast_df.to_csv(output_path, index=False)

    logger.info("✅ Future sales forecast saved to %s", output_path)
    logger.info("📊 Forecast preview:\n%s", forecast_df.to_string(index=False))


# -----------------------------
# CLI entry (safe)
# -----------------------------
if __name__ == "__main__":
    forecast_future_sales(n_weeks=4)
