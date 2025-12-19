import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Logger (Airflow-friendly)
# -----------------------------
logger = logging.getLogger(__name__)

# -----------------------------
# Main callable for Airflow
# -----------------------------
def run_preprocessing():

    # -----------------------------
    # Data directory (Airflow mount)
    # -----------------------------
    DATA_DIR = "/opt/airflow/data"
    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("📂 Using data directory: %s", DATA_DIR)

    # -----------------------------
    # Step 1: Load raw data
    # -----------------------------
    input_path = os.path.join(DATA_DIR, "walmart_sales.csv")
    logger.info("📥 Loading raw data from %s", input_path)

    df = pd.read_csv(input_path)

    # -----------------------------
    # Step 2: Convert Date
    # -----------------------------
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    # -----------------------------
    # Step 3: Sort by Store & Date
    # -----------------------------
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # -----------------------------
    # Step 4: Time features
    # -----------------------------
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)

    # -----------------------------
    # Step 5: Lag features (per store)
    # -----------------------------
    df["Weekly_Sales_lag_1"] = df.groupby("Store")["Weekly_Sales"].shift(1)
    df["Weekly_Sales_lag_2"] = df.groupby("Store")["Weekly_Sales"].shift(2)

    # -----------------------------
    # Step 6: Rolling features
    # -----------------------------
    df["Rolling_mean_4"] = (
        df.groupby("Store")["Weekly_Sales"]
          .shift(1)
          .rolling(window=4)
          .mean()
    )

    df["Rolling_mean_12"] = (
        df.groupby("Store")["Weekly_Sales"]
          .shift(1)
          .rolling(window=12)
          .mean()
    )

    # -----------------------------
    # Step 7: Drop NaNs
    # -----------------------------
    df = df.dropna().reset_index(drop=True)

    # -----------------------------
    # Step 8: Drop Date column
    # -----------------------------
    df = df.drop(columns=["Date"])

    # -----------------------------
    # Step 9: One-hot encode Store
    # -----------------------------
    df = pd.get_dummies(df, columns=["Store"], drop_first=True)

    # -----------------------------
    # Step 10: Separate target
    # -----------------------------
    y = df["Weekly_Sales"]
    X = df.drop(columns=["Weekly_Sales"])

    # -----------------------------
    # Step 11: Scale features
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # -----------------------------
    # Step 12: Combine & save
    # -----------------------------
    processed_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    output_path = os.path.join(DATA_DIR, "processed_walmart_sales.csv")
    processed_df.to_csv(output_path, index=False)

    logger.info("✅ Preprocessing completed successfully")
    logger.info("📁 Saved processed data to %s", output_path)
    logger.info("📊 Final shape: %s", processed_df.shape)


# -----------------------------
# CLI entry (safe)
# -----------------------------
if __name__ == "__main__":
    run_preprocessing()
