from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.append("/opt/airflow/src")

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2)
}

# ----------------------------
# Task wrappers
# ----------------------------

def preprocess_task():
    from preprocess_data import run_preprocessing
    run_preprocessing()

def train_task():
    from train import train_model
    train_model()

def test_task():
    from test import evaluate_model
    evaluate_model()

def future_forecast_task():
    from future_forecast import forecast_future_sales
    forecast_future_sales(n_weeks=4)

# ----------------------------
# DAG definition
# ----------------------------

with DAG(
    dag_id="walmart_ml_pipeline",
    description="End-to-end Walmart Sales ML Pipeline (Metrics, Predictions, Future Forecast)",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "forecasting", "airflow", "walmart"]
) as dag:

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_task
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_task
    )

    test = PythonOperator(
        task_id="test_model",
        python_callable=test_task
    )

    future_forecast = PythonOperator(
        task_id="future_sales_forecast",
        python_callable=future_forecast_task
    )

    preprocess >> train >> test >> future_forecast
