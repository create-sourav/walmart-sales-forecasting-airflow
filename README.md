# 🏬 Walmart Sales Forecasting — End-to-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue.svg)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Apache%20Airflow-2.0+-red.svg)](https://airflow.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end MLOps pipeline for forecasting Walmart weekly sales using Machine Learning, Docker, and Apache Airflow. This project demonstrates production-grade ML workflows with proper orchestration, containerization, and reproducibility.

## 📌 Project Overview

This project implements a complete MLOps pipeline that goes beyond model accuracy to showcase:

- **Structured project layout** following industry best practices
- **Reproducible training** using Docker containerization
- **Automated ML pipelines** orchestrated with Apache Airflow
- **Proper handling** of data, models, and experiments
- **Production-ready** design suitable for interviews and real-world applications

## 🎯 Business Problem

Walmart operates across multiple stores with fluctuating weekly sales influenced by various factors:

- 🎉 **Holidays** — Special events impact shopping behavior
- ⛽ **Fuel Price** — Transportation costs affect consumer spending
- 📊 **CPI** — Consumer Price Index reflects inflation
- 💼 **Unemployment** — Job market conditions influence purchasing power
- 🌦️ **Seasonal Patterns** — Weather and time of year drive demand

**Objective:** Predict weekly sales accurately using historical data and time-series-aware feature engineering.

## 🧠 Solution Approach

### 1️⃣ Data Understanding

- **Dataset Source:** [Kaggle – Walmart Sales Forecasting](https://www.kaggle.com/)
- **Observations:** Weekly sales per store with economic indicators
- **Time-Series Nature:** Data is preserved chronologically (no random shuffling)

### 2️⃣ Feature Engineering

Implemented time-series aware features to capture temporal patterns:

**Temporal Features:**
- Year, Month, Week extraction

**Lag Features:**
- `Weekly_Sales_lag_1` — Previous week's sales
- `Weekly_Sales_lag_2` — Sales from two weeks ago

**Rolling Statistics:**
- `Rolling_mean_4` — 4-week moving average
- `Rolling_mean_12` — 12-week moving average

**Categorical Encoding:**
- One-Hot Encoding for Store identifiers

**Scaling:**
- StandardScaler for numerical features

### 3️⃣ Model Training & Evaluation

Multiple models were trained and compared:

| Model | Description |
|-------|-------------|
| **Linear Regression** | Baseline model |
| **Random Forest Regressor** | Ensemble method with decision trees |
| **Gradient Boosting Regressor** ✅ | Best performing model |

**Evaluation Metrics:**
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)

### 📊 Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Gradient Boosting ✅ | Low | Low | High |
| Random Forest | Moderate | Moderate | High |
| Linear Regression | High | High | Low |

> **Note:** Metrics are evaluated using time-based train/test split, not random split, to respect the temporal nature of the data.

**Winner:** Gradient Boosting Regressor achieved the best balance of accuracy and generalization.

## ⚙️ MLOps Architecture

### 🔁 Pipeline Automation (Airflow)

The entire ML workflow is orchestrated using Apache Airflow:

- **DAG Name:** `walmart_ml_pipeline`
- **Pipeline Stages:**
  1. **Preprocess Data** — Feature engineering and transformation
  2. **Train Model** — Model training and comparison
  3. **Test Model** — Evaluation and prediction generation

Each stage is an isolated, repeatable task ensuring modularity and maintainability.

### 🐳 Containerization (Docker)

- All dependencies are containerized for consistency
- Ensures reproducibility across different machines
- Separate Docker setup for:
  - **ML Execution** — Model training and inference
  - **Airflow Orchestration** — Workflow management

## 📂 Project Structure

```
walmart_sales/
│
├── airflow/
│   ├── dags/
│   │   └── walmart_ml_pipeline.py   # Airflow DAG definition
│   ├── Dockerfile                   # Custom Airflow image
│   └── docker-compose.yml           # Airflow orchestration setup
│
├── src/
│   ├── inspect_data.py              # Data inspection utilities
│   ├── preprocess_data.py           # Feature engineering pipeline
│   ├── train.py                     # Model training script
│   ├── test.py                      # Model evaluation script
│   └── future_forecast.py           # Future predictions generator
│
├── data/
│   ├── walmart_sales.csv            # Raw dataset
│   ├── processed_walmart_sales.csv  # Processed features
│   └── future_sales_forecast.csv    # Forecast output
│
├── models/
│   ├── Gradient_Boosting.pkl        # Trained GB model
│   ├── Random_Forest.pkl            # Trained RF model
│   ├── Linear_Regression.pkl        # Trained LR model
│   └── model_performance_metrics.csv # Model comparison results
│
├── Dockerfile                        # ML container definition
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Docker (20.10+)
- Docker Compose
- Git

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/create-sourav/walmart-sales-forecasting-airflow.git
cd walmart-sales-forecasting-airflow
```

### 2️⃣ Build & Run Airflow

```bash
cd airflow
docker compose up -d
```

### 3️⃣ Access Airflow UI

Open your browser and navigate to:

```
http://localhost:8080
```

### 4️⃣ Trigger the ML Pipeline

1. In the Airflow UI, locate the `walmart_ml_pipeline` DAG
2. Enable the DAG using the toggle switch
3. Click **"Trigger DAG"** to start the pipeline
4. Monitor the execution of each task:
   - ✅ `preprocess_data`
   - ✅ `train_model`
   - ✅ `test_model`

### 🔐 Airflow Credentials

When using `airflow standalone`, credentials are auto-generated.

To view them:

```bash
docker logs airflow
```

## 🔄 Handling New Incoming Data (Incremental Retraining)

This pipeline is designed to support new incoming data without breaking the existing workflow.

### 📥 Scenario

Walmart receives new weekly sales data (for new dates or stores) periodically.

**Example:**
- New week's sales CSV arrives every Sunday
- Data structure remains the same as `walmart_sales.csv`

### 🧩 How New Data Is Incorporated

#### 1️⃣ Append New Data (Not Replace)

New data should be appended to the existing raw dataset:

**📂 Location:**
```
data/walmart_sales.csv
```

**Process:**
- Add new rows at the bottom
- Do not modify historical records
- Maintain the same schema (columns)

**Example Schema:**
```
Store | Date | Weekly_Sales | Holiday_Flag | Temperature | Fuel_Price | CPI | Unemployment
```

#### 2️⃣ Preprocessing Automatically Handles New Data

The `preprocess_data.py` script is idempotent and time-aware:

**What it does every run:**
- Reads the entire updated dataset
- Sorts by `Store` and `Date`
- Recomputes:
  - Lag features
  - Rolling averages
  - Time features
- Drops invalid rows caused by lag creation
- Saves a fresh processed dataset

**📄 Output:**
```
data/processed_walmart_sales.csv
```

➡️ **No manual feature updates needed when new data arrives.**

#### 3️⃣ Time-Based Training Preserves Order

The training script uses time-based splitting to ensure newer data is always used for testing:

```python
split = int(len(df) * 0.8)
train = df.iloc[:split]
test  = df.iloc[split:]
```

This ensures:
- **Older data** → Training
- **Newer data** → Validation/Testing
- **No data leakage** between train and test sets

#### 4️⃣ Model Retraining Is Automatic via Airflow

Once new data is added:

1. **Trigger the DAG manually** OR
2. **Let the scheduled DAG run** execute automatically

**Airflow automatically:**
1. Preprocesses updated data
2. Retrains all models
3. Evaluates performance
4. Saves updated models

**📂 Updated models overwrite older versions:**
```
models/gradient_boosting_model.pkl
```

This ensures:
- Always using the latest trained model
- No stale predictions

### 🔁 Typical Production Flow

```
New Data Arrives
    ↓
Raw CSV Updated
    ↓
Airflow DAG Triggered
    ↓
Preprocessing
    ↓
Model Retraining
    ↓
Model Testing
    ↓
Updated Model Saved
```

### 🧠 Why This Design Is MLOps-Correct

- ✅ **No manual intervention** — Fully automated pipeline
- ✅ **No feature leakage** — Time-based splitting
- ✅ **Fully reproducible** — Idempotent preprocessing
- ✅ **Scalable** — Works with weekly/monthly updates
- ✅ **Industry standard** — Batch-based retraining approach

## 🧪 Key MLOps Concepts Demonstrated

- ✅ **Time-Series Aware ML** — Proper handling of temporal data
- ✅ **Feature Engineering Best Practices** — Lag features and rolling statistics
- ✅ **Model Comparison & Evaluation** — Systematic model selection
- ✅ **Reproducible ML using Docker** — Containerized environments
- ✅ **Workflow Orchestration with Airflow** — Automated pipelines
- ✅ **Production-Style Project Structure** — Industry-standard organization
- ✅ **Incremental Retraining Support** — Handles new data seamlessly
- ✅ **GitHub-Ready MLOps Repository** — Professional presentation

## 📌 Important Notes

- Large model files should ideally be stored using **Git LFS** or a model registry (e.g., **MLflow**)
- This project focuses on **MLOps workflow design**, not just model accuracy
- Dataset files are kept intentionally for learning and demonstration purposes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sourav Mondal**  
MBA (Business Analytics)  
Aspiring Data Scientist / MLOps Engineer

- GitHub: [@create-sourav](https://github.com/create-sourav)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/sourav-mondal)

## ⭐ Show Your Support

If you find this project useful, please consider giving it a ⭐ on GitHub — it helps a lot!

---

**Built with ❤️ for the Data Science and MLOps community**
