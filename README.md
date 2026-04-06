# 🏭 Predictive Maintenance System

A complete end-to-end Machine Learning pipeline for industrial equipment failure prediction, built as a learning project covering data engineering, ML modeling, and business intelligence.

---

## 📋 Overview

This project simulates a real-world predictive maintenance system using sensor data from industrial machines. It predicts the probability of equipment failure based on operational parameters such as temperature, torque, rotational speed, and tool wear.

The system was built to demonstrate a full data pipeline — from raw data ingestion to a Power BI dashboard with failure alerts.

In a real factory environment, the pipeline would be scheduled to run daily — ingesting new sensor readings, generating updated predictions, and refreshing the dashboard automatically to support maintenance decisions in near real-time.

---

## 📈 Power BI Dashboard

Two-page dashboard connected directly to PostgreSQL:

**Page 1 — Overview**
![Overview](https://raw.githubusercontent.com/alan-colli/predictive_ML/main/BI_pictures/overview.png)

**Page 2 — Alert Panel**
![Alert Panel](https://raw.githubusercontent.com/alan-colli/predictive_ML/main/BI_pictures/alert_panel.png)

**Page 3 — Selected Machine**
![Alert Panel](https://raw.githubusercontent.com/alan-colli/predictive_ML/main/BI_pictures/selected_machine.png)

---

## 🗂️ Project Structure

```
predictive_ML/
│
├── data/
│   ├── raw_data/
│   │   └── ai4i2020.csv              # Original dataset (not versioned)
│   └── processed_data/
│       └── ai4i2020_processed.csv    # Cleaned dataset (not versioned)
│
├── sql/
│   └── 01_create_tables.sql          # PostgreSQL schema
│
├── scripts/
│   ├── pipeline.py                   # Data cleaning and preprocessing
│   ├── ingestion.py                  # CSV → PostgreSQL ingestion
│   ├── features.py                   # Feature engineering + scaling
│   ├── treino.py                     # Model training and evaluation
│   └── predict.py                    # Predictions → PostgreSQL
│
├── models/
│   ├── random_forest_model.pkl       # Trained Random Forest model
│   └── scaler.pkl                    # Fitted StandardScaler
│
├── BI_pictures/
│   ├── overview.png
│   └── alert_panel.png
│
├── .env                              # Credentials (not versioned)
├── .gitignore
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Layer            | Technology    |
| ---------------- | ------------- |
| Language         | Python 3.x    |
| Data Processing  | Pandas        |
| Database         | PostgreSQL    |
| Machine Learning | Scikit-learn  |
| Visualization    | Power BI      |
| Environment      | python-dotenv |
| DB Connector     | psycopg2      |

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- 10,000 records | 14 features
- Binary classification target: `machine_failure` (0 or 1)
- Class imbalance: ~96.6% no failure | ~3.4% failure

---

## ⚙️ Pipeline

### 1. Data Cleaning (`pipeline.py`)

- Renamed columns to snake_case
- Dropped identifier columns (`udi`, `product_id`)
- Encoded categorical variable `type`: L=0, M=1, H=2
- Detected and logged 18 inconsistencies between fault indicators and `machine_failure`
- Exported cleaned CSV to `processed_data/`

### 2. Database Ingestion (`ingestion.py`)

- Connected to PostgreSQL via `psycopg2`
- Bulk insert using `execute_values` for performance
- Idempotent via `TRUNCATE + RESTART IDENTITY` before each load

### 3. Feature Engineering (`features.py`)

- Read data directly from PostgreSQL
- Created domain-driven features:
  - `temp_difference` = process_temperature_k − air_temperature_k
  - `power` = (torque*nm × rotational_speed_rpm) / 9550 *(kW)\_
  - `torque_wear_ratio` = torque_nm / (tool_wear_min + 1)
- Excluded fault sub-type columns (TWF, HDF, PWF, OSF, RNF) to prevent **data leakage**
- Applied `StandardScaler` and saved as `scaler.pkl`

### 4. Model Training (`treino.py`)

- Algorithm: **Random Forest Classifier**
- `class_weight='balanced'` to handle class imbalance
- 80/20 train-test split with stratification
- **Threshold tuning**: 0.5 → 0.3 to improve recall on the minority class

#### Results

| Metric    | Class 0 (No Failure) | Class 1 (Failure) |
| --------- | -------------------- | ----------------- |
| Precision | 0.99                 | 0.87              |
| Recall    | 1.00                 | 0.81              |
| F1-Score  | 0.99                 | 0.84              |

**Confusion Matrix:**

```
[[1924    8]
 [  13   55]]
```

### 5. Predictions (`predict.py`)

- Loaded trained model and scaler from `.pkl` files
- Applied same feature engineering as training
- Used threshold of **0.3** for failure classification
- Saved results to `predictions` table in PostgreSQL

---

## 🗄️ Database Schema

```sql
sensor_data     -- cleaned sensor readings (10,000 rows)
predictions     -- model output with failure_probability and predicted_failure
```

---

## 🚀 How to Run

**1. Clone the repository**

```bash
git clone https://github.com/alan-colli/predictive_ML.git
cd predictive_ML
```

**2. Create and activate virtual environment**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the root:

```
PASSWORD=your_postgres_password
```

**5. Create the database**

```bash
psql -U postgres -c "CREATE DATABASE predictive_maintenance;"
```

**6. Run the pipeline in order**

```bash
python scripts/pipeline.py
python scripts/ingestion.py
python scripts/features.py
python scripts/treino.py
python scripts/predict.py
```

---

## 📦 Requirements

```
pandas
psycopg2-binary
python-dotenv
scikit-learn
```

Generate with:

```bash
pip freeze > requirements.txt
```

---

## 💡 Key Learnings

- **Data leakage** is a concrete risk when fault sub-type columns are present alongside the aggregate failure target
- **Threshold tuning** (0.5 → 0.3) meaningfully improved recall on the rare failure class — critical for imbalanced industrial datasets where missing a failure is costly
- **Domain knowledge** (engineering background) directly informed feature engineering choices — power, torque/wear ratio, temperature delta
- **Simplifying architecture** (dropping medallion layers) was the right call for a project of this scope

---
