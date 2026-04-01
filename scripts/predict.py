import pandas as pd
import psycopg2 as pg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import pickle
import os
 
load_dotenv()
password_from_env = os.getenv("PASSWORD")

# ─────────────────────────────────────────
# LOAD MODEL AND SCALER
# ─────────────────────────────────────────
with open("models/random_forest_model.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("[LOAD] Model and scaler loaded successfully.")

# ─────────────────────────────────────────
# CONNECT TO DATABASE
# ─────────────────────────────────────────
try:
    conn = pg2.connect(
        host="localhost",
        database="predictive_maintenance",
        user="postgres",
        password=password_from_env,
        port=5432
    )
    print("[DB] Connected to the database successfully.")
except Exception as e:
    print(f"[DB] Error connecting to the database: {e}")
    exit(1)

# ─────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────
query = """
    SELECT * FROM sensor_data
    ORDER BY RANDOM()
    LIMIT 10000
"""
df = pd.read_sql_query(query, conn)
print(f"[LOAD] {df.shape[0]} rows loaded from PostgreSQL.")

# ─────────────────────────────────────────
# PREPROCESS DATA
# ─────────────────────────────────────────
# domain features
df["temp_difference"]   = df["process_temperature_k"] - df["air_temperature_k"]
df["power"]             = (df["torque_nm"] * df["rotational_speed_rpm"]) / 9550
df["torque_wear_ratio"] = df["torque_nm"] / (df["tool_wear_min"] + 1)

# remove features that won't be used for prediction
X = df.drop(columns=["machine_failure", "id", "inserted_at", "twf", "hdf", "pwf", "osf", "rnf"])

# applying scaler
X_scaled = scaler.transform(X)
print(f"[FEATURES] Features prepared and scaled successfully.")

# ─────────────────────────────────────────
# MAKE PREDICTIONS
# ─────────────────────────────────────────
y_prob = modelo.predict_proba(X_scaled)[:, 1]
y_pred = (y_prob >= 0.3).astype(int)
print(f"[PREDICT] {y_pred.sum()} failures predicted out of {len(y_pred)} records.")

# ─────────────────────────────────────────
# SAVE PREDICTIONS BACK TO DATABASE
# ─────────────────────────────────────────
try:
    cursor = conn.cursor()

    # clear previous predictions before inserting new ones
    cursor.execute("TRUNCATE TABLE predictions RESTART IDENTITY;")
    print("[DB] Predictions table cleared.")

    values = [
        (
            int(df.iloc[i]["id"]),
            float(y_prob[i]),
            int(y_pred[i])
        )
        for i in range(len(y_pred))
    ]
 
    execute_values(cursor, """
        INSERT INTO predictions (sensor_data_id, failure_probability, predicted_failure)
        VALUES %s
    """, values)
 
    conn.commit()
    print(f"[SAVE] {len(values)} predictions saved to database.")
 
except pg2.Error as e:
    print(f"[ERROR] {e}")
 
finally:
    conn.close()