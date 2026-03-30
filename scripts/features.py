import pandas as pd
import psycopg2 as pg2
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import pickle
import os

load_dotenv()
password_from_env = os.getenv("PASSWORD")

conn = pg2.connect(
    host="localhost",
    database="predictive_maintenance",
    user="postgres",
    password=password_from_env
)

df = pd.read_sql_query("SELECT * FROM sensor_data", conn)
print(f"[LOAD] {df.shape[0]} rows and {df.shape[1]} columns loaded from PostgreSQL.")


# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def temp_difference(df):
    df["temp_difference"] = df["process_temperature_k"] - df["air_temperature_k"]
    print(f"[FEATURE] 'temp_difference' created successfully.")
    return df

def power(df):
    df["power"] = (df["torque_nm"] * df["rotational_speed_rpm"]) / 9550
    print(f"[FEATURE] 'power' created successfully.")
    return df

def torque_wear_ratio(df):
    df["torque_wear_ratio"] = df["torque_nm"] / (df["tool_wear_min"] + 1)
    print(f"[FEATURE] 'torque_wear_ratio' created successfully.")
    return df

def run_all_features(df):
    df = temp_difference(df)
    df = power(df)
    df = torque_wear_ratio(df)
    return df


# ─────────────────────────────────────────
# SCALING
# ─────────────────────────────────────────
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"[SCALER] Features scaled and scaler saved to models/scaler.pkl")
    return X_scaled


# ─────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────
df = run_all_features(df)

X = df.drop(columns=["machine_failure", "id", "inserted_at", "twf", "hdf", "pwf", "osf", "rnf"])
y = df["machine_failure"]
print(f"[FEATURES] X shape: {X.shape} | y shape: {y.shape}")

X_scaled = scale_features(X)