import pandas as pd
from psycopg2.extras import execute_values
import psycopg2 as pg2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

#PASSWORD#
password_from_env = os.getenv("PASSWORD")

#READING CSV#
df = pd.read_csv("data/processed_data/ai4i2020_processed.csv")

#CONNECTING TO DATABASE#
try:
    conn = pg2.connect(
        host="localhost",
        database="predictive_maintenance",
        user="postgres",
        password=password_from_env  
    )
except pg2.Error as e:
    print(f"Error connecting to PostgreSQL: {e}")

#INSERTING DATA INTO DATABASE#
try:
    cursor = conn.cursor()
    
    values = [
    (
        int(row['type']),
        float(row['air_temperature_k']),
        float(row['process_temperature_k']),
        int(row['rotational_speed_rpm']),
        float(row['torque_nm']),
        int(row['tool_wear_min']),
        int(row['machine_failure']),
        int(row['twf']),
        int(row['hdf']),
        int(row['pwf']),
        int(row['osf']),
        int(row['rnf'])
    )
    for index, row in df.iterrows()
]
    cursor.execute("TRUNCATE TABLE sensor_data RESTART IDENTITY CASCADE;")
    execute_values(cursor, """
        INSERT INTO sensor_data (
            type, air_temperature_k, process_temperature_k,
            rotational_speed_rpm, torque_nm, tool_wear_min,
            machine_failure, twf, hdf, pwf, osf, rnf
        ) VALUES %s
    """, values)
    
    conn.commit()
    print(f"[INGESTION] Data inserted successfully into PostgreSQL.")
except pg2.Error as e:
    print(f"Error inserting data into PostgreSQL: {e}")
finally:
    if conn:
        conn.close()

