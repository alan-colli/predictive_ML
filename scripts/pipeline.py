import pandas as pd
import os

#PATHS#
raw_path = "data/raw_data/ai4i2020.csv"

#LOADING DATA#
def load_data(raw_path):
    df = pd.read_csv(raw_path)
    print(f"[LOAD] {df.shape[0]} lines | {df.shape[1]} columns")
    return df

#RENAMING COLUMNS#
def rename_columns(df):
    rename_mapping = {
       "UDI"                        : "udi",
        "Product ID"                 : "product_id",
        "Type"                       : "type",
        "Air temperature [K]"        : "air_temperature_k",
        "Process temperature [K]"    : "process_temperature_k",
        "Rotational speed [rpm]"     : "rotational_speed_rpm",
        "Torque [Nm]"                : "torque_nm",
        "Tool wear [min]"            : "tool_wear_min",
        "Machine failure"            : "machine_failure",
        "TWF"                        : "twf",
        "HDF"                        : "hdf",
        "PWF"                        : "pwf",
        "OSF"                        : "osf",
        "RNF"                        : "rnf",  
    }
    df = df.rename(columns=rename_mapping)
    print(f"[RENAME] Columns renamed successfully.")
    return df

#ENCODING VARIABLES (STRING TO INT)#
def encode_type(df):
    type_mapping = {
            "L": 0,
            "M": 1,
            "H": 2
        }
    df = df.replace({"type": type_mapping})
        
    invalid = df["type"].isna().sum()
    if invalid > 0:
        print(f"[WARNING] Found {invalid} invalid entries in 'type' column. Filling with mode.")
        df["type"] = df["type"].fillna(df["type"].mode()[0])

    print(f"[ENCODE] Type column encoded successfully.")
    return df






df = load_data(raw_path)
df = rename_columns(df)
df = encode_type(df)