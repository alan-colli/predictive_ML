import pandas as pd
import os

#PATHS#
raw_path = "data/raw_data/ai4i2020.csv"
processed_path = "data/processed_data/ai4i2020_processed.csv"

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

#REMOVING USELESS COLUMNS#
def remove_useless_columns(df):
    df = df.drop(columns=["udi", "product_id"], errors="ignore")
    print(f"[REMOVE] Useless columns removed.")
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

#CHECKING INCONSISTENCES#
def check_inconsistences(df):
    fault_cols = ["twf", "hdf", "pwf", "osf", "rnf"]

    mask_inconsistences = (df[fault_cols].sum(axis=1) > 0) & (df["machine_failure"] == 0)
    inconsistences = mask_inconsistences.sum()
    if inconsistences > 0:
        print(f"[WARNING] Found {inconsistences} inconsistences between fault indicators and machine failure.")
    else:
        print(f"[CHECK] No inconsistences found between fault indicators and machine failure.")
    return df

#CHECKING NULL VALUES#
def check_null_values(df):
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print(f"[WARNING] Found {total_nulls} null values in the dataset:")
        print(null_counts[null_counts > 0])
    else:
        print(f"[CHECK] No null values found in the dataset.")
    return df

#SAVING PROCESSED CSV#
def save_processed_data(df, processed_path):
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"[SAVE] Processed data saved to {processed_path}")



#MAIN EXECUTION#
def run_pipeline():
    print("="*15)
    print("CLEANING PIPELINE - AI4I2020 Dataset")
    print("="*15)


    df = load_data(raw_path)
    df = rename_columns(df)
    df = encode_type(df)
    df = remove_useless_columns(df) 
    df = check_inconsistences(df)
    df = check_null_values(df)
    save_processed_data(df, processed_path)

    print("="*15)
    print("PIPELINE EXECUTED SUCCESSFULLY")
    print("="*15)

if __name__ == "__main__":
    run_pipeline()