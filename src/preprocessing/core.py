import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from datetime import datetime

def identify_column_types(df):
    column_types = {
        'categorical': [],
        'numerical': [],
        'datetime': [],
        'address': [],
        'coordinates': [],
        'identifier': [],
        'text': [],
        'boolean': []
    }
    
    for column in df.columns:
        # Check for identifiers
        if any(id_pattern in column.lower() for id_pattern in ['id', 'code', 'pid', 'upi']):
            column_types['identifier'].append(column)
            continue
            
        # Check for coordinates
        if any(coord in column.lower() for coord in ['latitude', 'longitude', 'coord']):
            column_types['coordinates'].append(column)
            continue
            
        # Check for address components (including zip code check)
        col_lower = column.lower()
        if any(addr in col_lower for addr in ['address', 'street', 'city', 'state']) or \
           ('zip' in col_lower and (df[column].dtype == 'object' or df[column].astype(str).str.len().mean() == 5)):
            column_types['address'].append(column)
            continue
            
        # Check data type
        dtype = df[column].dtype
        if dtype in ['int64', 'float64']:
            # Check if it's actually a boolean
            if set(df[column].dropna().unique()).issubset({0, 1}):
                column_types['boolean'].append(column)
            # Check if it's a condition column (should be numeric 1-5)
            elif 'cond' in column.lower() and set(df[column].dropna().unique()).issubset({1, 2, 3, 4, 5}):
                column_types['numerical'].append(column)
            else:
                column_types['numerical'].append(column)
        elif dtype == 'object':
            # Check if it's a date
            try:
                pd.to_datetime(df[column].dropna().iloc[0])
                column_types['datetime'].append(column)
            except:
                # Check if it's a text field
                if df[column].str.len().mean() > 50:
                    column_types['text'].append(column)
                else:
                    column_types['categorical'].append(column)
        elif dtype == 'bool':
            column_types['boolean'].append(column)
        elif dtype in ['datetime64[ns]', 'datetime64']:
            column_types['datetime'].append(column)
        else:
            column_types['categorical'].append(column)
            
    return column_types

def save_processed_data(X_train, X_test, y_train, y_test, output_dir="/sources/data/processed"):
    pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train_clean.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test_clean.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train_clean.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{output_dir}/y_test_clean.csv", index=False)

def split_features_target(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found in dataframe")
    return df.drop(columns=[target_column]), df[target_column]
