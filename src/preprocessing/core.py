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
            
        # Check for address components
        if any(addr in column.lower() for addr in ['address', 'street', 'city', 'state', 'zip']):
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

def extract_address_components(address_series):
    components = pd.DataFrame()
    
    # Extract house numbers
    components['house_number'] = address_series.str.extract(r'^(\d+)')
    
    # Extract street names
    components['street_name'] = address_series.str.extract(r'\d+\s+(.*?)\s*(?:,|$)')
    
    # Extract unit numbers
    components['unit_number'] = address_series.str.extract(r'(?:UNIT|APT|#)\s*(\w+)')
    
    return components

def create_spatial_features(df, lat_col, lon_col):
    df = df.copy()
    
    # Calculate distance from city center (example coordinates for demonstration)
    city_center = {'lat': 42.3601, 'lon': -71.0589}  # Boston coordinates
    
    def haversine_distance(row):
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1 = np.radians([row[lat_col], row[lon_col]])
        lat2, lon2 = np.radians([city_center['lat'], city_center['lon']])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    df['distance_to_center'] = df.apply(haversine_distance, axis=1)
    
    return df

def calculate_property_age(df, year_built_col, reference_year=None):
    df = df.copy()
    
    if reference_year is None:
        reference_year = datetime.now().year
        
    df['property_age'] = reference_year - df[year_built_col]
    
    # Age buckets
    df['age_category'] = pd.cut(
        df['property_age'],
        bins=[-float('inf'), 5, 10, 20, 30, 50, float('inf')],
        labels=['New', '5-10 years', '10-20 years', '20-30 years', '30-50 years', '50+ years']
    )
    
    return df

def save_processed_data(X_train, X_test, y_train, y_test, output_dir="/sources/data/processed"):
    pd.DataFrame(X_train).to_csv(f"{output_dir}/X_train_clean.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{output_dir}/X_test_clean.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train_clean.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{output_dir}/y_test_clean.csv", index=False)

def split_features_target(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found in dataframe")
    return df.drop(columns=[target_column]), df[target_column]
