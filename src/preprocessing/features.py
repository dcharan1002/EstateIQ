import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from .core import identify_column_types

def standardize_condition(df):
    """Standardize condition values by extracting descriptions after the hyphen"""
    df = df.copy()
    
    if 'OVERALL_COND' in df.columns:
        # Extract everything before and after hyphen
        df['OVERALL_COND'] = df['OVERALL_COND'].apply(lambda x: 
            x.split('-')[1].strip() if isinstance(x, str) and '-' in x 
            else 'Average')  # Default to Average if format is unexpected
        
        # Map common variations to standard values
        condition_map = {
            'FAIR': 'Fair',
            'GOOD': 'Good',
            'AVERAGE': 'Average',
            'AVG': 'Average',
            'DEFAULT': 'Average'
        }
        df['OVERALL_COND'] = df['OVERALL_COND'].str.upper().map(condition_map).fillna('Average')
    
    return df

def encode_categorical_features(df, max_categories=10, existing_encoders=None):
    """Encode categorical features with domain awareness"""
    df = df.copy()
    
    # Identify string columns to preserve
    string_columns = [col for col in df.columns if any(pattern in col.lower() 
                     for pattern in ['zip', 'postal', 'phone', 'id', 'code', 'placekey'])]
    
    # Get remaining categorical columns
    column_types = identify_column_types(df)
    categorical_cols = [col for col in column_types['categorical'] 
                       if col not in string_columns]
    
    encoders = existing_encoders or {}
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        
        # Special handling for property-specific categories
        if any(x in col.lower() for x in ['condition', 'quality', 'grade']):
            quality_map = {
                'POOR': 1, 
                'FAIR': 2, 
                'AVERAGE': 3, 
                'GOOD': 4, 
                'EXCELLENT': 5
            }
            df[col] = df[col].map(lambda x: quality_map.get(str(x).upper(), 3))
            
        elif any(x in col.lower() for x in ['style', 'type', 'struct']):
            if df[col].nunique() <= max_categories:
                if col in encoders:
                    # Use existing encoder
                    if isinstance(encoders[col], OneHotEncoder):
                        encoded = encoders[col].transform(df[[col]])
                        encoded_cols = [f"{col}_{cat}" for cat in encoders[col].categories_[0]]
                        df[encoded_cols] = encoded
                        df.drop(columns=[col], inplace=True)
                    else:
                        df[col] = encoders[col].transform(df[col])
                else:
                    # Create new encoder
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[col]])
                    encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    df[encoded_cols] = encoded
                    df.drop(columns=[col], inplace=True)
                    encoders[col] = encoder
            else:
                if col in encoders:
                    # Use existing encoder with fallback for unseen labels
                    known_labels = set(encoders[col].classes_)
                    df[col] = df[col].map(lambda x: x if x in known_labels else 'UNKNOWN')
                    df[col] = encoders[col].transform(df[col])
                else:
                    # Create new encoder with UNKNOWN category
                    unique_vals = df[col].unique()
                    encoder = LabelEncoder()
                    # Add UNKNOWN to ensure it's part of the classes
                    all_vals = np.append(unique_vals, 'UNKNOWN')
                    encoder.fit(all_vals)
                    df[col] = encoder.transform(df[col])
                    encoders[col] = encoder
        else:
            if df[col].nunique() <= max_categories:
                if col in encoders:
                    # Use existing encoder
                    if isinstance(encoders[col], OneHotEncoder):
                        encoded = encoders[col].transform(df[[col]])
                        encoded_cols = [f"{col}_{cat}" for cat in encoders[col].categories_[0]]
                        df[encoded_cols] = encoded
                        df.drop(columns=[col], inplace=True)
                    else:
                        df[col] = encoders[col].transform(df[col])
                else:
                    # Create new encoder
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[col]])
                    encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    df[encoded_cols] = encoded
                    df.drop(columns=[col], inplace=True)
                    encoders[col] = encoder
            else:
                if col in encoders:
                    # Use existing encoder with fallback for unseen labels
                    known_labels = set(encoders[col].classes_)
                    df[col] = df[col].map(lambda x: x if x in known_labels else 'UNKNOWN')
                    df[col] = encoders[col].transform(df[col])
                else:
                    # Create new encoder with UNKNOWN category
                    unique_vals = df[col].unique()
                    encoder = LabelEncoder()
                    # Add UNKNOWN to ensure it's part of the classes
                    all_vals = np.append(unique_vals, 'UNKNOWN')
                    encoder.fit(all_vals)
                    df[col] = encoder.transform(df[col])
                    encoders[col] = encoder
    
    # Preserve string columns as-is
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df, encoders

def create_ratio_features(df):
    """Create ratio features from numeric columns"""
    df = df.copy()
    
    # Ensure all columns being used are numeric
    if 'TOTAL_VALUE' in df.columns and 'GROSS_AREA' in df.columns:
        if pd.to_numeric(df['TOTAL_VALUE'], errors='coerce').notna().all() and \
           pd.to_numeric(df['GROSS_AREA'], errors='coerce').notna().all():
            df['price_per_sqft'] = pd.to_numeric(df['TOTAL_VALUE']) / pd.to_numeric(df['GROSS_AREA'])
    
    if all(col in df.columns for col in ['LAND_VALUE', 'TOTAL_VALUE']):
        if pd.to_numeric(df['LAND_VALUE'], errors='coerce').notna().all() and \
           pd.to_numeric(df['TOTAL_VALUE'], errors='coerce').notna().all():
            df['land_value_ratio'] = pd.to_numeric(df['LAND_VALUE']) / pd.to_numeric(df['TOTAL_VALUE'])
    
    if all(col in df.columns for col in ['BLDG_VALUE', 'TOTAL_VALUE']):
        if pd.to_numeric(df['BLDG_VALUE'], errors='coerce').notna().all() and \
           pd.to_numeric(df['TOTAL_VALUE'], errors='coerce').notna().all():
            df['building_value_ratio'] = pd.to_numeric(df['BLDG_VALUE']) / pd.to_numeric(df['TOTAL_VALUE'])
    
    return df

def create_property_features(df):
    """Create additional property features"""
    df = df.copy()
    
    # Standardize condition values
    df = standardize_condition(df)
    
    if 'YR_BUILT' in df.columns:
        current_year = pd.Timestamp.now().year
        df['property_age'] = current_year - pd.to_numeric(df['YR_BUILT'], errors='coerce')
        
        df['age_category'] = pd.cut(
            df['property_age'],
            bins=[-float('inf'), 5, 10, 20, 30, 50, float('inf')],
            labels=['New', '5-10 years', '10-20 years', '20-30 years', '30-50 years', '50+ years']
        )
    
    # Only create features if columns exist and are numeric
    if all(col in df.columns for col in ['FULL_BTH', 'HLF_BTH']):
        full_bath = pd.to_numeric(df['FULL_BTH'], errors='coerce').round().astype('Int64')
        half_bath = pd.to_numeric(df['HLF_BTH'], errors='coerce').round().astype('Int64')
        if full_bath.notna().all() and half_bath.notna().all():
            # Store both total and fractional bathrooms
            df['total_bathrooms_int'] = full_bath + half_bath  # Integer total
            df['total_bathrooms'] = full_bath + 0.5 * half_bath  # Keep fractional for analysis
    
    if all(col in df.columns for col in ['GROSS_AREA', 'LIVING_AREA']):
        gross_area = pd.to_numeric(df['GROSS_AREA'], errors='coerce')
        living_area = pd.to_numeric(df['LIVING_AREA'], errors='coerce')
        if gross_area.notna().all() and living_area.notna().all():
            df['non_living_area'] = gross_area - living_area
            df['living_area_ratio'] = living_area / gross_area
    
    if all(col in df.columns for col in ['YR_REMODEL', 'YR_BUILT']):
        df['has_renovation'] = pd.to_numeric(df['YR_REMODEL'], errors='coerce') > \
                             pd.to_numeric(df['YR_BUILT'], errors='coerce')
        df['years_since_renovation'] = current_year - pd.to_numeric(df['YR_REMODEL'], errors='coerce')
    
    return df
