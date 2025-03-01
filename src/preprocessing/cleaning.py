import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import re
import logging
from .core import identify_column_types

# Configure logger
logger = logging.getLogger(__name__)

def clean_address_data(df, address_cols):
    logger.info(f"Starting address data cleaning for columns: {address_cols}")
    df = df.copy()
    
    for col in address_cols:
        if col in df.columns:
            # Convert to uppercase for consistency
            df[col] = df[col].str.upper()
            
            # Remove extra whitespace
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ')
            
            # Standardize common abbreviations
            replacements = {
                r'\bAPT\b': 'APARTMENT',
                r'\bST\b': 'STREET',
                r'\bRD\b': 'ROAD',
                r'\bAVE\b': 'AVENUE',
                r'\bBLVD\b': 'BOULEVARD',
                r'\bN\b': 'NORTH',
                r'\bS\b': 'SOUTH',
                r'\bE\b': 'EAST',
                r'\bW\b': 'WEST'
            }
            
            for pattern, replacement in replacements.items():
                df[col] = df[col].str.replace(pattern, replacement, regex=True)
    
    return df

def ensure_string_columns(df, columns):
    logger.debug(f"Converting columns to string type: {columns}")
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def clean_numeric_columns(df, exclude_patterns=None):
    logger.info("Starting numeric column cleaning")
    df = df.copy()
    
    # Default patterns to exclude from numeric conversion
    if exclude_patterns is None:
        exclude_patterns = [
            'zip', 'postal', 'phone', 'id', 'code', 'placekey'
        ]
    logger.debug(f"Excluding patterns from numeric conversion: {exclude_patterns}")
    
    column_types = identify_column_types(df)
    numeric_cols = column_types['numerical']
    
    for col in df.columns:
        # Skip columns matching exclude patterns
        if any(pattern in col.lower() for pattern in exclude_patterns):
            continue
            
        if col in numeric_cols:
            # Remove currency symbols and commas
            df[col] = df[col].astype(str).str.replace('$', '')
            df[col] = df[col].str.replace(',', '')
            
            # Convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert appropriate columns to integers
            integer_patterns = [
                'bed', 'bath', 'room', 'floor', 'parking', 'garage', 
                'fireplaces', 'kitchens', 'num_', 'yr_', 'year', 
                'rc_units', 'res_units', 'com_units', 'unit_num', 'st_num',
                'cond'  # For condition-related columns
            ]
            # Special handling for condition columns and other integers
            if 'cond' in col.lower():
                # For condition columns, ensure values are between 1 and 5
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].clip(1, 5).round().astype('Int64')
            elif any(x in col.lower() for x in integer_patterns):
                df[col] = df[col].round().astype('Int64')
            
            # Remove obvious errors
            if any(x in col.lower() for x in ['area', 'sqft', 'feet', 'sf']):
                invalid_count = (df[col] <= 0).sum()
                if invalid_count > 0:
                    logger.warning(f"Found {invalid_count} invalid values (<= 0) in {col}")
                df.loc[df[col] <= 0, col] = np.nan
            elif any(x in col.lower() for x in ['price', 'value', 'tax']):
                invalid_count = (df[col] < 0).sum()
                if invalid_count > 0:
                    logger.warning(f"Found {invalid_count} negative values in {col}")
                df.loc[df[col] < 0, col] = np.nan
    
    # Ensure zip codes and IDs remain as strings
    string_columns = [col for col in df.columns if any(pattern in col.lower() 
                     for pattern in ['zip', 'postal', 'id', 'code'])]
    df = ensure_string_columns(df, string_columns)
    
    return df

def standardize_categorical_values(df, categorical_cols):
    logger.info(f"Standardizing categorical values for columns: {categorical_cols}")
    df = df.copy()
    
    for col in categorical_cols:
        if col in df.columns:
            # Keep as string and standardize
            df[col] = df[col].astype(str).str.upper().str.strip()
            
            # Group similar values
            if 'CONDITION' in col.upper():
                condition_map = {
                    'EXCELLENT': 'EXCELLENT',
                    'VERY GOOD': 'VERY GOOD',
                    'GOOD': 'GOOD',
                    'FAIR': 'FAIR',
                    'POOR': 'POOR'
                }
                df[col] = df[col].map(lambda x: next((v for k, v in condition_map.items() 
                                                     if k in str(x).upper()), x))
            
            elif 'STYLE' in col.upper():
                style_map = {
                    'COLONIAL': 'COLONIAL',
                    'RANCH': 'RANCH',
                    'CAPE': 'CAPE COD',
                    'CAPE COD': 'CAPE COD',
                    'CONTEMPORARY': 'CONTEMPORARY',
                    'SPLIT': 'SPLIT LEVEL',
                    'SPLIT LEVEL': 'SPLIT LEVEL'
                }
                df[col] = df[col].map(lambda x: next((v for k, v in style_map.items() 
                                                     if k in str(x).upper()), x))
    
    return df

def handle_missing_values(df, numerical_strategy="median", categorical_strategy="most_frequent"):
    logger.info(f"Handling missing values with strategies - numerical: {numerical_strategy}, categorical: {categorical_strategy}")
    df = df.copy()
    column_types = identify_column_types(df)
    
    # Handle numerical missing values
    numerical_cols = column_types['numerical']
    if numerical_cols:
        logger.debug(f"Processing {len(numerical_cols)} numerical columns")
        for col in numerical_cols:
            # Skip certain columns
            if any(x in col.lower() for x in ['zip', 'postal', 'id', 'code']):
                continue
            # Use column-specific imputation strategy
            if any(x in col.lower() for x in ['year', 'yr']):
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals.iloc[0])
                else:
                    logger.warning(f"No mode found for column {col}, using median imputation")
                    imputer = SimpleImputer(strategy=numerical_strategy)
                    df[[col]] = imputer.fit_transform(df[[col]])
            else:
                imputer = SimpleImputer(strategy=numerical_strategy)
                df[[col]] = imputer.fit_transform(df[[col]])
    
    # Handle categorical missing values
    categorical_cols = column_types['categorical']
    if categorical_cols:
        if categorical_strategy == "most_frequent":
            for col in categorical_cols:
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals.iloc[0])
                else:
                    logger.warning(f"No mode found for column {col}, using constant imputation")
                    df[col] = df[col].fillna('UNKNOWN')
        else:
            imputer = SimpleImputer(strategy=categorical_strategy)
            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
    
    # Handle boolean missing values
    boolean_cols = column_types['boolean']
    for col in boolean_cols:
        df[col] = df[col].fillna(False)
    
    return df

def handle_outliers(df, method="iqr", threshold=1.5):
    """
    Handle outliers in numerical columns using a memory-efficient approach.
    
    Args:
        df: Input DataFrame
        method: 'iqr' or 'zscore'
        threshold: Multiplier for IQR or standard deviation
    """
    logger.info(f"Detecting outliers using method: {method} with threshold: {threshold}")
    start_time = pd.Timestamp.now()
    
    df = df.copy()
    column_types = identify_column_types(df)
    numerical_cols = [col for col in column_types['numerical'] 
                     if not any(x in col.lower() for x in ['id', 'code', 'year', 'zip'])]
    
    if not numerical_cols:
        logger.info("No numerical columns to process for outliers")
        return df
    
    # Group columns by type for targeted processing
    price_cols = [col for col in numerical_cols if any(x in col.lower() for x in ['price', 'value'])]
    area_cols = [col for col in numerical_cols if any(x in col.lower() for x in ['area', 'sqft', 'feet', 'sf'])]
    other_cols = [col for col in numerical_cols if col not in price_cols + area_cols]
    
    # Process each column type
    for col in numerical_cols:
        # Calculate bounds based on method
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        else:  # zscore
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
        # Apply bounds based on column type
        if col in price_cols:
            # Only cap upper bound for prices
            outliers = (df[col] > upper_bound).sum()
            if outliers > 0:
                logger.warning(f"Found {outliers} upper outliers in {col}")
            df[col] = df[col].clip(upper=upper_bound)
            
        elif col in area_cols:
            # Apply both bounds with adjusted upper for area
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound * 1.5)
            
        else:
            # Standard bounds for other columns
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    duration = (pd.Timestamp.now() - start_time).total_seconds()
    logger.info(f"Completed outlier detection and handling in {duration:.2f} seconds")
    return df

def remove_duplicate_rows(df, consider_cols=None):
    logger.info("Removing duplicate rows" + (f" considering columns: {consider_cols}" if consider_cols else ""))
    initial_rows = len(df)
    
    if consider_cols:
        result = df.drop_duplicates(subset=consider_cols).reset_index(drop=True)
    else:
        result = df.drop_duplicates().reset_index(drop=True)
    
    removed_rows = initial_rows - len(result)
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} duplicate rows")
    
    return result
