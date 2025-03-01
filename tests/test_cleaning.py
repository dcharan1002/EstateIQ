import pytest
import pandas as pd
import numpy as np
from src.preprocessing.cleaning import (
    clean_address_data,
    clean_numeric_columns,
    standardize_categorical_values,
    handle_missing_values,
    handle_outliers,
    remove_duplicate_rows
)

def test_clean_address_data_basic():
    # Test data
    df = pd.DataFrame({
        'ADDRESS': ['123 MAIN ST', 'APT 2B ON 456 OAK RD', '789 E MAPLE AVE']
    })
    
    result = clean_address_data(df, ['ADDRESS'])
    
    assert all(addr == addr.upper() for addr in result['ADDRESS'])
    assert 'STREET' in result['ADDRESS'].iloc[0]
    assert 'ROAD' in result['ADDRESS'].iloc[1]
    assert 'AVENUE' in result['ADDRESS'].iloc[2]

@pytest.fixture
def sample_numeric_df():
    return pd.DataFrame({
        'price': ['$100,000', '200000', 'invalid'],
        'area': ['1,500', '2000', '-500'],
        'year': ['2020', '1999', 'unknown'],
        'rooms': ['3', '4.5', 'invalid'],
        'condition': ['1', '3', '6'],
        'zip_code': ['02108', '02109', '02110']
    })

def test_standardize_categorical_values_basic():
    df = pd.DataFrame({
        'condition': ['EXCELLENT', 'good', 'Fair', 'POOR'],
        'style': ['colonial', 'RANCH', 'Split', 'Cape']
    })
    
    result = standardize_categorical_values(df, ['condition', 'style'])
    
    # Check standardization
    assert all(isinstance(val, str) and val.isupper() for val in result['condition'])
    assert all(isinstance(val, str) and val.isupper() for val in result['style'])

@pytest.fixture
def sample_missing_df():
    return pd.DataFrame({
        'numeric': [1.0, np.nan, 3.0, np.nan, 5.0],
        'integer': [1, np.nan, 3, np.nan, 5],
        'categorical': ['A', 'B', 'B', 'A', None],
        'boolean': [True, False, False, True, None],
        'year': [2000.0, 2020.0, 2020.0, np.nan, 2010.0]
    })

def test_handle_outliers_basic():
    df = pd.DataFrame({
        'price': [100000, 200000, 1000000, 50000000],  # Outlier in last value
        'area': [1500, 2000, 2500, 3000]
    })
    
    result = handle_outliers(df)
    
    # Check that extreme outliers were handled
    assert result['price'].max() < 50000000
    assert not result['area'].isna().any()  # No outliers in area

def test_remove_duplicate_rows_basic():
    df = pd.DataFrame({
        'A': [1, 2, 1, 3],
        'B': ['a', 'b', 'a', 'c']
    })
    
    result = remove_duplicate_rows(df)
    assert len(result) == 3  # Only unique rows

def test_remove_duplicate_rows_subset():
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'value': ['a', 'a', 'b', 'b']
    })
    
    result = remove_duplicate_rows(df, consider_cols=['value'])
    assert len(result) == 2  # Only unique values in 'value' column
