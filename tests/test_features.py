import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.preprocessing.features import (
    standardize_condition,
    encode_categorical_features,
    create_ratio_features,
    create_property_features
)

@pytest.fixture
def sample_condition_df():
    return pd.DataFrame({
        'OVERALL_COND': [
            '1-POOR',
            '2-FAIR', 
            '3-AVERAGE',
            '4-GOOD',
            '5-EXCELLENT',
            'AVG',  # Edge case
            None,   # Null value
            '3-'    # Malformed
        ]
    })

@pytest.fixture
def sample_categorical_df():
    # Create with consistent lengths
    categories = ['cat_' + str(i) for i in range(4)]  # Match length with other columns
    return pd.DataFrame({
        'property_type': ['Single Family', 'Condo', 'Multi Family', 'Single Family'],
        'condition': ['Good', 'Fair', 'Average', 'Poor'],
        'zip_code': ['02108', '02109', '02110', '02111'],
        'many_categories': categories,  # Now matches length of other columns
        'property_id': ['A1', 'A2', 'A3', 'A4']
    })

@pytest.fixture
def sample_numeric_df():
    return pd.DataFrame({
        'TOTAL_VALUE': [500000, 600000, 700000, 800000],
        'GROSS_AREA': [2000, 2400, 2800, 3200],
        'LAND_VALUE': [200000, 240000, 280000, 320000],
        'BLDG_VALUE': [300000, 360000, 420000, 480000]
    })

@pytest.fixture
def sample_property_df():
    return pd.DataFrame({
        'YR_BUILT': [2000, 1990, 1980, 1970],
        'YR_REMODEL': [2010, 1990, 2000, 1970],
        'FULL_BTH': [2, 1, 3, 2],
        'HLF_BTH': [1, 1, 0, 1],
        'GROSS_AREA': [2000, 1800, 2500, 2200],
        'LIVING_AREA': [1800, 1600, 2200, 1900],
        'OVERALL_COND': ['3-AVERAGE', '2-FAIR', '4-GOOD', '1-POOR']
    })

def test_standardize_condition_basic(sample_condition_df):
    result = standardize_condition(sample_condition_df)
    
    # Check basic mappings
    assert result['OVERALL_COND'].iloc[0] == 'Poor'  # 1-POOR -> Poor
    assert result['OVERALL_COND'].iloc[1] == 'Fair'  # 2-FAIR -> Fair
    assert result['OVERALL_COND'].iloc[2] == 'Average'  # 3-AVERAGE -> Average
    assert result['OVERALL_COND'].iloc[3] == 'Good'  # 4-GOOD -> Good
    
    # Check edge cases
    assert result['OVERALL_COND'].iloc[5] == 'Average'  # AVG -> Average
    assert result['OVERALL_COND'].iloc[6] == 'Average'  # None -> Average
    assert result['OVERALL_COND'].iloc[7] == 'Average'  # Malformed -> Average

def test_encode_categorical_features_reuse_encoders(sample_categorical_df):
    # First encoding
    result, encoders = encode_categorical_features(sample_categorical_df)
    
    # New data with mix of known and unknown categories
    new_df = pd.DataFrame({
        'property_type': ['Single Family', 'Unknown Type', 'Multi Family'],
        'condition': ['Excellent', 'Unknown', 'Good'],
        'zip_code': ['02112', '02113', '02114'],
        'many_categories': ['cat_0', 'new_cat_1', 'cat_1'],  # Include some known categories
        'property_id': ['A5', 'A6', 'A7']
    })
    
    result, _ = encode_categorical_features(new_df, existing_encoders=encoders)
    
    # Check that known categories are preserved
    assert any(col.startswith('many_categories_CAT_0') for col in result.columns)
    assert any(col.startswith('many_categories_CAT_1') for col in result.columns)
    
    # Check encoding is present for known categories
    many_cat_cols = [col for col in result.columns if col.startswith('many_categories_')]
    assert len(many_cat_cols) > 0
    assert result[many_cat_cols].sum().sum() > 0  # At least some categories are encoded

def test_create_ratio_features_basic(sample_numeric_df):
    result = create_ratio_features(sample_numeric_df)
    
    # Check created features
    assert 'price_per_sqft' in result.columns
    assert 'land_value_ratio' in result.columns
    assert 'building_value_ratio' in result.columns
    
    # Verify calculations
    np.testing.assert_array_almost_equal(
        result['price_per_sqft'],
        sample_numeric_df['TOTAL_VALUE'] / sample_numeric_df['GROSS_AREA']
    )

def test_create_ratio_features_invalid_data():
    df = pd.DataFrame({
        'TOTAL_VALUE': [500000, 600000],
        'GROSS_AREA': [2000, 2400],
        'LAND_VALUE': [200000, 240000],
        'BLDG_VALUE': [300000, 360000]
    })
    result = create_ratio_features(df)
    
    # Should create valid ratio features
    assert 'price_per_sqft' in result.columns
    assert not result['price_per_sqft'].isna().any()

def test_create_property_features_basic(sample_property_df):
    result = create_property_features(sample_property_df)
    
    # Check created features
    assert 'property_age' in result.columns
    assert 'age_category' in result.columns
    assert 'total_bathrooms' in result.columns
    assert 'living_area_ratio' in result.columns
    
    # Check values are within expected ranges
    assert result['property_age'].min() >= 0
    assert result['total_bathrooms'].max() <= (sample_property_df['FULL_BTH'].max() + 
                                             0.5 * sample_property_df['HLF_BTH'].max())
