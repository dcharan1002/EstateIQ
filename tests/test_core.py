import pytest
import pandas as pd
import numpy as np
from src.preprocessing.core import (
    identify_column_types, split_features_target, save_processed_data,
    extract_address_components, create_spatial_features, calculate_property_age
)
from pathlib import Path
import tempfile

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [1.1, 2.2, 3.3],
        'cat1': ['a', 'b', 'c'],
        'cat2': ['x', 'y', 'z'],
        'target': [0, 1, 0],
        'property_id': ['A1', 'A2', 'A3'],
        'address': ['123 Main St', '456 Oak Ave #2B', '789 Pine Rd Unit 3'],
        'latitude': [42.1, 42.2, 42.3],
        'longitude': [-71.1, -71.2, -71.3],
        'description': ['Large house with view', 'Cozy apartment', 'Modern condo'],
        'is_renovated': [True, False, True],
        'last_sale': ['2020-01-01', '2021-02-02', '2022-03-03'],
        'year_built': [1990, 2000, 2010]
    })

def test_identify_column_types_basic(sample_df):
    column_types = identify_column_types(sample_df)
    assert set(column_types['categorical']) == {'cat1', 'cat2'}
    assert set(column_types['numerical']) == {'num1', 'num2', 'target'}
    assert set(column_types['identifier']) == {'property_id'}
    assert set(column_types['coordinates']) == {'latitude', 'longitude'}
    assert set(column_types['address']) == {'address'}
    assert set(column_types['text']) == {'description'}
    assert set(column_types['boolean']) == {'is_renovated'}
    assert set(column_types['datetime']) == {'last_sale'}

def test_identify_column_types_edge_cases():
    df = pd.DataFrame({
        'mixed_ints': ['1', 2, 3],                    # Mixed string/int numbers
        'all_null': [None, None, np.nan],             # All null values
        'bool_like': [0, 1, 1],                       # Numbers that look like booleans
        'short_id': ['A1', 'B2', 'C3'],              # Short identifier-like strings
        'empty_text': ['', ' ', '   '],              # Empty/whitespace strings
        'mixed_dates': ['2020-01-01', 'invalid', ''], # Mixed dates and invalid values
        'condition_nums': [1, 3, 5],                  # Numbers that could be conditions
        'long_text': ['a' * 100, 'b' * 100, 'c' * 100],  # Long text fields
        'coord_like': ['42.123, -71.123', '42.124, -71.124', '42.125, -71.125']  # Coordinate-like strings
    })
    
    column_types = identify_column_types(df)
    
    # Check specific edge cases
    assert 'mixed_ints' in column_types['numerical']  # Should detect as numerical
    assert 'bool_like' in column_types['boolean']     # Should detect as boolean
    assert 'short_id' in column_types['categorical']  # Should not be identified as ID
    assert 'empty_text' in column_types['categorical']  # Empty strings are categorical
    assert 'condition_nums' in column_types['numerical']  # Numbers 1-5 could be conditions

def test_identify_column_types_validation():
    # Test empty dataframe
    assert all(not types for types in identify_column_types(pd.DataFrame()).values())
    
    # Test single column dataframes
    assert 'col' in identify_column_types(pd.DataFrame({'col': [1, 2, 3]}))['numerical']
    assert 'col' in identify_column_types(pd.DataFrame({'col': ['a', 'b', 'c']}))['categorical']

def test_split_features_target(sample_df):
    X, y = split_features_target(sample_df, 'target')
    assert 'target' not in X.columns
    assert isinstance(y, pd.Series)
    assert len(y) == len(sample_df)
    assert all(y == sample_df['target'])

def test_split_features_target_invalid_target():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    with pytest.raises(ValueError):
        split_features_target(df, 'nonexistent')

def test_extract_address_components_malformed():
    addresses = pd.Series([
        '123Main St',  # No space after number
        'Apt#5B',     # No space before unit
        ' 456 Oak ',  # Extra spaces
        '789-A Pine', # Hyphenated number
        'Unit 7C',    # No street number
        '10th Street' # Ordinal number
    ])
    components = extract_address_components(addresses)
    
    assert components['house_number'].notna().sum() >= 3  # Should find at least 123, 456, 789
    assert components['unit_number'].notna().sum() >= 2   # Should find 5B, 7C
    assert all(isinstance(x, str) for x in components['street_name'].dropna())

def test_extract_address_components_basic():
    addresses = pd.Series([
        '123 Main St',
        '456 Oak Ave #2B',
        '789 Pine Road Unit 3',
        'Invalid Address'
    ])
    components = extract_address_components(addresses)
    
    assert list(components['house_number'].dropna()) == ['123', '456', '789']
    assert list(components['street_name'].dropna()) == ['Main St', 'Oak Ave', 'Pine Road']
    assert list(components['unit_number'].dropna()) == ['2B', '3']

def test_extract_address_components_edge_cases():
    addresses = pd.Series([
        '',  # Empty string
        '123',  # Just number
        'Main Street',  # Just street
        'APT 5B',  # Just unit
        None,  # None value
        '123-456 Complex Street Unit 7C'  # Complex address
    ])
    components = extract_address_components(addresses)
    
    assert components['house_number'].notna().sum() == 2  # Should find 123 and 123-456
    assert components['street_name'].notna().sum() == 2  # Should find 'Main Street' and 'Complex Street'
    assert components['unit_number'].notna().sum() == 2  # Should find '5B' and '7C'

def test_create_spatial_features_basic():
    df = pd.DataFrame({
        'latitude': [42.3601, 42.4, 42.5],
        'longitude': [-71.0589, -71.1, -71.2]
    })
    result = create_spatial_features(df, 'latitude', 'longitude')
    
    assert 'distance_to_center' in result.columns
    assert len(result['distance_to_center']) == len(df)
    assert result['distance_to_center'][0] == pytest.approx(0, abs=0.1)  # First row is city center
    assert all(result['distance_to_center'] >= 0)  # All distances should be positive

def test_create_spatial_features_edge_cases():
    # Test edge case coordinates and invalid values
    df = pd.DataFrame({
        'latitude': [90, -90, 0, None, 91, -91, 'invalid'],  # Including invalid values
        'longitude': [180, -180, 0, None, 181, -181, 'invalid']  # Including invalid values
    })
    result = create_spatial_features(df, 'latitude', 'longitude')
    
    assert result['distance_to_center'].notna().sum() == 3  # Only valid coordinates should have results
    assert pd.isna(result['distance_to_center'].iloc[3:]).all()  # Invalid values should be NaN

def test_identify_column_types_empty_and_null():
    # Test with empty and all-null columns
    df = pd.DataFrame({
        'empty_col': [],
        'null_col': [None, None],
        'nan_col': [np.nan, np.nan],
        'mixed_null': [None, np.nan, pd.NA]
    })
    column_types = identify_column_types(df)
    
    # All columns should be categorized somehow
    all_columns = set(df.columns)
    categorized_columns = set()
    for type_cols in column_types.values():
        categorized_columns.update(type_cols)
    assert all_columns == categorized_columns

def test_create_spatial_features_type_validation():
    # Test with wrong column types
    df = pd.DataFrame({
        'latitude': ['not_a_number', '42.1', '42.2'],
        'longitude': [-71.1, -71.2, -71.3]
    })
    with pytest.raises(ValueError, match="Invalid coordinate values"):
        create_spatial_features(df, 'latitude', 'longitude')

    # Test with columns having mixed types
    df = pd.DataFrame({
        'latitude': [42.1, None, 'invalid'],
        'longitude': [-71.1, -71.2, -71.3]
    })
    result = create_spatial_features(df, 'latitude', 'longitude')
    assert pd.isna(result['distance_to_center'].iloc[1:]).all()
    assert not pd.isna(result['distance_to_center'].iloc[0])

def test_create_spatial_features_invalid_columns():
    # Test with missing columns
    df = pd.DataFrame({'invalid_col': [1, 2, 3]})
    with pytest.raises(KeyError):
        create_spatial_features(df, 'latitude', 'longitude')
    
    # Test with wrong column names
    df = pd.DataFrame({
        'lat': [42.1, 42.2, 42.3],
        'lon': [-71.1, -71.2, -71.3]
    })
    with pytest.raises(KeyError):
        create_spatial_features(df, 'latitude', 'longitude')

def test_calculate_property_age_basic():
    df = pd.DataFrame({
        'year_built': [1990, 2000, 2010, 2020]
    })
    reference_year = 2025
    
    result = calculate_property_age(df, 'year_built', reference_year)
    
    assert 'property_age' in result.columns
    assert 'age_category' in result.columns
    assert list(result['property_age']) == [35, 25, 15, 5]
    assert all(result['age_category'].isin([
        'New', '5-10 years', '10-20 years', '20-30 years', '30-50 years', '50+ years'
    ]))

def test_calculate_property_age_edge_cases():
    df = pd.DataFrame({
        'year_built': [
            None,  # Missing value
            1800,  # Very old
            2050,  # Future date
            0,    # Invalid year
            'invalid',  # Non-numeric
            999999,  # Unreasonable value
            -1000   # Negative year
        ]
    })
    reference_year = 2025
    
    result = calculate_property_age(df, 'year_built', reference_year)
    
    # Check handling of various edge cases
    assert pd.isna(result['property_age'].iloc[0])  # None should be NaN
    assert result['property_age'].iloc[1] == 225  # 2025 - 1800
    assert result['property_age'].iloc[2] == -25  # Future year
    assert result['property_age'].iloc[3] == 2025  # Zero year
    assert pd.isna(result['property_age'].iloc[4])  # Invalid string
    assert result['property_age'].iloc[5] == -997974  # Large year
    assert result['property_age'].iloc[6] == 3025  # Negative year

def test_calculate_property_age_invalid_reference():
    df = pd.DataFrame({'year_built': [2000]})
    
    # Test invalid reference years
    with pytest.raises(ValueError):
        calculate_property_age(df, 'year_built', reference_year='invalid')
    with pytest.raises(ValueError):
        calculate_property_age(df, 'year_built', reference_year=-1)

def test_save_processed_data_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data with various data types
        X_train = pd.DataFrame({
            'numeric': [1, 2],
            'categorical': ['A', 'B'],
            'boolean': [True, False],
            'float': [1.1, 2.2]
        })
        X_test = pd.DataFrame({
            'numeric': [3, 4],
            'categorical': ['C', 'D'],
            'boolean': [False, True],
            'float': [3.3, 4.4]
        })
        y_train = pd.Series([0, 1])
        y_test = pd.Series([1, 0])
        
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=tmpdir)
        
        # Verify files exist
        assert Path(f"{tmpdir}/X_train_clean.csv").exists()
        assert Path(f"{tmpdir}/X_test_clean.csv").exists()
        assert Path(f"{tmpdir}/y_train_clean.csv").exists()
        assert Path(f"{tmpdir}/y_test_clean.csv").exists()
        
        # Verify data integrity
        loaded_X_train = pd.read_csv(f"{tmpdir}/X_train_clean.csv")
        loaded_X_test = pd.read_csv(f"{tmpdir}/X_test_clean.csv")
        loaded_y_train = pd.read_csv(f"{tmpdir}/y_train_clean.csv").squeeze()
        loaded_y_test = pd.read_csv(f"{tmpdir}/y_test_clean.csv").squeeze()
        
        pd.testing.assert_frame_equal(loaded_X_train, X_train)
        pd.testing.assert_frame_equal(loaded_X_test, X_test)
        pd.testing.assert_series_equal(loaded_y_train, y_train)
        pd.testing.assert_series_equal(loaded_y_test, y_test)

def test_save_processed_data_invalid_paths():
    # Test with non-existent directory
    with pytest.raises(FileNotFoundError):
        save_processed_data(
            pd.DataFrame({'a': [1]}),
            pd.DataFrame({'a': [2]}),
            pd.Series([0]),
            pd.Series([1]),
            output_dir="/nonexistent/path"
        )

def test_save_processed_data_permission_error(monkeypatch):
    def mock_to_csv(*args, **kwargs):
        raise PermissionError("Mock permission denied")
    
    # Patch pandas to_csv to simulate permission error
    monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(PermissionError):
            save_processed_data(
                pd.DataFrame({'a': [1]}),
                pd.DataFrame({'a': [2]}),
                pd.Series([0]),
                pd.Series([1]),
                output_dir=tmpdir
            )

def test_save_processed_data_edge_cases():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test data with edge cases
        X_train = pd.DataFrame({
            'nulls': [None, np.nan],
            'special_chars': ['$100', '€200'],
            'dates': [pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')],
            'mixed_types': [1, 'two']
        })
        X_test = pd.DataFrame({
            'nulls': [np.nan, None],
            'special_chars': ['¥300', '£400'],
            'dates': [pd.Timestamp('2022-01-01'), pd.Timestamp('2023-01-01')],
            'mixed_types': ['three', 4]
        })
        y_train = pd.Series([True, False])
        y_test = pd.Series([False, True])
        
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=tmpdir)
        
        # Verify data integrity with special handling for dates and NaNs
        loaded_X_train = pd.read_csv(f"{tmpdir}/X_train_clean.csv", parse_dates=['dates'])
        loaded_X_test = pd.read_csv(f"{tmpdir}/X_test_clean.csv", parse_dates=['dates'])
        loaded_y_train = pd.read_csv(f"{tmpdir}/y_train_clean.csv").squeeze()
        loaded_y_test = pd.read_csv(f"{tmpdir}/y_test_clean.csv").squeeze()
        
        # Compare with NaN equality
        pd.testing.assert_frame_equal(loaded_X_train, X_train, check_dtype=False)
        pd.testing.assert_frame_equal(loaded_X_test, X_test, check_dtype=False)
        assert loaded_y_train.equals(y_train)
        assert loaded_y_test.equals(y_test)
