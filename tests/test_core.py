import pytest
import pandas as pd
import numpy as np
from src.preprocessing.core import (
    identify_column_types, split_features_target
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
