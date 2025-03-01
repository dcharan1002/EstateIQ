import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import tempfile
from src.preprocessing.reporting import FeatureReporter

@pytest.fixture
def sample_property_df():
    return pd.DataFrame({
        'BLDG_VALUE': [500000, 600000, 700000, 800000],
        'LAND_VALUE': [200000, 250000, 300000, 350000],
        'LIVING_AREA': [2000, 2400, 2800, 3200],
        'GROSS_AREA': [2500, 2900, 3300, 3700],
        'TT_RMS': [8, 9, 10, 11],
        'BED_RMS': [3, 4, 4, 5],
        'FULL_BTH': [2, 2, 3, 3],
        'HLF_BTH': [1, 1, 1, 2],
        'LAND_SF': [5000, 6000, 7000, 8000],
        'property_age': [20, 15, 25, 10],
        'age_category': ['20-30 years', '10-20 years', '20-30 years', '10-20 years'],
        'total_bathrooms': [2.5, 2.5, 3.5, 4.0],
        'living_area_ratio': [0.8, 0.83, 0.85, 0.86],
        'non_living_area': [500, 500, 500, 500],
        'has_renovation': [True, False, True, False],
        'years_since_renovation': [5, None, 10, None],
        'price_per_sqft': [250, 250, 250, 250],
        'land_value_ratio': [0.4, 0.42, 0.43, 0.44],
        'building_value_ratio': [0.6, 0.58, 0.57, 0.56]
    })

def test_feature_reporter_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = FeatureReporter(tmpdir)
        assert os.path.exists(tmpdir)

def test_generate_combined_plots(sample_property_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = FeatureReporter(tmpdir)
        reporter.generate_combined_plots(sample_property_df)
        
        # Check if plot file was created
        assert Path(f"{tmpdir}/feature_analysis.png").exists()
        
        # Verify file size is reasonable (not empty)
        assert os.path.getsize(f"{tmpdir}/feature_analysis.png") > 1000

def test_generate_all_reports(sample_property_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = FeatureReporter(tmpdir)
        reporter.generate_all_reports(sample_property_df)
        
        # Check all expected files were created
        expected_files = [
            'feature_analysis.png',
            'feature_summary.csv',
            'feature_report.md'
        ]
        for file in expected_files:
            assert Path(f"{tmpdir}/{file}").exists()

def test_feature_reporter_with_missing_data():
    df = pd.DataFrame({
        'BLDG_VALUE': [500000, None, 700000],
        'LAND_VALUE': [None, 250000, 300000],
        'LIVING_AREA': [2000, 2400, None],
        'BED_RMS': ['3', '4', 'invalid'],
        'property_age': [20, None, 25]
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = FeatureReporter(tmpdir)
        reporter.generate_all_reports(df)
        
        # Verify files were created despite missing data
        assert Path(f"{tmpdir}/feature_analysis.png").exists()
        assert Path(f"{tmpdir}/feature_summary.csv").exists()
        assert Path(f"{tmpdir}/feature_report.md").exists()
        
        # Check if missing values are properly reported
        summary_df = pd.read_csv(f"{tmpdir}/feature_summary.csv")
        assert (summary_df['Missing_Values'] > 0).any()

def test_feature_reporter_with_invalid_directory():
    with pytest.raises(Exception):
        FeatureReporter('/nonexistent/directory/path').generate_all_reports(pd.DataFrame())

def test_feature_reporter_with_empty_dataframe():
    df = pd.DataFrame()
    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = FeatureReporter(tmpdir)
        reporter.generate_all_reports(df)
        
        # Check if files were created with empty data
        assert Path(f"{tmpdir}/feature_summary.csv").exists()
        assert Path(f"{tmpdir}/feature_report.md").exists()
        
        # Verify empty dataframe is handled properly
        summary_df = pd.read_csv(f"{tmpdir}/feature_summary.csv")
        assert len(summary_df) == 0

def test_feature_reporter_with_single_column():
    df = pd.DataFrame({'single_col': [1, 2, 3]})
    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = FeatureReporter(tmpdir)
        reporter.generate_all_reports(df)
        
        # Check if files were created with minimal data
        assert Path(f"{tmpdir}/feature_summary.csv").exists()
        assert Path(f"{tmpdir}/feature_report.md").exists()
        
        # Verify single column is properly analyzed
        summary_df = pd.read_csv(f"{tmpdir}/feature_summary.csv")
        assert len(summary_df) == 1
        assert summary_df['Feature'].iloc[0] == 'single_col'
