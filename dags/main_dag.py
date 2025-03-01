from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import os

from src.data.download import download_boston_housing_data
from src.preprocessing.reporting import FeatureReporter
from src.preprocessing.cleaning import (
    handle_missing_values, handle_outliers, remove_duplicate_rows,
    clean_numeric_columns, clean_address_data, standardize_categorical_values
)
from src.preprocessing.core import (
    split_features_target, save_processed_data, identify_column_types
)
from src.preprocessing.features import (
    encode_categorical_features, create_ratio_features, 
    create_property_features
)

import pathlib

# Define data directories
DATA_DIR = pathlib.Path('/opt/airflow/data')
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FEATURES_DIR = DATA_DIR / 'features'

# Directory for syncing with source repo
SOURCES_DATA_DIR = pathlib.Path('/sources/data')

# Define numeric columns for consistent handling across tasks
NUMERIC_COLS = ['TOTAL_VALUE', 'GROSS_AREA', 'LAND_VALUE', 'BLDG_VALUE']

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def clean_data(**context):
    # Read data with numeric converters
    converters = {col: parse_numeric for col in NUMERIC_COLS}
    
    df = pd.read_csv(RAW_DATA_DIR / 'boston_2025.csv', 
                    low_memory=False,
                    converters=converters)
    
    # Get column types
    column_types = identify_column_types(df)
    
    # Clean in proper order
    df = remove_duplicate_rows(df)
    df = handle_missing_values(df)  # Handle missing values first
    df = clean_numeric_columns(df)  # Then clean numeric columns
    df = clean_address_data(df, column_types['address'])
    df = standardize_categorical_values(df, column_types['categorical'])
    df = handle_outliers(df)
    
    # Split data early
    X, y = split_features_target(df, target_column='TOTAL_VALUE')
    
    # Split into train/test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create clean data directory and save
    clean_dir = PROCESSED_DATA_DIR / 'clean'
    clean_dir.mkdir(exist_ok=True)
    
    # Save cleaned data with descriptive names
    X_train.to_csv(clean_dir / 'X_train_clean.csv', index=False)
    X_test.to_csv(clean_dir / 'X_test_clean.csv', index=False)
    pd.Series(y_train).to_csv(clean_dir / 'y_train_clean.csv', index=False)
    pd.Series(y_test).to_csv(clean_dir / 'y_test_clean.csv', index=False)

def engineer_features(**context):
    # Read cleaned train data with numeric converters
    converters = {col: parse_numeric for col in NUMERIC_COLS}
    
    clean_dir = PROCESSED_DATA_DIR / 'clean'
    X_train = pd.read_csv(clean_dir / 'X_train_clean.csv', 
                         low_memory=False,
                         converters=converters)
    X_test = pd.read_csv(clean_dir / 'X_test_clean.csv', 
                        low_memory=False,
                        converters=converters)
    
    # Process train data
    X_train, encoders = encode_categorical_features(X_train)
    
    # Process test data using same encoders
    X_test = encode_categorical_features(X_test, existing_encoders=encoders)[0]
    
    # Create additional features for both sets
    X_train = create_ratio_features(X_train)
    X_train = create_property_features(X_train)
    
    X_test = create_ratio_features(X_test)
    X_test = create_property_features(X_test)
    
    # Create engineered data directory and save
    engineered_dir = PROCESSED_DATA_DIR / 'engineered'
    engineered_dir.mkdir(exist_ok=True)
    
    # Save engineered features
    X_train.to_csv(engineered_dir / 'X_train_engineered.csv', index=False)
    X_test.to_csv(engineered_dir / 'X_test_engineered.csv', index=False)

def generate_feature_reports(**context):
    # Read data with numeric converters
    converters = {col: parse_numeric for col in NUMERIC_COLS}
    
    engineered_dir = PROCESSED_DATA_DIR / 'engineered'
    clean_dir = PROCESSED_DATA_DIR / 'clean'
    
    X_train = pd.read_csv(engineered_dir / 'X_train_engineered.csv', 
                         low_memory=False,
                         converters=converters)
    y_train = pd.read_csv(clean_dir / 'y_train_clean.csv',
                         converters={'0': parse_numeric}).squeeze()
    
    # Combine features and target for reporting
    df = X_train.copy()
    df['TOTAL_VALUE'] = pd.to_numeric(y_train, errors='coerce')  # Ensure numeric

    reporter = FeatureReporter(FEATURES_DIR)
    reporter.generate_all_reports(df)

def save_data(**context):
    """Save final processed data to the repository's data directory"""
    # Read engineered features with numeric converters
    converters = {col: parse_numeric for col in NUMERIC_COLS}
    
    engineered_dir = PROCESSED_DATA_DIR / 'engineered'
    clean_dir = PROCESSED_DATA_DIR / 'clean'
    
    X_train = pd.read_csv(engineered_dir / 'X_train_engineered.csv', 
                         low_memory=False,
                         converters=converters)
    X_test = pd.read_csv(engineered_dir / 'X_test_engineered.csv', 
                        low_memory=False,
                        converters=converters)
    y_train = pd.read_csv(clean_dir / 'y_train_clean.csv',
                         converters={'0': parse_numeric}).squeeze()
    y_test = pd.read_csv(clean_dir / 'y_test_clean.csv',
                        converters={'0': parse_numeric}).squeeze()
    
    # Save final data to repo
    final_data_dir = DATA_DIR / 'final'
    final_data_dir.mkdir(exist_ok=True)
    
    X_train.to_csv(final_data_dir / 'X_train.csv', index=False)
    X_test.to_csv(final_data_dir / 'X_test.csv', index=False)
    pd.Series(y_train).to_csv(final_data_dir / 'y_train.csv', index=False)
    pd.Series(y_test).to_csv(final_data_dir / 'y_test.csv', index=False)

def parse_numeric(x):
    """Convert string numbers with commas to float"""
    try:
        if isinstance(x, str):
            return float(x.replace(',', ''))
        return float(x)
    except (ValueError, TypeError):
        return None

with DAG(
    'boston_housing_pipeline',
    default_args=default_args,
    description='End-to-end Boston Housing data pipeline with feature analysis',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    download_task = PythonOperator(
        task_id='download_data',
        python_callable=download_boston_housing_data,
        dag=dag,
    )


    clean_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        dag=dag,
    )

    feature_engineering_task = PythonOperator(
        task_id='engineer_features',
        python_callable=engineer_features,
        dag=dag,
    )

    generate_reports_task = PythonOperator(
        task_id='generate_feature_reports',
        python_callable=generate_feature_reports,
        dag=dag,
    )

    save_task = PythonOperator(
        task_id='save_data',
        python_callable=save_data,
        dag=dag,
    )

    # Set task dependencies
    download_task >> clean_task >> \
    feature_engineering_task >> [generate_reports_task, save_task]
