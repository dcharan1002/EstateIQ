import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

def select_features(X_train, X_test, y_train, threshold='median'):
    """
    Select important features using RandomForest feature importance.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training target
    threshold : str or float
        Threshold for feature selection ('mean', 'median', or float value)
    
    Returns:
    --------
    tuple: (X_train_selected, X_test_selected, selected_features)
    """
    # Initialize base model for feature selection
    base_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit selector
    selector = SelectFromModel(
        estimator=base_rf,
        threshold=threshold
    )
    selector.fit(X_train, y_train)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"\nSelected {len(selected_features)} features out of {X_train.shape[1]}")
    print("Top 10 important features:", selected_features[:10])
    
    # Transform datasets
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Convert to DataFrame with feature names
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
    
    return X_train_selected, X_test_selected, selected_features

def load_data():
    """Load and prepare the final processed data."""
    # Define data directory
    FINAL_DATA_DIR = Path("data/final")

    # Load data sets
    X_train = pd.read_csv(FINAL_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(FINAL_DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(FINAL_DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(FINAL_DATA_DIR / "y_test.csv").squeeze()

    # Identify and remove non-numeric columns
    mixed_type_columns = set(X_train.select_dtypes(include=['object']).columns) | \
                        set(X_test.select_dtypes(include=['object']).columns)

    # Drop these columns from both train and test
    X_train = X_train.drop(columns=mixed_type_columns)
    X_test = X_test.drop(columns=mixed_type_columns)

    print(f"Dropped columns: {mixed_type_columns}")
    
    # Select important features
    print("\nPerforming feature selection...")
    X_train, X_test, selected_features = select_features(X_train, X_test, y_train)
    
    return X_train, X_test, y_train, y_test

def split_validation_data(X_train, y_train, val_size=0.2, random_state=42):
    """Split training data into training and validation sets."""
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, 
        y_train, 
        test_size=val_size, 
        random_state=random_state
    )
    return X_train_final, X_val, y_train_final, y_val
