import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split,  RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Step 1: Load the final processed data
def load_data():
    # Define data directory
    FINAL_DATA_DIR = Path("Data/final")

    # Load data sets
    X_train = pd.read_csv(FINAL_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(FINAL_DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(FINAL_DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(FINAL_DATA_DIR / "y_test.csv").squeeze()

    # Identify columns with object (non-numeric) data types in both train and test
    mixed_type_columns = set(X_train.select_dtypes(include=['object']).columns) | set(X_test.select_dtypes(include=['object']).columns)

    # Drop these columns from both train and test
    X_train = X_train.drop(columns=mixed_type_columns)
    X_test = X_test.drop(columns=mixed_type_columns)

    print(f"Dropped columns: {mixed_type_columns}")
    
    return X_train, X_test, y_train, y_test

# Step 2: Split data further into training and validation sets
def split_data(X_train, y_train):
    # Split the training data into training and validation sets
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train_final, X_val, y_train_final, y_val

# Step 3: Define Models
def define_models():
    # Define pipeline for XGBoost model
    xgb_model = Pipeline([
        ('scaler', StandardScaler()),  # Feature Scaling
        ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
    ])
    
    # Define pipeline for Random Forest model
    rf_model = Pipeline([
        ('scaler', StandardScaler()),  # Feature Scaling
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    return xgb_model, rf_model

# Step 4: Train and Evaluate Model using Hold-Out Validation Set
def evaluate_model_with_holdout(pipeline, X_train, y_train, X_val, y_val):
    """Trains and evaluates the model using hold-out validation set."""
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_val_pred)
    
    return mse, rmse, r2

# Step 5: Perform Hyperparameter Tuning using RandomizedSearchCV
def hyperparameter_tuning(rf_pipeline, xgb_pipeline, X_train_final, y_train_final):
    # Hyperparameter grid for Random Forest Regressor
    rf_param_dist = {
        'model__n_estimators': [50, 100, 200, 300],
        'model__max_depth': [None, 10, 20, 30, 40],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__bootstrap': [True, False]
    }

    # Hyperparameter grid for XGBoost Regressor
    xgb_param_dist = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7, 10],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0]
    }

    # RandomizedSearch for Random Forest
    rf_random_search = RandomizedSearchCV(rf_pipeline, rf_param_dist, n_iter=10, cv=5, 
                                          scoring='neg_mean_squared_error', n_jobs=-1, verbose=1, random_state=42)
    rf_random_search.fit(X_train_final, y_train_final)
    print(f"Best Hyperparameters for Random Forest: {rf_random_search.best_params_}")
    best_rf_model = rf_random_search.best_estimator_

    # RandomizedSearch for XGBoost
    xgb_random_search = RandomizedSearchCV(xgb_pipeline, xgb_param_dist, n_iter=10, cv=5, 
                                           scoring='neg_mean_squared_error', n_jobs=-1, verbose=1, random_state=42)
    xgb_random_search.fit(X_train_final, y_train_final)
    print(f"Best Hyperparameters for XGBoost: {xgb_random_search.best_params_}")
    best_xgb_model = xgb_random_search.best_estimator_

    return best_rf_model, best_xgb_model


# Step 6: Final Model Evaluation using Test Set
def evaluate_final_model(best_model, X_test, y_test):
    y_test_pred = best_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"\nFinal Model Performance (Test Set):")
    print(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")
    return mse_test, rmse_test, r2_test

# Main execution flow
def main():
    # Load Data
    X_train, X_test, y_train, y_test = load_data()

    # Split data into training and validation sets
    X_train_final, X_val, y_train_final, y_val = split_data(X_train, y_train)

    # Define models
    xgb_pipeline, rf_pipeline = define_models()

    # Evaluate models using the hold-out validation set
    print("\nEvaluating XGBoost Model on Hold-Out Set:")
    xgb_metrics = evaluate_model_with_holdout(xgb_pipeline, X_train_final, y_train_final, X_val, y_val)
    print(f"MSE: {xgb_metrics[0]:.4f}, RMSE: {xgb_metrics[1]:.4f}, R2: {xgb_metrics[2]:.4f}")

    print("\nEvaluating Random Forest Model on Hold-Out Set:")
    rf_metrics = evaluate_model_with_holdout(rf_pipeline, X_train_final, y_train_final, X_val, y_val)
    print(f"MSE: {rf_metrics[0]:.4f}, RMSE: {rf_metrics[1]:.4f},  R2: {rf_metrics[2]:.4f}")

    # Perform Hyperparameter Tuning and Get Best Models
    best_rf_model, best_xgb_model = hyperparameter_tuning(rf_pipeline, xgb_pipeline, X_train_final, y_train_final)

    # Evaluate the best model on the test set
    print("\nEvaluating Best Random Forest Model on Test Set:")
    evaluate_final_model(best_rf_model, X_test, y_test)

    print("\nEvaluating Best XGBoost Model on Test Set:")
    evaluate_final_model(best_xgb_model, X_test, y_test)

if __name__ == "__main__":
    main()
