import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb


def load_data():
    # Load data sets
    # Choose the fie location
    X_train = pd.read_csv(r"C:\Users\dchar\OneDrive\Desktop\Project\EstateIQ\data\final\X_train.csv", low_memory=False)
    X_test = pd.read_csv(r"C:\Users\dchar\OneDrive\Desktop\Project\EstateIQ\data\final\X_test.csv", low_memory=False)
    y_train = pd.read_csv(r"C:\Users\dchar\OneDrive\Desktop\Project\EstateIQ\data\final\y_train.csv").squeeze()
    y_test = pd.read_csv(r"C:\Users\dchar\OneDrive\Desktop\Project\EstateIQ\data\final\y_test.csv").squeeze()  
    return X_train, X_test, y_train, y_test

def cleaning(X_train,X_test):
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    cat_cols = X_train.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col], X_test[col]], axis=0).astype(str))
        X_train_encoded[col] = le.transform(X_train[col].astype(str))
        X_test_encoded[col] = le.transform(X_test[col].astype(str))
    return X_train_encoded,X_test_encoded


def define_models(X_train_encoded,X_test_encoded,y_train,y_test):
# XGBoost Model
    xgb_model = XGBRegressor(
    n_estimators=100,
    random_state=42,
    verbosity=0,
    enable_categorical=True
    )
# Ensure categorical columns are properly marked
    X_train_encoded['BLDG_TYPE'] = X_train_encoded['BLDG_TYPE'].astype('category')
    X_test_encoded['BLDG_TYPE'] = X_test_encoded['BLDG_TYPE'].astype('category')
    xgb_model.fit(X_train_encoded, y_train)
# Extract feature importance
    xgb_importances = pd.Series(xgb_model.feature_importances_, index=X_train_encoded.columns)
    top_15_xgb_features = xgb_importances.sort_values(ascending=False).head(15)
    print("Top 15 Features from XGBoost:\n", top_15_xgb_features)
# Random forest model    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_encoded, y_train)
# Extract feature importance
    feature_importances = pd.Series(rf_model.feature_importances_, index=X_train_encoded.columns)
    top_15_rf_features = feature_importances.sort_values(ascending=False).head(15)
    print("Top 15 Features from Random Forest:\n", top_15_rf_features)
    X_train_selected = X_train_encoded[top_15_rf_features.index]
    X_test_selected = X_test_encoded[top_15_rf_features.index]
# Model Evaluation
    X_train_selected = X_train_encoded[top_15_rf_features.index]
    X_test_selected = X_test_encoded[top_15_rf_features.index]

    rf_model.fit(X_train_selected, y_train)
    rf_preds = rf_model.predict(X_test_selected)

    xgb_model.fit(X_train_selected, y_train)
    xgb_preds = xgb_model.predict(X_test_selected)
    # Evaluate Both Models
    evaluate_model("Random Forest", y_test, rf_preds)
    evaluate_model("XGBoost", y_test, xgb_preds)
    return X_train_selected,X_test_selected


def evaluate_model(Model_name, y_true, y_pred):
    print(f"\n {Model_name} Evaluation:")
    print("R² Score       :", r2_score(y_true, y_pred))
    print("MAE            :", mean_absolute_error(y_true, y_pred))
    print("MSE            :", mean_squared_error(y_true, y_pred))
    print("RMSE           :", np.sqrt(mean_squared_error(y_true, y_pred)))


def hyperparameter_tuning(X_train_selected,X_test_selected,y_train,y_test):
    # Define parameter grids
    rf_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
    }

    xgb_param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1]
    }
# XGBoost tuning
    xgb_random = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(),
    param_distributions=xgb_param_grid,
    n_iter=5,
    cv=2,
    verbose=2,
    random_state=42
    )

    xgb_random.fit(X_train_selected, y_train)
    best_xgb = xgb_random.best_estimator_

# Random Forest tuning
    rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(),
    param_distributions=rf_param_grid,
    n_iter=5,
    cv=2,
    verbose=2,
    random_state=42
    )

    rf_random.fit(X_train_selected, y_train)
    best_rf = rf_random.best_estimator_

# Re-evaluate with tuned models
    rf_preds_tuned = best_rf.predict(X_test_selected)
    xgb_preds_tuned = best_xgb.predict(X_test_selected)
    print("\nEvaluating Best Random Forest Model on Test Set:")
    evaluate_model("Tuned Random Forest", y_test, rf_preds_tuned)
    print("\nEvaluating Best XGBoost Model on Test Set:")
    evaluate_model("Tuned XGBoost", y_test, xgb_preds_tuned)

def main():
    # Load Data
    X_train, X_test, y_train, y_test = load_data()
    X_train_encoded, X_test_encoded = cleaning(X_train,X_test)

    # Define models and feature selection
    X_train_selected,X_test_selected = define_models(X_train_encoded,X_test_encoded,y_train,y_test)

    #Perform Hyperparameter Tuning and Get Best Models
    hyperparameter_tuning(X_train_selected,X_test_selected, y_train, y_test)


if __name__ == "__main__":
    main()
