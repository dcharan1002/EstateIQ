from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_xgb_pipeline(random_state=42):
    """Create XGBoost pipeline with preprocessing."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(
            objective='reg:squarederror',
            random_state=random_state,
            tree_method='hist'  # More efficient tree method
        ))
    ])

def get_xgb_param_grid():
    """Get hyperparameter grid for XGBoost tuning."""
    return {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7, 10],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
        'model__min_child_weight': [1, 3, 5],  # Added min_child_weight parameter
        'model__gamma': [0, 0.1, 0.2],  # Added gamma parameter for pruning
        'model__reg_alpha': [0, 0.1, 1.0],  # L1 regularization
        'model__reg_lambda': [0, 0.1, 1.0]   # L2 regularization
    }

def train_and_tune_xgb(X_train, y_train, X_val, y_val):
    """
    Train and tune XGBoost model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
        
    Returns:
    --------
    sklearn Pipeline: Best tuned model
    """
    from ..utils.evaluation import perform_randomized_search, evaluate_model
    
    # Create and evaluate base model
    base_pipeline = create_xgb_pipeline()
    base_pipeline.fit(X_train, y_train)
    print("\nBase XGBoost Model Performance:")
    evaluate_model(base_pipeline, X_val, y_val, "Validation")
    
    # Perform hyperparameter tuning
    print("\nTuning XGBoost Hyperparameters...")
    param_grid = get_xgb_param_grid()
    best_model = perform_randomized_search(
        create_xgb_pipeline(),
        param_grid,
        X_train,
        y_train,
        n_iter=15  # Increased iterations due to larger parameter space
    )
    
    # Evaluate tuned model
    print("\nTuned XGBoost Model Performance:")
    evaluate_model(best_model, X_val, y_val, "Validation")
    
    return best_model
