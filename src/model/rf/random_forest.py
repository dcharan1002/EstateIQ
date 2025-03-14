from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_rf_pipeline(random_state=42):
    """Create Random Forest pipeline with preprocessing."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=random_state))
    ])

def get_rf_param_grid():
    """Get hyperparameter grid for Random Forest tuning."""
    return {
        'model__n_estimators': [50, 100, 200, 300],
        'model__max_depth': [None, 10, 20, 30, 40],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__bootstrap': [True, False],
        'model__max_features': ['sqrt', 'log2', None],
        'model__max_samples': [0.7, 0.8, 0.9, None]
    }

def train_and_tune_rf(X_train, y_train, X_val, y_val):
    """
    Train and tune Random Forest model.
    
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
    from src.model.utils.evaluation import perform_randomized_search, evaluate_model
    
    print(f"\nStarting Random Forest training with {X_train.shape[0]} samples and {X_train.shape[1]} features...")
    
    # Create and evaluate base model
    print("\nTraining base Random Forest model...")
    base_pipeline = create_rf_pipeline()
    base_pipeline.fit(X_train, y_train)
    print("\nBase Random Forest Model Performance:")
    evaluate_model(base_pipeline, X_val, y_val, "Validation")
    
    # Perform hyperparameter tuning
    print("\nStarting Random Forest hyperparameter tuning...")
    param_grid = get_rf_param_grid()
    print("\nHyperparameter search space:")
    for param, values in param_grid.items():
        print(f"- {param}: {values}")
    
    best_model = perform_randomized_search(
        create_rf_pipeline(),
        param_grid,
        X_train,
        y_train,
    )
    
    # Evaluate tuned model
    print("\nTuned Random Forest Model Performance:")
    evaluate_model(best_model, X_val, y_val, "Validation")
    
    return best_model
