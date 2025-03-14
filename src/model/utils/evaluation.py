import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

def evaluate_model(model, X_data, y_true, dataset_name=""):
    """
    Evaluate model performance using MSE, RMSE, and R2 score.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to evaluate
    X_data : pd.DataFrame
        Feature data
    y_true : pd.Series
        True target values
    dataset_name : str, optional
        Name of the dataset being evaluated (e.g., "Validation" or "Test")
    
    Returns:
    --------
    tuple: (mse, rmse, r2)
    """
    y_pred = model.predict(X_data)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    if dataset_name:
        print(f"\nModel Performance ({dataset_name} Set):")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
    
    return mse, rmse, r2

def perform_randomized_search(
    pipeline,
    param_dist,
    X_train,
    y_train,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=-1
):
    """
    Perform RandomizedSearchCV for hyperparameter tuning.
    
    Parameters:
    -----------
    pipeline : sklearn Pipeline
        Model pipeline to tune
    param_dist : dict
        Hyperparameter distribution for random search
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_iter : int, optional
        Number of parameter settings sampled
    cv : int, optional
        Number of cross-validation folds
    random_state : int, optional
        Random state for reproducibility
    n_jobs : int, optional
        Number of parallel jobs
        
    Returns:
    --------
    sklearn estimator: Best model from random search
    """
    print(f"\nStarting RandomizedSearchCV with {n_iter} iterations and {cv}-fold CV")
    print(f"Training data shape: {X_train.shape}")
    
    # Create RandomizedSearchCV with progress callback
    random_search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        verbose=2,  # Increased verbosity
        random_state=random_state
    )
    
    # Time the training
    start_time = time.time()
    print("\nFitting RandomizedSearchCV...")
    random_search.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Print detailed results
    print(f"\nRandomizedSearchCV completed in {train_time:.2f} seconds")
    print("\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"- {param}: {value}")
    print(f"\nBest cross-validation MSE: {-random_search.best_score_:.4f}")
    
    # Print top 3 models
    cv_results = random_search.cv_results_
    top_idx = np.argsort(cv_results['mean_test_score'])[-3:][::-1]
    print("\nTop 3 models:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"\n{rank}. Mean CV MSE: {-cv_results['mean_test_score'][idx]:.4f}")
        print(f"   Std CV MSE: {cv_results['std_test_score'][idx]:.4f}")
        print("   Parameters:")
        for param, value in cv_results['params'][idx].items():
            print(f"   - {param}: {value}")
    
    return random_search.best_estimator_
