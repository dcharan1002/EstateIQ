from utils.data import load_data, split_validation_data
from utils.evaluation import evaluate_model
from rf.random_forest import train_and_tune_rf
from xgb.xgboost import train_and_tune_xgb

def train_models():
    """Train and evaluate both Random Forest and XGBoost models."""
    # Load data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Split into training and validation sets
    print("\nSplitting data into training and validation sets...")
    X_train_final, X_val, y_train_final, y_val = split_validation_data(X_train, y_train)
    
    # Train and tune Random Forest
    print("\n" + "="*50)
    print("Training and tuning Random Forest model...")
    print("="*50)
    best_rf = train_and_tune_rf(X_train_final, y_train_final, X_val, y_val)
    
    # Train and tune XGBoost
    print("\n" + "="*50)
    print("Training and tuning XGBoost model...")
    print("="*50)
    best_xgb = train_and_tune_xgb(X_train_final, y_train_final, X_val, y_val)
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Model Evaluation on Test Set")
    print("="*50)
    
    print("\nRandom Forest Final Performance:")
    rf_test_metrics = evaluate_model(best_rf, X_test, y_test, "Test")
    
    print("\nXGBoost Final Performance:")
    xgb_test_metrics = evaluate_model(best_xgb, X_test, y_test, "Test")
    
    # Compare final models
    print("\n" + "="*50)
    print("Model Comparison Summary")
    print("="*50)
    models = {
        "Random Forest": rf_test_metrics,
        "XGBoost": xgb_test_metrics
    }
    
    best_model = min(models.items(), key=lambda x: x[1][0])  # Compare based on MSE
    print(f"\nBest performing model: {best_model[0]}")
    print(f"MSE: {best_model[1][0]:.4f}")
    print(f"RMSE: {best_model[1][1]:.4f}")
    print(f"R2: {best_model[1][2]:.4f}")
    
    return best_rf, best_xgb

if __name__ == "__main__":
    # This ensures the script runs from the project root directory
    import sys
    from pathlib import Path
    
    # Add the project root to Python path if not already there
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    train_models()
