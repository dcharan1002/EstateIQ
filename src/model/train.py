import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging
import pandas as pd
import joblib
from datetime import datetime
from .models import get_model
from .utils.shap_analysis import generate_shap_analysis
from .utils.metrics import calculate_metrics
from .utils.visualization import (
    plot_residuals,
    plot_predictions,
    plot_model_comparison,
    plot_hyperparameter_sensitivity
)
from .utils.notifications import (
    notify_training_completion,
    notify_error
)
from .validate import validate_model, check_bias, VALIDATION_THRESHOLDS, BIAS_THRESHOLDS
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Configure MLflow with local tracking"""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("estate_price_prediction")

def cleaning(X_train,X_test):
    """Clean and encode categorical features."""
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col], X_test[col]], axis=0).astype(str))
        X_train_encoded[col] = le.transform(X_train[col].astype(str))
        X_test_encoded[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    
    return X_train_encoded, X_test_encoded, encoders

def load_data():
    """Load preprocessed data from Data directory"""
    data_dir = Path("Data/final")
    logger.info(f"Looking for data in: {data_dir}")
    
    try:
        X_train = pd.read_csv(data_dir / "X_train.csv").head(100)
        X_test = pd.read_csv(data_dir / "X_test.csv").head(100)
        y_train = pd.read_csv(data_dir / "y_train.csv").squeeze().iloc[:100]
        y_test = pd.read_csv(data_dir / "y_test.csv").squeeze().iloc[:100]
        X_train['BLDG_TYPE'] = X_train['BLDG_TYPE'].astype('category')
        X_test['BLDG_TYPE'] = X_test['BLDG_TYPE'].astype('category')
        logger.info(f"Loaded data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
        
        # Store original categorical features
        for col in ['STRUCTURE_CLASS', 'OWNER_OCC', 'OVERALL_COND', 'INT_COND']:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype('category')
                X_test[col] = X_test[col].astype('category')
        
        # Get encoded versions for training
        X_train_encoded, X_test_encoded, encoders = cleaning(X_train, X_test)
        
        return X_train, X_train_encoded, X_test, X_test_encoded, y_train, y_test, encoders
    except Exception as e:
        context = {
            "data_dir": str(data_dir),
            "attempted_files": ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
        }
        notify_error(e, context)
        raise

def create_hyperparameter_results(model, metrics):
    """Create hyperparameter results dataframe for visualization"""
    return pd.DataFrame([{
        'config': 'default',
        'r2': metrics['r2'],
        'rmse': metrics['rmse'],
        'mae': metrics['mae']
    }])

def save_model(model, model_dir):
    """Save model to specified directory with proper serialization"""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model_latest.joblib"
    
    if hasattr(model, 'model'):
        joblib.dump(model.model, model_path)
    else:
        joblib.dump(model, model_path)
    
    logger.info(f"Model saved to {model_path}")
    return model_path

def train_model():
    """Train and validate model with MLflow tracking"""
    training_context = {
        "stage": "initialization",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        setup_mlflow()
        logger.info("MLflow configured successfully")
        
        # Load raw data first
        X_train_raw, X_train, X_test_raw, X_test, y_train, y_test, encoders = load_data()
        logger.info("Data loaded successfully")
        training_context["stage"] = "data_loading_complete"

        model_configs = {
            'random_forest': {'n_estimators': 100},
            'xgboost': {'n_estimators': 100, 'learning_rate': 0.1}
        }

        best_model = None
        best_score = 0
        best_metrics = None
        best_run_id = None
        model_predictions = {}
        validation_results = {}

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        training_context["timestamp"] = timestamp
        
        artifacts_dir = Path("artifacts")
        results_dir = Path("results")
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        with mlflow.start_run(run_name=f"model_training_{timestamp}") as parent_run:
            mlflow.set_tag("timestamp", timestamp)
            all_plots = {}
            training_context["stage"] = "training_started"
            training_context["mlflow_run_id"] = parent_run.info.run_id
            
            for model_name, params in model_configs.items():
                logger.info(f"Training {model_name}...")
                training_context["current_model"] = model_name
                model_results_dir = results_dir / f"{model_name}_{timestamp}"
                os.makedirs(model_results_dir, exist_ok=True)
                
                with mlflow.start_run(run_name=model_name, nested=True) as run:
                    model = get_model(model_name, **params)
                    model.train(X_train, y_train)
                    
                    # Generate predictions
                    y_pred = model.predict(X_test)
                    model_predictions[model_name] = y_pred
                    metrics = calculate_metrics(model, X_test, y_test)
                    
                    # Generate all visualizations
                    logger.info(f"Generating visualizations for {model_name}...")
                    plots = {}
                    
                    # 1. Residuals Plot
                    residuals_path = plot_residuals(y_test, y_pred, model_name)
                    plots[f"{model_name} Residuals"] = residuals_path
                    mlflow.log_artifact(residuals_path)
                    
                    # 2. Predictions vs Actual Plot
                    predictions_path = plot_predictions(y_test, y_pred, model_name)
                    plots[f"{model_name} Predictions"] = predictions_path
                    mlflow.log_artifact(predictions_path)
                    
                    # 3. Hyperparameter Sensitivity Plot
                    results_df = create_hyperparameter_results(model, metrics)
                    sensitivity_path = plot_hyperparameter_sensitivity(results_df, model_name)
                    plots[f"{model_name} Hyperparameters"] = sensitivity_path
                    mlflow.log_artifact(sensitivity_path)
                    
                    # 4. SHAP Analysis
                    logger.info(f"Generating SHAP analysis for {model_name}...")
                    shap_analysis = generate_shap_analysis(
                        model.model if hasattr(model, 'model') else model,
                        X_test,
                        results_dir=model_results_dir
                    )
                    
                    # Add SHAP plots
                    if shap_analysis and 'plots' in shap_analysis:
                        for plot_name, plot_path in shap_analysis['plots'].items():
                            plots[f"{model_name} {plot_name}"] = plot_path
                            if os.path.exists(plot_path):
                                mlflow.log_artifact(plot_path)
                    
                    # Store all plots for this model
                    all_plots.update(plots)
                    
                    mlflow.log_metrics(metrics)
                    if model_name == 'xgboost':
                        mlflow.xgboost.log_model(model.model, model_name)
                    else:
                        mlflow.sklearn.log_model(model.model, model_name)
                    
                    if metrics['r2'] > best_score:
                        best_score = metrics['r2']
                        best_model = model
                        best_metrics = metrics
                        best_run_id = run.info.run_id
            
            # Generate model comparison plot
            if len(model_predictions) == 2:
                logger.info("Generating model comparison plot...")
                comparison_path = plot_model_comparison(
                    y_test,
                    model_predictions['random_forest'],
                    model_predictions['xgboost']
                )
                mlflow.log_artifact(comparison_path)
                all_plots['Model Comparison'] = comparison_path
            
            training_context["stage"] = "training_complete"
            
            if best_model is not None:
                # Save the best model first
                model_path = save_model(best_model, artifacts_dir)
                logger.info(f"Best model saved to: {model_path}")
                
                mlflow.set_tag("best_model", "true")
                mlflow.set_tag("best_model_type", best_model.name)
                mlflow.set_tag("best_run_id", best_run_id)
                mlflow.log_metrics(best_metrics)
                mlflow.log_artifact(str(model_path))
                
                # Check bias using raw data for interpretable results
                logger.info("Checking for bias in the best model...")
                bias_report = check_bias(
                    best_model.model if hasattr(best_model, 'model') else best_model,
                    X_test,  # Use raw data for bias checking
                    y_test
                )
                
                # Run model validation
                logger.info("Running model validation...")
                training_context["stage"] = "validation"
                validation_passed, validation_metrics, _ = validate_model(
                    model=best_model.model if hasattr(best_model, 'model') else best_model,
                    run_id=parent_run.info.run_id,
                    X_val=X_test,  # Use encoded data for validation
                    y_val=y_test
                )
                
                # Send training report with bias analysis
                notify_training_completion(
                    model_name=best_model.name,
                    metrics=best_metrics,
                    plots=all_plots,
                    run_id=parent_run.info.run_id,
                    validation_thresholds=VALIDATION_THRESHOLDS,
                    bias_report=bias_report,
                    success=validation_passed
                )
                
                logger.info(f"Best model: {best_model.name}")
                logger.info(f"Best model metrics: {best_metrics}")
                return validation_passed
            
            else:
                raise Exception("No models were successfully trained")
                
    except Exception as e:
        notify_error(e, training_context)
        raise

if __name__ == "__main__":
    success = train_model()
    if not success:
        exit(1)
