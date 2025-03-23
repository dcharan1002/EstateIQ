import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging
import pandas as pd
import joblib
from datetime import datetime
from .models import get_model
from .utils.bias_analysis import BiasAnalyzer
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
    data_dir = Path("Data/final")
    logger.info(f"Looking for data in: {data_dir}")
    
    try:
        X_train = pd.read_csv(data_dir / "X_train.csv").head(1000)
        X_test = pd.read_csv(data_dir / "X_test.csv").head(1000)
        y_train = pd.read_csv(data_dir / "y_train.csv").squeeze().iloc[:1000]
        y_test = pd.read_csv(data_dir / "y_test.csv").squeeze().iloc[:1000]
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
        
        # Create validation split (20% of training data)
        validation_size = int(len(X_train_encoded) * 0.2)
        X_val_encoded = X_train_encoded.tail(validation_size).reset_index(drop=True)
        X_val = X_train.tail(validation_size).reset_index(drop=True)
        y_val = y_train.tail(validation_size).reset_index(drop=True)
        
        # Update training data to exclude validation set
        X_train_encoded = X_train_encoded.head(-validation_size).reset_index(drop=True)
        X_train = X_train.head(-validation_size).reset_index(drop=True)
        y_train = y_train.head(-validation_size).reset_index(drop=True)
        
        logger.info(f"Created validation split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} samples")
        
        return X_train, X_train_encoded, X_test, X_test_encoded, X_val, X_val_encoded, y_train, y_val, y_test, encoders
    except Exception as e:
        context = {
            "data_dir": str(data_dir),
            "attempted_files": ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
        }
        notify_error(e, context)
        raise

def create_hyperparameter_results(model, metrics):
    return pd.DataFrame([{
        'config': 'default',
        'r2': metrics['r2'],
        'rmse': metrics['rmse'],
        'mae': metrics['mae']
    }])

def save_model(model, model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model_latest.joblib"
    
    if hasattr(model, 'model'):
        joblib.dump(model.model, model_path)
    else:
        joblib.dump(model, model_path)
    
    logger.info(f"Model saved to {model_path}")
    return model_path

def evaluate_and_mitigate_bias(model, X_val, y_val):
    """Analyze and mitigate bias with enhanced reporting"""
    bias_analyzer = BiasAnalyzer(model)
    
    # Initial bias analysis
    logger.info("Performing initial bias analysis...")
    bias_results = bias_analyzer.analyze_bias(X_val, y_val)
    bias_report = bias_analyzer.generate_bias_report(bias_results)
    
    if bias_report['bias_detected']:
        significant_disparities = bias_report.get('significant_disparities', [])
        logger.warning(f"Bias detected in {len(significant_disparities)} features")
        
        # Log initial disparities
        for disparity in significant_disparities:
            if 'feature' in disparity:
                feature = disparity['feature']
                logger.info(f"\nBias details for {feature}:")
                for metric, value in disparity['disparities'].items():
                    logger.info(f"- {metric}: {value:.3f}")
                if 'interpretation' in disparity:
                    for interpretation in disparity['interpretation']:
                        logger.info(f"- {interpretation}")
        
        # Attempt mitigation for each feature
        mitigated_model = model
        for disparity in significant_disparities:
            if 'feature' in disparity:
                feature = disparity['feature']
                logger.info(f"\nAttempting bias mitigation for feature: {feature}")
                
                # Try mitigation
                mitigated_model = bias_analyzer.mitigate_bias(X_val, y_val, feature)
                
                # Analyze results for this feature
                temp_analyzer = BiasAnalyzer(mitigated_model)
                feature_results = temp_analyzer.analyze_bias(X_val, y_val)
                feature_report = temp_analyzer.generate_bias_report(feature_results)
                
                # Check if mitigation helped this feature
                if feature in feature_report.get('significant_disparities', []):
                    logger.warning(f"Mitigation for {feature} did not meet fairness thresholds")
                else:
                    logger.info(f"Successfully mitigated bias for {feature}")
                
                model = mitigated_model
        
        # Final bias check
        logger.info("\nPerforming final bias analysis...")
        bias_analyzer = BiasAnalyzer(mitigated_model)
        new_results = bias_analyzer.analyze_bias(X_val, y_val)
        new_report = bias_analyzer.generate_bias_report(new_results)
        
        if new_report['bias_detected']:
            remaining_disparities = len(new_report.get('significant_disparities', []))
            logger.warning(
                f"Bias mitigation partially successful. "
                f"Remaining disparities in {remaining_disparities} features"
            )
        else:
            logger.info("Bias successfully mitigated across all features")
            bias_report = new_report
    else:
        logger.info("No significant bias detected in initial analysis")
    
    return model, bias_report

def train_model():
    training_context = {
        "stage": "initialization",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        setup_mlflow()
        logger.info("MLflow configured successfully")
        
        # Load raw data first
        X_train_raw, X_train, X_test_raw, X_test, X_val_raw, X_val, y_train, y_val, y_test, encoders = load_data()
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
                
                # Step 1: Model Validation
                logger.info("Running model validation...")
                validation_passed, validation_metrics, _ = validate_model(
                    model=best_model.model if hasattr(best_model, 'model') else best_model,
                    run_id=parent_run.info.run_id,
                    X_val=X_val,
                    y_val=y_val
                )
                
                # Step 2: Bias Analysis and Mitigation
                logger.info("Checking for socioeconomic bias...")
                initial_model = best_model.model if hasattr(best_model, 'model') else best_model
                
                # First attempt with standard mitigation
                mitigated_model, bias_report = evaluate_and_mitigate_bias(
                    initial_model, X_val, y_val
                )
                
                # Check if initial mitigation was successful
                if bias_report['bias_detected']:
                    logger.info("Initial mitigation incomplete, attempting stronger mitigation...")
                    # Try again with the mitigated model as a starting point
                    mitigated_model, bias_report = evaluate_and_mitigate_bias(
                        mitigated_model, X_val, y_val
                    )
                    
                    # If still biased, try one final time with stronger weights
                    if bias_report['bias_detected']:
                        logger.info("Second mitigation incomplete, attempting final mitigation...")
                        analyzer = BiasAnalyzer(mitigated_model, constraint_weight=0.8)  # Stronger constraint
                        mitigated_model, bias_report = analyzer.analyze_bias(X_val, y_val), analyzer.generate_bias_report({})
                
                best_model = mitigated_model
                
                # Send training report
                notify_training_completion(
                    model_name=best_model.name if hasattr(best_model, 'name') else best_model.__class__.__name__,
                    metrics=best_metrics,
                    plots=all_plots,
                    run_id=parent_run.info.run_id,
                    validation_thresholds=VALIDATION_THRESHOLDS,
                    bias_report=bias_report,
                    success=validation_passed
                )
                
                model_name = best_model.name if hasattr(best_model, 'name') else best_model.__class__.__name__
                logger.info(f"Best model: {model_name}")
                logger.info(f"Best model metrics: {best_metrics}")
                
                # Log final statuses separately
                logger.info(f"Performance Validation: {'✓' if validation_passed else '✗'}")
                logger.info(f"Fairness Check: {'✓' if not bias_report['bias_detected'] else '✗'}")
                
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
