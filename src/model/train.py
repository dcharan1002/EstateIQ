import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging
import pandas as pd
import joblib
from datetime import datetime

import numpy as np
from sqlalchemy import Column
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
from .validate import validate_model, VALIDATION_THRESHOLDS
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Configure MLflow with environment-based tracking"""
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'estate_price_prediction')
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow configured with tracking URI: {tracking_uri}")
    logger.info(f"MLflow experiment name: {experiment_name}")

def select_features(X):
    """Select the 15 most important features from the dataset"""
    # Select base numeric features
    numeric_features = [
        'GROSS_AREA', 'LIVING_AREA', 'LAND_SF', 'YR_BUILT',
        'BED_RMS', 'FULL_BTH', 'HLF_BTH', 'NUM_PARKING',
        'FIREPLACES', 'KITCHENS', 'TT_RMS', 'ZIP_CODE',
        'YR_REMODEL'
    ]
    
    # Select categorical features (already one-hot encoded)
    categorical_features = [
        # Structure and material features
        'STRUCTURE_CLASS_C - BRICK/CONCR',
        'STRUCTURE_CLASS_D - WOOD/FRAME',
        'STRUCTURE_CLASS_B - REINF CONCR',
        
        # Interior condition features
        'INT_COND_E - EXCELLENT',
        'INT_COND_G - GOOD',
        'INT_COND_A - AVERAGE',
        'INT_COND_F - FAIR',
        'INT_COND_P - POOR',
        
        # Overall condition features
        'OVERALL_COND_E - EXCELLENT',
        'OVERALL_COND_VG - VERY GOOD',
        'OVERALL_COND_G - GOOD',
        'OVERALL_COND_A - AVERAGE',
        'OVERALL_COND_F - FAIR',
        'OVERALL_COND_P - POOR',
        
        # Kitchen features
        'KITCHEN_STYLE2_M - MODERN',
        'KITCHEN_STYLE2_L - LUXURY',
        'KITCHEN_STYLE2_S - SEMI-MODERN',
        'KITCHEN_TYPE_F - FULL EAT IN',
        
        # Amenities and systems
        'AC_TYPE_C - CENTRAL AC',
        'AC_TYPE_D - DUCTLESS AC',
        'HEAT_TYPE_F - FORCED HOT AIR',
        'HEAT_TYPE_W - HT WATER/STEAM',
        
        # Property characteristics
        'PROP_VIEW_E - EXCELLENT',
        'PROP_VIEW_G - GOOD',
        'CORNER_UNIT_Y - YES',
        'ORIENTATION_E - END',
        'ORIENTATION_F - FRONT/STREET',
        
        # Exterior features
        'EXT_COND_E - EXCELLENT',
        'EXT_COND_G - GOOD',
        'ROOF_COVER_S - SLATE',
        'ROOF_COVER_A - ASPHALT SHINGL'
    ]
    
    # Check which columns are actually in the DataFrame
    available_numeric = [col for col in numeric_features if col in X.columns]
    available_categorical = [col for col in categorical_features if col in X.columns]
    
    # Copy selected features
    selected = X[available_numeric + available_categorical].copy()
    
    # Add derived features
    selected['property_age'] = 2025 - selected['YR_BUILT'].astype(int)
    selected['total_bathrooms'] = selected['FULL_BTH'] + 0.5 * selected['HLF_BTH']
    # Calculate ratios with safe division
    selected['living_area_ratio'] = np.where(
        selected['GROSS_AREA'] > 0,
        selected['LIVING_AREA'] / selected['GROSS_AREA'],
        0
    )
    
    # Add new derived features
    if 'YR_REMODEL' in selected.columns:
        selected['years_since_renovation'] = 2025 - selected['YR_REMODEL'].fillna(selected['YR_BUILT']).astype(int)
        selected['has_renovation'] = (selected['YR_REMODEL'] > selected['YR_BUILT']).astype(int)
    
    if 'LAND_SF' in selected.columns and 'GROSS_AREA' in selected.columns:
        selected['floor_area_ratio'] = np.where(
            selected['LAND_SF'] > 0,
            selected['GROSS_AREA'] / selected['LAND_SF'],
            0
        )
        selected['non_living_area'] = np.maximum(0, selected['GROSS_AREA'] - selected['LIVING_AREA'])
    
    if 'TT_RMS' in selected.columns and 'LIVING_AREA' in selected.columns:
        selected['rooms_per_area'] = np.where(
            selected['LIVING_AREA'] > 0,
            selected['TT_RMS'] / selected['LIVING_AREA'],
            0
        )
    
    # Replace any remaining infinities or NaNs with 0
    selected = selected.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Create composite condition scores
    condition_map = {'E': 5, 'VG': 4.5, 'G': 4, 'A': 3, 'F': 2, 'P': 1}
    
    # Helper function to calculate condition score
    def calculate_condition_score(row, condition_cols):
        max_score = 3  # default score
        for col in condition_cols:
            if row[col]:  # if this condition is true
                # Extract condition code (e.g., 'E' from 'INT_COND_E - EXCELLENT')
                cond_code = col.split(' - ')[0].split('_')[-1]
                score = condition_map.get(cond_code, 0)
                max_score = max(max_score, score)
        return max_score

    # Calculate condition scores
    int_cond_cols = [col for col in selected.columns if col.startswith('INT_COND_')]
    if int_cond_cols:
        selected['interior_score'] = selected.apply(
            lambda row: calculate_condition_score(row, int_cond_cols), axis=1
        )
    
    ext_cond_cols = [col for col in selected.columns if col.startswith('EXT_COND_')]
    if ext_cond_cols:
        selected['exterior_score'] = selected.apply(
            lambda row: calculate_condition_score(row, ext_cond_cols), axis=1
        )
    
    overall_cond_cols = [col for col in selected.columns if col.startswith('OVERALL_COND_')]
    if overall_cond_cols:
        selected['overall_score'] = selected.apply(
            lambda row: calculate_condition_score(row, overall_cond_cols), axis=1
        )
    
    return selected

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
        # Read CSV with more robust parsing settings
        csv_options = {
            'on_bad_lines': 'skip',  # Skip problematic rows
            'escapechar': '\\',      # Handle escaped characters
            'quoting': 1,            # Quote mode QUOTE_ALL
            'encoding': 'utf-8'      # Explicit encoding
        }
        
        # Read the CSV files
        X_train = pd.read_csv(data_dir / "X_train.csv", **csv_options).head(1000)
        X_test = pd.read_csv(data_dir / "X_test.csv", **csv_options).head(1000)
        y_train = pd.read_csv(data_dir / "y_train.csv", **csv_options).squeeze().iloc[:1000]
        y_test = pd.read_csv(data_dir / "y_test.csv", **csv_options).squeeze().iloc[:1000]

        # Clean column names by stripping whitespace
        X_train.columns = X_train.columns.str.strip()
        X_test.columns = X_test.columns.str.strip()
        logger.info(f"Loaded data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
        
        X_train = select_features(X_train)
        X_test = select_features(X_test)
        
        # Log column names to help with debugging
        logger.info("Available columns in X_train:")
        for col in X_train.columns:
            logger.info(f"  {col}")
        
        logger.info("Available columns in X_test:")
        for col in X_test.columns:
            logger.info(f"  {col}")
        
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

def create_hyperparameter_results(model, metrics: dict[str, float] | None) -> pd.DataFrame:
    if metrics is None:
        return pd.DataFrame([{
            'config': 'tuned',
            'r2': 0.0,
            'rmse': 0.0,
            'mae': 0.0
        }])
    return pd.DataFrame([{
        'config': 'tuned',
        'r2': metrics.get('r2', 0.0),
        'rmse': metrics.get('rmse', 0.0),
        'mae': metrics.get('mae', 0.0)
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
            'random_forest': {
                'n_estimators': 500,
                'max_features': 'sqrt',
                'min_samples_leaf': 2,
                'max_depth': 20
            },
            'xgboost': {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 8,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
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
                    
                    # First tune hyperparameters
                    logger.info(f"Tuning hyperparameters for {model_name}...")
                    model.tune_hyperparameters(X_train, y_train)
                    
                    # Then train with tuned parameters and validation data if XGBoost
                    if model_name == 'xgboost':
                        model.train(X_train, y_train, X_val, y_val)
                    else:
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
                    # logger.info(f"Generating SHAP analysis for {model_name}...")
                    # shap_analysis = generate_shap_analysis(
                    #     model.model if hasattr(model, 'model') else model,
                    #     X_test,
                    #     results_dir=model_results_dir # type: ignore
                    # )
                    
                    # # Add SHAP plots
                    # if shap_analysis and 'plots' in shap_analysis:
                    #     for plot_name, plot_path in shap_analysis['plots'].items():
                    #         plots[f"{model_name} {plot_name}"] = plot_path
                    #         if os.path.exists(plot_path):
                    #             mlflow.log_artifact(plot_path)
                    
                    # Store all plots for this model
                    all_plots.update(plots)
                    
                    if metrics is not None:
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
                if best_metrics is not None:
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
                    model_name=getattr(best_model, 'name', best_model.__class__.__name__),
                    metrics=best_metrics,
                    plots=all_plots,
                    run_id=parent_run.info.run_id,
                    validation_thresholds=VALIDATION_THRESHOLDS,
                    bias_report=bias_report,
                    success=validation_passed
                )
                
                model_name = getattr(best_model, 'name', best_model.__class__.__name__)
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
