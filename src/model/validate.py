import os
import json
import logging
import joblib
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
from utils.notifications import notify_error
from utils.visualization import plot_bias_analysis, create_metrics_visualization
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validation thresholds
VALIDATION_THRESHOLDS = {
    'r2': 0.8,
    'rmse': 1.0,
    'mae': 0.8
}

# Bias thresholds
BIAS_THRESHOLDS = {
    'prediction_disparity': 0.1,
    'performance_disparity': 0.1
}

# Features to check for bias
BIAS_CHECK_FEATURES = [
    'STRUCTURE_CLASS',  # Building structure classification
    'OWNER_OCC',       # Owner occupied status
    'OVERALL_COND',    # Overall condition
    'INT_COND'         # Interior condition
]

def check_bias(model, X, y, sensitive_features=BIAS_CHECK_FEATURES):
    """Check for bias across different demographic and structural groups"""
    bias_report = {
        'bias_detected': False,
        'details': {},
        'plots': {},
        'summary': {}
    }

    predictions = model.predict(X)
    available_features = [f for f in sensitive_features if f in X.columns]

    if not available_features:
        logger.warning("No bias check features found in dataset")
        return bias_report

    for feature in available_features:
        feature_values = X[feature].unique()
        group_metrics = {}

        # Calculate metrics for each group
        for group in feature_values:
            mask = X[feature] == group
            if sum(mask) == 0:
                continue

            group_preds = predictions[mask]
            group_truth = y[mask]

            group_metrics[str(group)] = {
                'size': int(sum(mask)),
                'mean_prediction': float(np.mean(group_preds)),
                'mean_actual': float(np.mean(group_truth)),
                'rmse': float(np.sqrt(np.mean((group_preds - group_truth) ** 2)))
            }

        # Calculate disparities
        if group_metrics:
            mean_predictions = [m['mean_prediction'] for m in group_metrics.values()]
            mean_rmse = [m['rmse'] for m in group_metrics.values()]

            prediction_disparity = max(mean_predictions) - min(mean_predictions)
            performance_disparity = max(mean_rmse) - min(mean_rmse)

            feature_bias = {
                'prediction_disparity': prediction_disparity,
                'performance_disparity': performance_disparity,
                'group_metrics': group_metrics
            }

            bias_report['details'][feature] = feature_bias

            # Generate and store bias analysis plot
            bias_plot_path = plot_bias_analysis(bias_report, feature)
            if bias_plot_path:
                bias_report['plots'][feature] = bias_plot_path

            # Check if any threshold is exceeded
            if (prediction_disparity > BIAS_THRESHOLDS['prediction_disparity'] or
                performance_disparity > BIAS_THRESHOLDS['performance_disparity']):
                bias_report['bias_detected'] = True
                bias_report['summary'][feature] = {
                    'status': 'Bias Detected',
                    'prediction_disparity': prediction_disparity,
                    'performance_disparity': performance_disparity
                }
            else:
                bias_report['summary'][feature] = {
                    'status': 'No Significant Bias',
                    'prediction_disparity': prediction_disparity,
                    'performance_disparity': performance_disparity
                }

    return bias_report

def validate_model(model, run_id, X_val, y_val):
    """Run model validation and bias detection with provided data"""
    validation_context = {
        "stage": "initialization",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "run_id": run_id
    }
    
    try:
        # Make predictions
        y_pred = model.predict(X_val)
        validation_context["stage"] = "predictions_made"

        # Calculate metrics
        metrics = {
            'r2': float(np.corrcoef(y_pred, y_val)[0, 1] ** 2),
            'rmse': float(np.sqrt(np.mean((y_pred - y_val) ** 2))),
            'mae': float(np.mean(np.abs(y_pred - y_val)))
        }
        validation_context.update({
            "stage": "metrics_calculated",
            "metrics": metrics
        })

        # Generate metrics visualization
        metrics_plot = create_metrics_visualization(metrics, VALIDATION_THRESHOLDS)
        validation_context["metrics_plot"] = metrics_plot

        # Check bias across multiple features
        bias_report = check_bias(model, X_val, y_val)
        validation_context.update({
            "stage": "bias_checked",
            "bias_detected": bias_report['bias_detected']
        })

        # Create validation results directory
        results_dir = Path("results/validation")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Log results to MLflow
        with mlflow.start_run(run_id=run_id, nested=True) as validation_run:
            # Log metrics and plots
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(metrics_plot)
            
            # Log bias analysis results
            if bias_report:
                mlflow.log_dict(bias_report, "bias_report.json")
                for feature, plot_path in bias_report.get('plots', {}).items():
                    if plot_path and Path(plot_path).exists():
                        mlflow.log_artifact(plot_path)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'timestamp': timestamp,
                'run_id': run_id,
                'metrics': metrics,
                'bias_report': bias_report,
                'validation_thresholds': VALIDATION_THRESHOLDS,
                'bias_thresholds': BIAS_THRESHOLDS
            }

            # Save results locally
            results_path = results_dir / f"validation_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

        # Check if validation passes
        validation_passed = all(
            metrics[metric] >= threshold
            for metric, threshold in VALIDATION_THRESHOLDS.items()
        )

        if not validation_passed or (bias_report and bias_report['bias_detected']):
            logger.warning("Validation failed or bias detected")
            return False, metrics, bias_report

        logger.info("Model validation successful")
        return True, metrics, bias_report

    except Exception as e:
        notify_error(e, validation_context)
        raise
