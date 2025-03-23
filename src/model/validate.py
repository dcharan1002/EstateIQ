import os
import json
import logging
import mlflow
import numpy as np
from datetime import datetime
from .utils.notifications import notify_error
from .utils.bias_analysis import BiasAnalyzer
from .utils.visualization import create_metrics_visualization
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load validation thresholds from environment
VALIDATION_THRESHOLDS = {
    'r2': float(os.getenv('VALIDATION_R2_THRESHOLD', 0.8)),
    'rmse': float(os.getenv('VALIDATION_RMSE_THRESHOLD', 1.0)),
    'mae': float(os.getenv('VALIDATION_MAE_THRESHOLD', 0.8))
}

# Load bias thresholds from environment
BIAS_THRESHOLDS = {
    'prediction_disparity': float(os.getenv('BIAS_PREDICTION_DISPARITY_THRESHOLD', 0.1)),
    'performance_disparity': float(os.getenv('BIAS_PERFORMANCE_DISPARITY_THRESHOLD', 0.1))
}

def check_bias(model, X, y):    
    # Initialize bias analyzer
    analyzer = BiasAnalyzer(model)
    
    # Perform bias analysis
    bias_results = analyzer.analyze_bias(X, y)
    
    # Generate comprehensive report
    significant_disparities = analyzer.generate_bias_report(bias_results)
    
    logger.info(f"Significant disparities: {significant_disparities}")
    bias_detected = len(significant_disparities) > 0
    
    return {
        'bias_detected': bias_detected,
        'details': bias_results,
        'significant_disparities': significant_disparities,
        'summary': {
            'status': 'Bias Detected' if bias_detected else 'No Significant Bias',
            'features_analyzed': len(bias_results),
            'features_with_bias': len(significant_disparities)
        }
    }

def validate_model(model, run_id, X_val, y_val):
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

        # Create validation results directory
        results_dir = Path("results/validation")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Check performance metrics
        validation_passed = all(
            metrics[metric] >= threshold
            for metric, threshold in VALIDATION_THRESHOLDS.items()
        )

        # Save validation results to JSON
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "metrics": metrics,
            "thresholds": VALIDATION_THRESHOLDS,
            "validation_passed": validation_passed
        }
        
        results_path = results_dir / "validation_latest.json"
        with open(results_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation results saved to {results_path}")

        # Log results to MLflow
        with mlflow.start_run(run_id=run_id, nested=True) as validation_run:
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(metrics_plot)
            mlflow.log_artifact(str(results_path))

        if not validation_passed:
            logger.warning("Model validation failed: Performance metrics below thresholds")
        else:
            logger.info("Model validation passed: Performance metrics meet thresholds")

        return validation_passed, metrics, {}


    except Exception as e:
        notify_error(e, validation_context)
        raise
