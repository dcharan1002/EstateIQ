import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(model, X_test, y_test):
    try:
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'r2': float(r2_score(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'mape': float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
        }
        
        # Calculate additional detailed metrics
        residuals = y_test - y_pred
        metrics.update({
            'std_residuals': float(np.std(residuals)),
            'max_error': float(np.max(np.abs(residuals))),
            'median_error': float(np.median(np.abs(residuals))),
            'mean_prediction': float(np.mean(y_pred)),
            'std_prediction': float(np.std(y_pred))
        })
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise
