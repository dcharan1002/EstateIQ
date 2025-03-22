import shap
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
import os
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

logger = logging.getLogger(__name__)

def generate_shap_analysis(model, X_test, results_dir="results", max_display=10):
    """
    Generate SHAP values and plots for model interpretation
    
    Args:
        model: Trained model instance
        X_test: Test features
        results_dir: Directory to save plots and analysis
        max_display: Maximum number of features to display in summary plots
        
    Returns:
        dict: Dictionary containing SHAP values and plot paths
    """
    try:
        # Ensure results directory exists
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame if needed
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
            
        # Use TreeExplainer for tree-based models (RF and XGBoost)
        if isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
            explainer = shap.TreeExplainer(model)
            # For tree models, we can directly compute SHAP values
            shap_values = explainer.shap_values(X_test)
            expected_value = explainer.expected_value
        else:
            # Fallback to KernelExplainer for other models
            sample_size = min(100, len(X_test))
            background = shap.sample(X_test, sample_size)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test)
            expected_value = explainer.expected_value
            
        # Generate and save summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_test,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        summary_plot_path = results_dir / "shap_summary_plot.png"
        plt.savefig(summary_plot_path)
        mlflow.log_artifact(str(summary_plot_path))
        plt.close()
        
        # Generate waterfall plot for the most important prediction
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=X_test.iloc[0],
            feature_names=X_test.columns
        ), show=False)
        waterfall_plot_path = results_dir / "shap_waterfall_plot.png"
        plt.savefig(waterfall_plot_path)
        mlflow.log_artifact(str(waterfall_plot_path))
        plt.close()
        
        # Generate and save detailed analysis
        analysis = {
            'shap_values': shap_values,
            'expected_value': expected_value,
            'feature_importance': {},
            'feature_interactions': {},
            'summary_stats': {}
        }
        
        # Calculate and log feature importance
        feature_importance = np.abs(shap_values).mean(0)
        importance_dict = {}
        for idx, importance in enumerate(feature_importance):
            feature_name = X_test.columns[idx]
            importance_val = float(importance)
            importance_dict[feature_name] = importance_val
            analysis['feature_importance'][feature_name] = importance_val
            # Log individual feature importance
            mlflow.log_metric(f"shap_importance_{feature_name}", importance_val)
        
        # Save and log feature importance CSV
        importance_df = pd.DataFrame(
            importance_dict.items(), 
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        importance_path = results_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))
        
        # Calculate and log summary statistics
        for feature in X_test.columns:
            feature_shap = shap_values[:, X_test.columns.get_loc(feature)]
            stats = {
                'mean_impact': float(np.mean(feature_shap)),
                'abs_mean_impact': float(np.mean(np.abs(feature_shap))),
                'std_impact': float(np.std(feature_shap)),
                'max_impact': float(np.max(np.abs(feature_shap)))
            }
            analysis['summary_stats'][feature] = stats
            
            # Log summary stats to MLflow
            for stat_name, stat_value in stats.items():
                mlflow.log_metric(f"shap_{feature}_{stat_name}", stat_value)
        
        # Generate feature interaction plots for top features
        plt.figure(figsize=(15, 10))
        top_features = sorted(
            analysis['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for idx, (feature_name, _) in enumerate(top_features):
            feature_idx = X_test.columns.get_loc(feature_name)
            plt.subplot(2, 3, idx + 1)
            shap.dependence_plot(
                feature_idx,
                shap_values,
                X_test,
                show=False,
                ax=plt.gca()
            )
            plt.title(f"SHAP Dependence\n{feature_name}")
        
        plt.tight_layout()
        interaction_plot_path = results_dir / "shap_interaction_plots.png"
        plt.savefig(interaction_plot_path)
        mlflow.log_artifact(str(interaction_plot_path))
        plt.close()
        
        # Save analysis summary
        analysis['plots'] = {
            'summary_plot': str(summary_plot_path),
            'waterfall_plot': str(waterfall_plot_path),
            'interaction_plots': str(interaction_plot_path)
        }
        
        # Log SHAP values summary
        shap_summary = {
            'mean_abs_shap': float(np.mean(np.abs(shap_values))),
            'max_abs_shap': float(np.max(np.abs(shap_values))),
            'total_shap_importance': float(np.sum(feature_importance))
        }
        mlflow.log_metrics(shap_summary)
        
        # Log key insights
        logger.info("\nTop influential features:")
        for feature, importance in top_features:
            logger.info(f"{feature}: {importance:.4f}")
            
        return analysis
        
    except Exception as e:
        logger.error(f"Error generating SHAP analysis: {e}")
        raise
