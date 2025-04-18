import shap
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

logger = logging.getLogger(__name__)

def generate_shap_analysis(model, X_test, results_dir="results", max_display=10):
    try:
        # Ensure results directory exists
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame if needed and sample data
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        
        # Create small sample for SHAP analysis
        sample_size = min(100, len(X_test))
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
            
        # Use TreeExplainer for tree-based models with optimized settings
        if isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
            explainer = shap.TreeExplainer(
                model,
                feature_perturbation="tree_path_dependent",
                approximate=True,  # Use fast approximation
            )
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_sample)
            expected_value = explainer.expected_value
        else:
            # Fallback to KernelExplainer with minimal background set
            background_size = min(20, len(X_test_sample))
            background = shap.sample(X_test_sample, background_size)
            explainer = shap.KernelExplainer(
                model.predict,
                background,
                nsamples=50 
            )
            shap_values = explainer.shap_values(X_test_sample)
            expected_value = explainer.expected_value
            
        # Generate and save summary plot
        plt.figure(figsize=(8, 5))
        shap.summary_plot(
            shap_values,
            X_test_sample,
            plot_type="bar",
            max_display=max_display,
            show=False,
            plot_size=(8, 6) 
        )
        plt.tight_layout()
        summary_plot_path = results_dir / "shap_summary_plot.png"
        plt.savefig(summary_plot_path)
        mlflow.log_artifact(str(summary_plot_path))
        plt.close()
        
        # Generate waterfall plot for most important prediction 
        plt.figure(figsize=(8, 6)) 
        most_impactful_idx = np.argmax(np.abs(shap_values).sum(axis=1))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[most_impactful_idx],
            base_values=expected_value,
            data=X_test_sample.iloc[most_impactful_idx],
            feature_names=X_test_sample.columns
        ), show=False)
        waterfall_plot_path = results_dir / "shap_waterfall_plot.png"
        plt.savefig(waterfall_plot_path)
        mlflow.log_artifact(str(waterfall_plot_path))
        plt.close()
        
        with np.errstate(divide='ignore', invalid='ignore'): 
            feature_importance = np.abs(shap_values).mean(0)
            feature_importance = np.nan_to_num(feature_importance)
        
        importance_pairs = sorted(
            [(col, float(imp)) for col, imp in zip(X_test_sample.columns, feature_importance)],
            key=lambda x: x[1],
            reverse=True
        )[:max_display]
        importance_dict = dict(importance_pairs)
        del feature_importance
        
        # Save to CSV
        importance_df = pd.DataFrame(
            importance_dict.items(),
            columns=['feature', 'importance']
        )
        importance_path = results_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))
        
        # Log metrics
        for feature, importance in importance_dict.items():
            mlflow.log_metric(f"shap_importance_{feature}", importance)
        
        # Generate single interaction plot for most important feature
        plt.figure(figsize=(8, 4))
        top_feature = list(importance_dict.keys())[0]
        feature_idx = X_test_sample.columns.get_loc(top_feature)
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_test_sample,
            show=False,
            ax=plt.gca(),
            dot_size=20
        )
        plt.title(f"SHAP: {top_feature}")
        plt.tight_layout()
        interaction_plot_path = results_dir / "shap_interaction_plots.png"
        plt.savefig(interaction_plot_path)
        mlflow.log_artifact(str(interaction_plot_path))
        plt.close()
        
        mean_abs_shap = float(np.mean(np.abs(shap_values)))
        mlflow.log_metric('mean_abs_shap', mean_abs_shap)
        
        # Log key insights
        logger.info("\nTop influential features:")
        for feature, importance in importance_dict.items():
            logger.info(f"{feature}: {importance:.4f}")
            
        # Return minimal analysis results
        return {
            'feature_importance': importance_dict,
            'plots': {
                'summary_plot': str(summary_plot_path),
                'waterfall_plot': str(waterfall_plot_path),
                'interaction_plot': str(interaction_plot_path)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating SHAP analysis: {e}")
        raise
