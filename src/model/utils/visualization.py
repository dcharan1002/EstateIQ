import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_residuals(y_true, y_pred, model_name):
    """Plot residuals distribution."""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title(f"{model_name} Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    residual_plot_path = f"{model_name}_residuals.png"
    plt.savefig(residual_plot_path)
    plt.close()
    return residual_plot_path

def plot_predictions(y_true, y_pred, model_name):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"{model_name} Predictions vs Actual")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.tight_layout()
    
    pred_plot_path = f"{model_name}_predictions.png"
    plt.savefig(pred_plot_path)
    plt.close()
    return pred_plot_path

def plot_model_comparison(y_test, rf_preds, xgb_preds):
    """Plot model comparison of residuals."""
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_test - rf_preds, label='Random Forest Residuals', shade=True)
    sns.kdeplot(y_test - xgb_preds, label='XGBoost Residuals', shade=True)
    plt.legend()
    plt.title("Residual Distribution Comparison")
    plt.xlabel("Residuals")
    plt.tight_layout()
    
    comparison_plot_path = "model_comparison.png"
    plt.savefig(comparison_plot_path)
    plt.close()
    return comparison_plot_path

def plot_hyperparameter_sensitivity(results_df, model_name):
    """Plot hyperparameter sensitivity analysis."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, markers=True)
    plt.title(f"{model_name} Hyperparameter Sensitivity")
    plt.xlabel("Hyperparameter Configurations")
    plt.ylabel("Performance Metric (e.g., R2 Score)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    sensitivity_plot_path = f"{model_name}_hyperparameter_sensitivity.png"
    plt.savefig(sensitivity_plot_path)
    plt.close()
    return sensitivity_plot_path

def create_metrics_visualization(metrics, thresholds):
    """Create visualization comparing metrics against thresholds."""
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    metric_names = list(metrics.keys())
    values = list(metrics.values())
    threshold_values = [thresholds.get(m, 0) for m in metric_names]
    
    x = range(len(metric_names))
    width = 0.35
    
    # Create grouped bar chart
    plt.bar([i - width/2 for i in x], values, width, label='Actual', color='skyblue')
    plt.bar([i + width/2 for i in x], threshold_values, width, label='Threshold', 
            color='lightcoral', alpha=0.6)
    
    # Customize plot
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Performance Metrics vs Thresholds')
    plt.xticks(x, metric_names)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(threshold_values):
        plt.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    metrics_plot_path = "metrics_comparison.png"
    plt.savefig(metrics_plot_path)
    plt.close()
    
    return metrics_plot_path

def plot_bias_analysis(bias_report, feature):
    """Create detailed bias analysis visualization for a feature."""
    if feature not in bias_report['details']:
        return None
    
    feature_metrics = bias_report['details'][feature]['group_metrics']
    groups = list(feature_metrics.keys())
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Predictions vs Actuals by Group
    ax1 = fig.add_subplot(gs[0, :])
    plot_data = pd.DataFrame({
        'Group': groups,
        'Mean Prediction': [feature_metrics[g]['mean_prediction'] for g in groups],
        'Mean Actual': [feature_metrics[g]['mean_actual'] for g in groups],
        'Sample Size': [feature_metrics[g]['size'] for g in groups]
    })
    
    # Create bar plot with predictions and actuals
    x = np.arange(len(groups))
    width = 0.35
    ax1.bar(x - width/2, plot_data['Mean Actual'], width, label='Actual', color='skyblue')
    ax1.bar(x + width/2, plot_data['Mean Prediction'], width, label='Predicted', color='lightcoral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=45)
    ax1.set_title(f'Mean Predictions vs Actuals by {feature}')
    ax1.legend()
    
    # 2. RMSE by Group
    ax2 = fig.add_subplot(gs[1, 0])
    rmse_values = [feature_metrics[g]['rmse'] for g in groups]
    ax2.bar(groups, rmse_values, color='lightseagreen')
    ax2.set_title(f'RMSE by {feature}')
    ax2.set_xticklabels(groups, rotation=45)
    
    # 3. Sample Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    sizes = [feature_metrics[g]['size'] for g in groups]
    ax3.pie(sizes, labels=groups, autopct='%1.1f%%')
    ax3.set_title('Sample Distribution')
    
    # Add disparity information
    disparities = bias_report['details'][feature]
    disparity_text = (
        f"Prediction Disparity: {disparities['prediction_disparity']:.3f}\n"
        f"Performance Disparity: {disparities['performance_disparity']:.3f}"
    )
    plt.figtext(0.95, 0.02, disparity_text, ha='right', va='bottom', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.suptitle(f'Bias Analysis for {feature}', y=0.95, fontsize=14)
    plt.tight_layout()
    
    # Save plot
    bias_plot_path = f"bias_analysis_{feature}.png"
    plt.savefig(bias_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return bias_plot_path
