import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


# 1. Residual Plot
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title(f"{model_name} Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    residual_plot_path = f"{model_name}_residuals.png"
    plt.savefig(residual_plot_path)
    mlflow.log_artifact(residual_plot_path)
    plt.close()
    return residual_plot_path

# 2. Predictions vs Actual
def plot_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"{model_name} Predictions vs Actual")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.tight_layout()
    
    pred_plot_path = f"{model_name}_predictions.png"
    plt.savefig(pred_plot_path)
    mlflow.log_artifact(pred_plot_path)
    plt.close()
    return pred_plot_path

# 3. Model Comparison
def plot_model_comparison(y_test, rf_preds, xgb_preds):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_test - rf_preds, label='Random Forest Residuals', shade=True)
    sns.kdeplot(y_test - xgb_preds, label='XGBoost Residuals', shade=True)
    plt.legend()
    plt.title("Residual Distribution Comparison")
    plt.xlabel("Residuals")
    plt.tight_layout()
    
    comparison_plot_path = "model_comparison.png"
    plt.savefig(comparison_plot_path)
    mlflow.log_artifact(comparison_plot_path)
    plt.close()
    return comparison_plot_path

# 4. Hyperparameter Sensitivity Plot
def plot_hyperparameter_sensitivity(results_df, model_name):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, markers=True)
    plt.title(f"{model_name} Hyperparameter Sensitivity")
    plt.xlabel("Hyperparameter Configurations")
    plt.ylabel("Performance Metric (e.g., R2 Score)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    sensitivity_plot_path = f"{model_name}_hyperparameter_sensitivity.png"
    plt.savefig(sensitivity_plot_path)
    mlflow.log_artifact(sensitivity_plot_path)
    plt.close()
    return sensitivity_plot_path
