import shap
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import numpy as np

def compute_shap_importance(model, X_train, feature_names, model_name):
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    
    # Sample data to speed up SHAP computation
    X_sampled = X_train.sample(n=min(5000, len(X_train)), random_state=42)
    X_sampled = X_sampled.astype(float)


    # Compute SHAP values
    explainer = shap.Explainer(model, X_sampled)
    shap_values = explainer(X_sampled)

    # Create and save SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sampled, show=False)

    # Define path for saving SHAP images
    shap_plot_path = f"shap_importance_{model_name}.png"
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()

    print(f"SHAP importance plot saved: {shap_plot_path}")

    # Log image as artifact in MLflow
    mlflow.log_artifact(shap_plot_path, artifact_path="shap_analysis")


    return shap_plot_path
