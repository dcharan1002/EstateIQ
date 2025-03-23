import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sklearn.metrics as skm
from sklearn.base import BaseEstimator, clone
import matplotlib.pyplot as plt
import json
from fairlearn.metrics import MetricFrame, make_derived_metric
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.reductions import BoundedGroupLoss
import warnings

logger = logging.getLogger(__name__)

def mean_prediction(y_true, y_pred):
    """Custom metric for mean predictions"""
    return np.mean(y_pred)

def abs_prediction_difference(y_true, y_pred):
    """Custom metric for absolute difference between predictions and true values"""
    return np.mean(np.abs(y_pred - y_true))

class BiasAnalyzer:
    def __init__(self, model, constraint_weight=0.5):
        self.model = model
        self.results_dir = Path("results/bias_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Fairness constraint weight for bias mitigation
        self.constraint_weight = constraint_weight
        
        # Define sensitive features and their privileged values
        self.sensitive_features = {
            'OWN_OCC': {
                'privileged_value': 1,
                'description': 'Owner-occupied properties'
            },
            'ZIP_CODE': {
                'privileged_value': None,  # Will be determined dynamically
                'description': 'High-value ZIP codes'
            }
        }
        
        # Minimum samples required for valid group analysis
        self.min_group_size = 5
        
        # Threshold for identifying high-value zip codes
        self.zip_value_threshold = None
        
        # Fairness metrics to compute
        self.metrics = {
            'rmse': lambda y_true, y_pred: np.sqrt(skm.mean_squared_error(y_true, y_pred)),
            'r2': skm.r2_score,
            'mae': skm.mean_absolute_error,
            'mean_prediction': mean_prediction,
            'prediction_difference': abs_prediction_difference
        }

    def calculate_metrics(self, y_true, y_pred):
        """Calculate metrics safely with size checks"""
        if len(y_true) < self.min_group_size:
            return None

        try:
            return {
                'rmse': float(np.sqrt(skm.mean_squared_error(y_true, y_pred))),
                'r2': float(skm.r2_score(y_true, y_pred)),
                'mae': float(skm.mean_absolute_error(y_true, y_pred))
            }
        except:
            return None

    def determine_zip_code_groups(self, X, y):
        """Determine high-value ZIP codes based on median property values"""
        zip_medians = {}
        for zip_code in X['ZIP_CODE'].unique():
            mask = X['ZIP_CODE'] == zip_code
            if sum(mask) >= self.min_group_size:
                zip_medians[zip_code] = np.median(y[mask])
        
        if zip_medians:
            self.zip_value_threshold = np.median(list(zip_medians.values()))
            high_value_zips = {zip_code for zip_code, median in zip_medians.items() 
                             if median > self.zip_value_threshold}
            
            # Update sensitive feature definition
            self.sensitive_features['ZIP_CODE']['privileged_value'] = high_value_zips
            
            logger.info(f"ZIP_CODE threshold set to {self.zip_value_threshold}")
            logger.info(f"Identified {len(high_value_zips)} high-value ZIP codes")
            
            # Create binary ZIP_CODE feature for fairness analysis
            X['HIGH_VALUE_ZIP'] = X['ZIP_CODE'].isin(high_value_zips).astype(int)

    def create_fairness_metrics(self, y_true, y_pred, sensitive_features):
        """Create fairlearn MetricFrame and calculate disparities"""
        # Create base metrics
        metric_frame = MetricFrame(
            metrics=self.metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        
        # Calculate disparities manually
        disparities = {}
        for metric_name in self.metrics.keys():
            metric_by_group = metric_frame.by_group[metric_name]
            if len(metric_by_group) == 2:  # Binary sensitive feature
                # Get values for privileged (1) and unprivileged (0) groups
                priv_value = metric_by_group.get(1, None)
                unpriv_value = metric_by_group.get(0, None)
                
                if priv_value is not None and unpriv_value is not None:
                    # Calculate difference
                    disparities[f'{metric_name}_difference'] = priv_value - unpriv_value
                    # Calculate ratio (if applicable)
                    if unpriv_value != 0:
                        disparities[f'{metric_name}_ratio'] = priv_value / unpriv_value
        
        return metric_frame, disparities

    def analyze_bias(self, X, y):
        """Analyze bias using fairlearn metrics"""
        X_copy = X.copy()
        predictions = self.model.predict(X)
        bias_results = {}
        
        # Determine ZIP_CODE groups if needed
        if 'ZIP_CODE' in X.columns:
            self.determine_zip_code_groups(X_copy, y)
            
        # Analyze each sensitive feature
        for feature, info in self.sensitive_features.items():
            if feature not in X_copy.columns and feature != 'ZIP_CODE':
                continue
            
            # Use HIGH_VALUE_ZIP for ZIP_CODE analysis
            analysis_feature = 'HIGH_VALUE_ZIP' if feature == 'ZIP_CODE' else feature
            if analysis_feature not in X_copy.columns:
                continue
                
            # Create MetricFrame and calculate disparities
            metric_frame, disparities = self.create_fairness_metrics(
                y_true=y,
                y_pred=predictions,
                sensitive_features=X_copy[analysis_feature]
            )
            
            # Calculate group metrics
            group_metrics = {
                'overall': metric_frame.overall,
                'by_group': metric_frame.by_group,
                'disparities': {
                    name: float(value)
                    for name, value in disparities.items()
                    if not np.isnan(value) and not np.isinf(value)
                }
            }
            
            bias_results[feature] = {
                'group_metrics': group_metrics,
                'feature_info': info
            }
            
            # Generate visualization
            self._plot_group_metrics(metric_frame, disparities, feature)

        return bias_results

    def mitigate_bias(self, X, y, sensitive_feature):
        """
        Enhanced bias mitigation using multiple approaches:
        1. Progressive constraint relaxation
        2. Dynamic sample reweighting
        3. Post-processing calibration
        """
        if sensitive_feature not in X.columns and sensitive_feature != 'ZIP_CODE':
            return self.model
            
        # Use HIGH_VALUE_ZIP for ZIP_CODE analysis
        analysis_feature = 'HIGH_VALUE_ZIP' if sensitive_feature == 'ZIP_CODE' else sensitive_feature
        if analysis_feature not in X.columns:
            return self.model
            
        try:
            X_copy = X.copy()
            best_model = None
            best_disparity = float('inf')
            best_metric = None
            
            # Calculate group statistics
            group_sizes = X_copy[analysis_feature].value_counts()
            total_samples = len(X_copy)
            
            # Calculate initial predictions and group means
            initial_preds = self.model.predict(X_copy)
            group_0_mask = X_copy[analysis_feature] == 0
            group_1_mask = X_copy[analysis_feature] == 1
            group_0_mean = np.mean(initial_preds[group_0_mask])
            group_1_mean = np.mean(initial_preds[group_1_mask])
            prediction_gap = abs(group_1_mean - group_0_mean)
            
            # Calculate error rates for each group
            group_0_rmse = np.sqrt(skm.mean_squared_error(y[group_0_mask], initial_preds[group_0_mask]))
            group_1_rmse = np.sqrt(skm.mean_squared_error(y[group_1_mask], initial_preds[group_1_mask]))
            rmse_ratio = group_1_rmse / group_0_rmse
            
            logger.info(f"Initial RMSE ratio (privileged/unprivileged): {rmse_ratio:.3f}")
            logger.info(f"Initial prediction gap: ${prediction_gap:,.2f}")
            
            # Dynamic weight calculation based on both prediction and error gaps
            if rmse_ratio < 0.8:  # Higher errors in privileged group
                base_weight_ratio = max(1.0, (1/rmse_ratio) * prediction_gap / (group_0_mean + group_1_mean))
                weights = np.ones(total_samples)
                weights[group_1_mask] = base_weight_ratio  # Increase weight for privileged group
            else:
                base_weight_ratio = max(1.0, rmse_ratio * prediction_gap / (group_0_mean + group_1_mean))
                weights = np.ones(total_samples)
                weights[group_0_mask] = base_weight_ratio  # Increase weight for unprivileged group
            
            # Progressive constraint weights - use tighter bounds when RMSE ratio is far from 1
            if abs(1 - rmse_ratio) > 0.3:
                constraint_weights = np.logspace(-3, -1, 15)  # More weights, tighter range
            else:
                constraint_weights = np.logspace(-4, -1, 10)  # Standard range
            
            for iteration in range(2):  # Two passes with different strategies
                for weight in constraint_weights:
                    try:
                        # Adjust weights based on current disparities
                        if iteration == 0:
                            # First pass: Focus on balancing predictions
                            group_weight = base_weight_ratio if group_1_mean > group_0_mean else 1/base_weight_ratio
                            weights[group_1_mask] = group_weight
                        else:
                            # Second pass: Focus on error rates
                            weights = np.ones(total_samples)
                            for group in [0, 1]:
                                group_mask = X_copy[analysis_feature] == group
                                if group in group_sizes:
                                    weights[group_mask] = total_samples / (2 * group_sizes[group])
                        
                        # Try different loss functions
                        for loss_type in ["squared_loss", "absolute_loss"]:
                            constraint = BoundedGroupLoss(
                                loss=loss_type,
                                upper_bound=weight
                            )
                            
                            mitigator = ExponentiatedGradient(
                                estimator=clone(self.model),
                                constraints=constraint,
                                eps=0.001  # Smaller epsilon for more precise optimization
                            )
                            
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                mitigator.fit(
                                    X_copy,
                                    y,
                                    sensitive_features=X_copy[analysis_feature],
                                    sample_weight=weights
                                )
                            
                            # Get predictions and calculate disparities
                            predictions = mitigator.predict(X_copy)
                            metric_frame, disparities = self.create_fairness_metrics(
                                y_true=y,
                                y_pred=predictions,
                                sensitive_features=X_copy[analysis_feature]
                            )
                            
                            # Enhanced disparity scoring that considers direction of RMSE ratio
                            rmse_ratio = disparities.get('rmse_ratio', float('inf'))
                            rmse_disparity = abs(rmse_ratio - 1.0) if rmse_ratio >= 1.0 else abs(1/rmse_ratio - 1.0)
                            pred_diff = abs(disparities.get('mean_prediction_difference', float('inf')))
                            pred_disparity = pred_diff / np.mean(y)  # Normalize by mean target
                            r2_disparity = abs(disparities.get('r2_difference', float('inf')))
                            
                            # Adaptive scoring based on initial RMSE ratio
                            if rmse_ratio < 0.8:
                                # When privileged group has higher errors, focus more on RMSE
                                current_disparity = (
                                    0.5 * rmse_disparity +    # Increased weight on RMSE
                                    0.3 * pred_disparity +    # Reduced weight on predictions
                                    0.2 * r2_disparity        # Maintained weight on R²
                                )
                            else:
                                # When unprivileged group has higher errors or ratios are close
                                current_disparity = (
                                    0.5 * pred_disparity +    # Focus on prediction fairness
                                    0.3 * rmse_disparity +    # Moderate weight on RMSE
                                    0.2 * r2_disparity        # Maintained weight on R²
                                )
                            
                            if current_disparity < best_disparity:
                                best_disparity = current_disparity
                                best_model = mitigator.predictor_
                                best_metric = {
                                    'iteration': iteration,
                                    'weight': weight,
                                    'loss': loss_type,
                                    'rmse_disparity': rmse_disparity,
                                    'pred_disparity': pred_disparity,
                                    'r2_disparity': r2_disparity,
                                    'pred_diff': pred_diff
                                }
                                logger.info(
                                    f"Found better model (iteration {iteration}):\n"
                                    f"- Loss type: {loss_type}\n"
                                    f"- Weight: {weight:.4f}\n"
                                    f"- Prediction difference: ${pred_diff:,.2f}\n"
                                    f"- RMSE disparity: {rmse_disparity:.3f}\n"
                                    f"- Overall disparity: {current_disparity:.3f}"
                                )
                    
                    except Exception as inner_e:
                        logger.warning(f"Failed with weight {weight}: {str(inner_e)}")
                        continue
            
            if best_model is not None:
                logger.info(
                    f"Selected best model:\n"
                    f"- Iteration: {best_metric['iteration']}\n"
                    f"- Configuration: weight={best_metric['weight']:.4f}, loss={best_metric['loss']}\n"
                    f"- Prediction difference: ${best_metric['pred_diff']:,.2f}\n"
                    f"- RMSE disparity: {best_metric['rmse_disparity']:.3f}\n"
                    f"- R² disparity: {best_metric['r2_disparity']:.3f}"
                )
                
                # Apply post-processing calibration
                calibrated_model = self._calibrate_predictions(
                    best_model, X, y, analysis_feature
                )
                return calibrated_model
            else:
                logger.error("All mitigation attempts failed")
                return self.model
                
        except Exception as e:
            logger.error(f"Fairlearn bias mitigation failed: {str(e)}")
            return self.model

    def _calibrate_predictions(self, model, X, y, sensitive_feature):
        """
        Post-processing calibration to further reduce prediction disparities
        """
        try:
            # Create a wrapper model for calibration
            class CalibratedModel:
                def __init__(self, base_model, calibration_factors):
                    self.base_model = base_model
                    self.calibration_factors = calibration_factors
                
                def predict(self, X):
                    base_preds = self.base_model.predict(X)
                    calibrated_preds = base_preds.copy()
                    
                    # Apply group-specific calibration
                    for group, factor in self.calibration_factors.items():
                        mask = X[sensitive_feature] == group
                        calibrated_preds[mask] *= factor
                    
                    return calibrated_preds
            
            # Calculate initial predictions
            initial_preds = model.predict(X)
            group_0_mask = X[sensitive_feature] == 0
            group_1_mask = X[sensitive_feature] == 1
            
            # Calculate group-specific calibration factors
            calibration_factors = {}
            group_0_true_mean = np.mean(y[group_0_mask])
            group_1_true_mean = np.mean(y[group_1_mask])
            group_0_pred_mean = np.mean(initial_preds[group_0_mask])
            group_1_pred_mean = np.mean(initial_preds[group_1_mask])
            
            # Calculate relative calibration to maintain overall scale
            if group_0_pred_mean > 0 and group_1_pred_mean > 0:
                mean_ratio = (group_1_true_mean / group_0_true_mean) / (group_1_pred_mean / group_0_pred_mean)
                if mean_ratio > 1:
                    calibration_factors[1] = mean_ratio
                    calibration_factors[0] = 1.0
                else:
                    calibration_factors[0] = 1/mean_ratio
                    calibration_factors[1] = 1.0
            
            if calibration_factors:
                # Create calibrated model
                calibrated_model = CalibratedModel(model, calibration_factors)
                
                # Verify calibration improved predictions
                cal_preds = calibrated_model.predict(X)
                _, initial_disparities = self.create_fairness_metrics(
                    y, initial_preds, X[sensitive_feature]
                )
                _, calibrated_disparities = self.create_fairness_metrics(
                    y, cal_preds, X[sensitive_feature]
                )
                
                initial_gap = abs(initial_disparities.get('mean_prediction_difference', float('inf')))
                calibrated_gap = abs(calibrated_disparities.get('mean_prediction_difference', float('inf')))
                
                if calibrated_gap < initial_gap:
                    logger.info(
                        f"Post-processing calibration reduced prediction gap "
                        f"from ${initial_gap:,.2f} to ${calibrated_gap:,.2f}"
                    )
                    return calibrated_model
                else:
                    logger.warning("Calibration did not improve prediction disparities")
            
            return model
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            return model

    def _plot_group_metrics(self, metric_frame, disparities, feature):
        """Plot fairness metrics and disparities"""
        metrics_to_plot = ['rmse', 'r2', 'mean_prediction']
        group_metrics = metric_frame.by_group
        
        # Plot metrics by group
        plt.figure(figsize=(15, 5))
        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(1, 3, i)
            
            values = group_metrics[metric]
            colors = ['#e74c3c', '#2ecc71']  # Red for unprivileged, green for privileged
            
            plt.bar([0, 1], values, color=colors)
            plt.title(f'{metric.upper()} by Group - {feature}')
            plt.xticks([0, 1], ['Unprivileged', 'Privileged'])
            plt.ylabel(metric.upper())
        
        plt.tight_layout()
        plot_path = self.results_dir / f'fairness_metrics_{feature}.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        # Plot disparities
        if disparities:
            plt.figure(figsize=(10, 5))
            disparity_values = list(disparities.values())
            disparity_names = list(disparities.keys())
            
            plt.bar(range(len(disparity_values)), disparity_values)
            plt.title(f'Fairness Disparities - {feature}')
            plt.xticks(range(len(disparity_names)), disparity_names, rotation=45)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            plt.tight_layout()
            
            disparity_path = self.results_dir / f'fairness_disparities_{feature}.png'
            plt.savefig(disparity_path, bbox_inches='tight')
            plt.close()

    def generate_bias_report(self, bias_results):
        """Generate comprehensive fairness report using MetricFrame results"""
        significant_disparities = []
        bias_detected = False
        
        # Calculate base price from all available predictions
        all_predictions = []
        for result in bias_results.values():
            metrics = result['group_metrics']
            for group in [0, 1]:
                if group in metrics['by_group'].index:
                    all_predictions.append(metrics['by_group']['mean_prediction'][group])
        
        base_price = np.median(all_predictions) if all_predictions else 500000
        logger.info(f"Using base price of ${base_price:,.2f} for disparity thresholds")
        
        # Set adaptive thresholds based on initial disparities
        initial_rmse_ratio = next(
            (result['group_metrics']['disparities'].get('rmse_ratio', 1.0) 
                for result in bias_results.values()),
            1.0
        )
        
        # Adjust thresholds based on initial RMSE ratio
        if initial_rmse_ratio < 0.8:
            # When privileged group has higher errors, use tighter bounds
            THRESHOLDS = {
                'rmse_ratio_upper': 1.1,     # Tighter upper bound
                'rmse_ratio_lower': 0.9,     # Tighter lower bound
                'r2_difference': 0.05,       # Smaller R² difference allowed
                'mean_prediction_difference': base_price * 0.10  # 10% of base price
            }
        else:
            # Standard thresholds
            THRESHOLDS = {
                'rmse_ratio_upper': 1.2,     # Upper bound for RMSE ratio
                'rmse_ratio_lower': 0.8,     # Lower bound for RMSE ratio
                'r2_difference': 0.1,        # Absolute difference in R²
                'mean_prediction_difference': base_price * 0.15  # 15% of base price
            }
        
        logger.info(f"Using {'tighter' if initial_rmse_ratio < 0.8 else 'standard'} thresholds "
                    f"based on initial RMSE ratio of {initial_rmse_ratio:.3f}")
        
        for feature, results in bias_results.items():
            metrics = results['group_metrics']
            
            # Check for significant disparities
            disparities = {
                'rmse_ratio': metrics['disparities'].get('rmse_ratio', 1.0),
                'r2_difference': abs(metrics['disparities'].get('r2_difference', 0.0)),
                'mean_prediction_difference': abs(metrics['disparities'].get('mean_prediction_difference', 0.0))
            }
            
            # Enhanced disparity checks
            rmse_ratio = disparities['rmse_ratio']
            is_significant = (
                # Check if RMSE ratio is outside acceptable range
                (rmse_ratio > THRESHOLDS['rmse_ratio_upper'] or 
                 rmse_ratio < THRESHOLDS['rmse_ratio_lower']) or
                # Check R² difference
                abs(disparities['r2_difference']) > THRESHOLDS['r2_difference'] or
                # Check mean prediction difference relative to base price
                abs(disparities['mean_prediction_difference']) > THRESHOLDS['mean_prediction_difference']
            ) and not (np.isnan(rmse_ratio) or np.isinf(rmse_ratio))
            
            if is_significant:
                significant_disparities.append({
                    'feature': feature,
                    'disparities': disparities,
                    'group_metrics': {
                        'by_group': metrics['by_group'].to_dict(),
                        'overall': metrics['overall']
                    },
                    'interpretation': self._interpret_fairlearn_disparities(feature, disparities)
                })
                bias_detected = True
        
        # Create comprehensive report
        report_data = {
            'bias_detected': bias_detected,
            'details': bias_results,
            'significant_disparities': significant_disparities,
            'summary': {
                'status': 'Bias Detected' if bias_detected else 'No Significant Bias',
                'features_analyzed': len(bias_results),
                'features_with_bias': len(significant_disparities),
                'thresholds_used': THRESHOLDS,
                'base_price': base_price
            },
            'methodology': {
                'fairness_metrics_used': list(self.metrics.keys()),
                'sensitive_features': {
                    name: info['description']
                    for name, info in self.sensitive_features.items()
                }
            }
        }
        
        # Save detailed report
        report_path = self.results_dir / 'fairness_report.json'
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
            
        return report_data

    def _interpret_fairlearn_disparities(self, feature, disparities):
        """Interpret fairness metrics from MetricFrame analysis"""
        interpretations = []
        
        rmse_ratio = disparities['rmse_ratio']
        if rmse_ratio > 1.2:
            interpretations.append(
                f"Model predictions show {((rmse_ratio - 1) * 100):.1f}% "
                f"higher error rates for disadvantaged groups in {feature}"
            )
        elif rmse_ratio < 0.8:
            interpretations.append(
                f"Model predictions show {((1 - rmse_ratio) * 100):.1f}% "
                f"higher error rates for privileged groups in {feature}"
            )
        
        r2_diff = disparities['r2_difference']
        if abs(r2_diff) > 0.1:
            better_group = "privileged" if r2_diff > 0 else "unprivileged"
            interpretations.append(
                f"Model performance (R²) is {abs(r2_diff * 100):.1f}% better "
                f"for {better_group} groups in {feature}"
            )
        
        pred_diff = disparities['mean_prediction_difference']
        if abs(pred_diff) > 0:  # Always report prediction differences
            higher_group = "privileged" if pred_diff > 0 else "unprivileged"
            interpretations.append(
                f"Average predictions are ${abs(pred_diff):,.2f} higher "
                f"for {higher_group} groups in {feature}"
            )
        
        return interpretations
