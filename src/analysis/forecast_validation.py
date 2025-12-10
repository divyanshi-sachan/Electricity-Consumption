"""
Forecast validation and quality assessment functions.

This module provides comprehensive forecast validation including RMSE, MAE, MAPE, R²,
training vs validation curves, residual plots, and feature importance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    mean_absolute_percentage_error, r2_score
)


def calculate_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Calculate comprehensive forecast validation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
    
    Returns:
        Dictionary with all validation metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Calculate additional metrics
    mean_actual = np.mean(y_true)
    nrmse = (rmse / mean_actual) * 100 if mean_actual != 0 else 0
    
    metrics = {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'nrmse': nrmse,
        'mean_actual': mean_actual,
        'mean_predicted': np.mean(y_pred),
        'std_actual': np.std(y_true),
        'std_predicted': np.std(y_pred)
    }
    
    return metrics


def plot_training_validation_curve(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    epochs_or_steps: np.ndarray,
    metric_name: str = "RMSE",
    ax=None
):
    """
    Plot training vs validation curve.
    
    Args:
        train_scores: Training scores over time
        val_scores: Validation scores over time
        epochs_or_steps: Epoch/step numbers
        metric_name: Name of the metric being plotted
        ax: Optional matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs_or_steps, train_scores, label=f'Training {metric_name}', 
            color='steelblue', linewidth=2)
    ax.plot(epochs_or_steps, val_scores, label=f'Validation {metric_name}', 
            color='coral', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch/Step', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'Training vs Validation {metric_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return ax


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax=None
):
    """
    Plot residual analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        ax: Optional matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = y_true - y_pred
    
    ax.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax.text(0.05, 0.95, f'Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return ax


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_n: int = 10,
    ax=None
):
    """
    Plot feature importance (for tree-based models).
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to show
        ax: Optional matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    return ax


def generate_validation_report(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    model_name: str = "Model",
    feature_names: Optional[list] = None,
    feature_importances: Optional[np.ndarray] = None
) -> Tuple[Dict, plt.Figure]:
    """
    Generate comprehensive validation report with all plots.
    
    Args:
        y_train: Training true values
        y_train_pred: Training predictions
        y_test: Test true values
        y_test_pred: Test predictions
        model_name: Name of the model
        feature_names: Optional list of feature names
        feature_importances: Optional feature importance values
    
    Returns:
        Tuple of (metrics dictionary, matplotlib figure)
    """
    # Calculate metrics
    train_metrics = calculate_forecast_metrics(y_train, y_train_pred, f"{model_name} (Train)")
    test_metrics = calculate_forecast_metrics(y_test, y_test_pred, f"{model_name} (Test)")
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Actual vs Predicted (Train)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_train, y_train_pred, alpha=0.5, s=20)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual (Train)', fontsize=12)
    ax1.set_ylabel('Predicted (Train)', fontsize=12)
    ax1.set_title(f'{model_name} - Training Set', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Actual vs Predicted (Test)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_test, y_test_pred, alpha=0.5, s=20)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual (Test)', fontsize=12)
    ax2.set_ylabel('Predicted (Test)', fontsize=12)
    ax2.set_title(f'{model_name} - Test Set', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual Plot (Test)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_residuals(y_test, y_test_pred, ax3)
    
    # 4. Residual Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    residuals = y_test - y_test_pred
    ax4.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residuals', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Feature Importance (if provided)
    if feature_names is not None and feature_importances is not None:
        ax5 = fig.add_subplot(gs[2, :])
        plot_feature_importance(feature_names, feature_importances, ax=ax5)
    else:
        # Metrics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        metrics_text = f"""
        Training Metrics:
        - RMSE: {train_metrics['rmse']:.2f}
        - MAE: {train_metrics['mae']:.2f}
        - MAPE: {train_metrics['mape']:.2f}%
        - R²: {train_metrics['r2']:.4f}
        
        Test Metrics:
        - RMSE: {test_metrics['rmse']:.2f}
        - MAE: {test_metrics['mae']:.2f}
        - MAPE: {test_metrics['mape']:.2f}%
        - R²: {test_metrics['r2']:.4f}
        """
        ax5.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Forecast Validation Report: {model_name}', fontsize=16, fontweight='bold', y=0.995)
    
    return {'train': train_metrics, 'test': test_metrics}, fig

