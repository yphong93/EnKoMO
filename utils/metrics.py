import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class MetricsCalculator:
    """
    Calculate evaluation metrics for time series prediction
    """
    
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(MetricsCalculator.mse(y_true, y_pred))
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mape(y_true, y_pred, epsilon=1e-8):
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = np.abs(y_true) > epsilon
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def r2_score(y_true, y_pred):
        """R-squared Score"""
        return r2_score(y_true.flatten(), y_pred.flatten())
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred):
        """
        Calculate all metrics
        
        Args:
            y_true: [N, T, D] or [T, D] ground truth
            y_pred: [N, T, D] or [T, D] predictions
        
        Returns:
            dict: Dictionary of metrics
        """
        # Flatten for overall metrics
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        metrics = {
            'MSE': MetricsCalculator.mse(y_true_flat, y_pred_flat),
            'RMSE': MetricsCalculator.rmse(y_true_flat, y_pred_flat),
            'MAE': MetricsCalculator.mae(y_true_flat, y_pred_flat),
            'MAPE': MetricsCalculator.mape(y_true_flat, y_pred_flat),
            'R2': MetricsCalculator.r2_score(y_true_flat, y_pred_flat)
        }
        
        return metrics
    
    @staticmethod
    def calculate_step_metrics(y_true, y_pred, short_steps=[50, 100, 200], long_steps=[500, 1000, 1500]):
        """
        Calculate metrics for specific prediction steps
        
        Args:
            y_true: [N, T, D] ground truth
            y_pred: [N, T, D] predictions
            short_steps: List of short-term step lengths
            long_steps: List of long-term step lengths
        
        Returns:
            dict: Dictionary of step-wise metrics
        """
        results = {}
        
        T = y_true.shape[1]
        
        # Short-term metrics
        for step in short_steps:
            if step <= T:
                y_true_step = y_true[:, :step, :]
                y_pred_step = y_pred[:, :step, :]
                metrics = MetricsCalculator.calculate_all_metrics(y_true_step, y_pred_step)
                results[f'short_{step}'] = metrics
        
        # Long-term metrics
        for step in long_steps:
            if step <= T:
                y_true_step = y_true[:, :step, :]
                y_pred_step = y_pred[:, :step, :]
                metrics = MetricsCalculator.calculate_all_metrics(y_true_step, y_pred_step)
                results[f'long_{step}'] = metrics
        
        return results

def calculate_metrics(y_true, y_pred, short_steps=[50, 100, 200], long_steps=[500, 1000, 1500]):
    """
    Convenience function to calculate all metrics
    """
    all_metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred)
    step_metrics = MetricsCalculator.calculate_step_metrics(y_true, y_pred, short_steps, long_steps)
    
    return {
        'overall': all_metrics,
        'stepwise': step_metrics
    }

