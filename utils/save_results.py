"""
Utilities for saving experiment results in various formats
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Any

def save_metrics_to_csv(metrics_dict: Dict[str, Any], save_path: str, filename: str = 'metrics_summary.csv'):
    """
    Save metrics to CSV file
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        save_path: Directory to save CSV file
        filename: Name of CSV file
    """
    os.makedirs(save_path, exist_ok=True)
    
    rows = []
    for model_name, metrics in metrics_dict.items():
        if metrics is None:
            continue
        
        if 'overall' in metrics:
            overall = metrics['overall']
            row = {
                'Model': model_name,
                'MSE': overall.get('MSE', np.nan),
                'RMSE': overall.get('RMSE', np.nan),
                'MAE': overall.get('MAE', np.nan),
                'MAPE': overall.get('MAPE', np.nan),
                'R2': overall.get('R2', np.nan)
            }
            # Add Lyapunov time if available
            if 'lyapunov' in metrics:
                lyap = metrics['lyapunov']
                row['Lyapunov_Time_GT'] = lyap.get('ground_truth', np.nan)
                row['Lyapunov_Time_Pred'] = lyap.get('predictions', np.nan)
                row['Lyapunov_Exp_GT'] = lyap.get('lyap_exp_ground_truth', np.nan)
                row['Lyapunov_Exp_Pred'] = lyap.get('lyap_exp_predictions', np.nan)
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(save_path, filename)
        df.to_csv(csv_path, index=False, float_format='%.6f')
        return csv_path
    return None

def save_stepwise_metrics_to_csv(metrics_dict: Dict[str, Any], save_path: str, filename: str = 'stepwise_metrics.csv'):
    """
    Save stepwise metrics to CSV
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        save_path: Directory to save CSV file
        filename: Name of CSV file
    """
    os.makedirs(save_path, exist_ok=True)
    
    rows = []
    for model_name, metrics in metrics_dict.items():
        if metrics is None or 'stepwise' not in metrics:
            continue
        
        stepwise = metrics['stepwise']
        for step_name, step_metrics in stepwise.items():
            row = {
                'Model': model_name,
                'Step': step_name,
                'MSE': step_metrics.get('MSE', np.nan),
                'RMSE': step_metrics.get('RMSE', np.nan),
                'MAE': step_metrics.get('MAE', np.nan),
                'MAPE': step_metrics.get('MAPE', np.nan),
                'R2': step_metrics.get('R2', np.nan)
            }
            # Note: Lyapunov time is calculated for overall trajectory, not per step
            # So we don't add it to stepwise metrics
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(save_path, filename)
        df.to_csv(csv_path, index=False, float_format='%.6f')
        return csv_path
    return None

def save_predictions_to_csv(predictions: np.ndarray, ground_truths: np.ndarray, 
                           save_path: str, model_name: str, sample_idx: int = 0):
    """
    Save predictions and ground truth to CSV
    
    Args:
        predictions: [N, T, D] predictions
        ground_truths: [N, T, D] ground truth
        save_path: Directory to save CSV file
        model_name: Name of the model
        sample_idx: Sample index to save
    """
    os.makedirs(save_path, exist_ok=True)
    
    pred = predictions[sample_idx]  # [T, D]
    gt = ground_truths[sample_idx]   # [T, D]
    
    T, D = pred.shape
    
    # Create DataFrame
    data = {}
    for d in range(D):
        data[f'GT_Dim{d}'] = gt[:, d]
        data[f'Pred_Dim{d}'] = pred[:, d]
        data[f'Error_Dim{d}'] = gt[:, d] - pred[:, d]
        data[f'AbsError_Dim{d}'] = np.abs(gt[:, d] - pred[:, d])
    
    df = pd.DataFrame(data)
    df.insert(0, 'TimeStep', range(len(df)))
    
    csv_path = os.path.join(save_path, f'{model_name}_predictions_sample_{sample_idx}.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    return csv_path

def plot_metrics_comparison(metrics_dict: Dict[str, Any], save_path: str, 
                           system_name: str = 'Unknown'):
    """
    Create comparison plots for all metrics
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        save_path: Directory to save plots
        system_name: Name of the system
    """
    os.makedirs(save_path, exist_ok=True)
    
    models = []
    mse_values = []
    rmse_values = []
    mae_values = []
    mape_values = []
    r2_values = []
    
    for model_name, metrics in metrics_dict.items():
        if metrics is None or 'overall' not in metrics:
            continue
        
        overall = metrics['overall']
        models.append(model_name)
        mse_values.append(overall.get('MSE', np.nan))
        rmse_values.append(overall.get('RMSE', np.nan))
        mae_values.append(overall.get('MAE', np.nan))
        mape_values.append(overall.get('MAPE', np.nan))
        r2_values.append(overall.get('R2', np.nan))
    
    if not models:
        return
    
    # Create comparison bar plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Metrics Comparison - {system_name}', fontsize=16, fontweight='bold')
    
    metrics_data = [
        ('MSE', mse_values, 'lower'),
        ('RMSE', rmse_values, 'lower'),
        ('MAE', mae_values, 'lower'),
        ('MAPE (%)', mape_values, 'lower'),
        ('R² Score', r2_values, 'higher'),
    ]
    
    for idx, (metric_name, values, better) in enumerate(metrics_data):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        bars = ax.bar(models, values, alpha=0.7, edgecolor='black')
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name)
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight best value
        if better == 'lower':
            best_idx = np.nanargmin(values)
        else:
            best_idx = np.nanargmax(values)
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(0.9)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'metrics_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_stepwise_comparison(metrics_dict: Dict[str, Any], save_path: str, 
                            system_name: str = 'Unknown'):
    """
    Create stepwise comparison plots
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        save_path: Directory to save plots
        system_name: Name of the system
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Extract stepwise data
    step_data = {}
    for model_name, metrics in metrics_dict.items():
        if metrics is None or 'stepwise' not in metrics:
            continue
        
        stepwise = metrics['stepwise']
        for step_name, step_metrics in stepwise.items():
            if step_name not in step_data:
                step_data[step_name] = {}
            step_data[step_name][model_name] = step_metrics
    
    if not step_data:
        return
    
    # Create plots for each metric
    metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
    
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        step_names = sorted(step_data.keys())
        models = list(set([model for steps in step_data.values() for model in steps.keys()]))
        
        x = np.arange(len(step_names))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            values = [step_data[step].get(model, {}).get(metric, np.nan) 
                     for step in step_names]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model, alpha=0.7)
        
        ax.set_xlabel('Prediction Step', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison by Prediction Step - {system_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(step_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(save_path, f'stepwise_{metric.lower()}_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

def save_robustness_results(robustness_dict: Dict[str, Any], save_path: str, 
                            model_name: str):
    """
    Save robustness test results to CSV and plot
    
    Args:
        robustness_dict: Dictionary with noise levels as keys and metrics as values
        save_path: Directory to save files
        model_name: Name of the model
    """
    os.makedirs(save_path, exist_ok=True)
    
    if not robustness_dict:
        return
    
    # Save to CSV
    rows = []
    for noise_level, metrics in robustness_dict.items():
        if metrics is None:
            continue
        row = {
            'NoiseLevel': noise_level,
            'MSE': metrics.get('MSE', np.nan),
            'RMSE': metrics.get('RMSE', np.nan),
            'MAE': metrics.get('MAE', np.nan),
            'MAPE': metrics.get('MAPE', np.nan),
            'R2': metrics.get('R2', np.nan)
        }
        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(save_path, f'{model_name}_robustness.csv')
        df.to_csv(csv_path, index=False, float_format='%.6f')
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Robustness Test - {model_name}', fontsize=16, fontweight='bold')
        
        noise_levels = df['NoiseLevel'].values
        metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
        
        for idx, metric in enumerate(metrics_to_plot):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            ax.plot(noise_levels, df[metric].values, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Noise Level')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} vs Noise Level')
            ax.grid(True, alpha=0.3)
        
        axes[1, 2].axis('off')
        plt.tight_layout()
        plot_path = os.path.join(save_path, f'{model_name}_robustness.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return csv_path, plot_path
    
    return None, None

def save_multi_model_predictions_csv(predictions_dict, ground_truths, save_path, system_name, sample_idx=0, normalize_stats=None):
    """
    Save all model predictions and ground truth to a single CSV file
    
    Args:
        predictions_dict: Dictionary {model_name: predictions} where predictions are [N, T, D]
        ground_truths: [N, T, D] ground truth
        save_path: Directory to save CSV file
        system_name: Name of the system
        sample_idx: Which sample to save
        normalize_stats: (mean, std) tuple for denormalization
    
    Returns:
        str: Path to saved CSV file
    """
    os.makedirs(save_path, exist_ok=True)
    
    gt = ground_truths[sample_idx].copy()
    
    # Denormalize ground truth if stats provided
    if normalize_stats is not None and normalize_stats[0] is not None:
        mean, std = normalize_stats
        mean = np.array(mean)
        std = np.array(std)
        gt = gt * std + mean
    
    # Prepare data dictionary
    data = {}
    
    # Add time step
    data['Time_Step'] = range(len(gt))
    
    # Add dimensions for ground truth
    dims = gt.shape[1]
    for dim in range(dims):
        data[f'GT_Dim_{dim+1}'] = gt[:, dim]
    
    # Add predictions for each model
    for model_name, predictions in predictions_dict.items():
        if predictions is None:
            continue
            
        pred = predictions[sample_idx].copy()
        
        # Denormalize predictions if stats provided
        if normalize_stats is not None and normalize_stats[0] is not None:
            pred = pred * std + mean
        
        for dim in range(dims):
            data[f'{model_name}_Dim_{dim+1}'] = pred[:, dim]
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    csv_path = os.path.join(save_path, f'all_models_predictions_sample_{sample_idx}.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    
    return csv_path

def save_multi_model_averaged_csv(predictions_dict, ground_truths, save_path, system_name, normalize_stats=None):
    """
    Save averaged predictions for all models to CSV file
    
    Args:
        predictions_dict: Dictionary {model_name: predictions} where predictions are [N, T, D]
        ground_truths: [N, T, D] ground truth
        save_path: Directory to save CSV file
        system_name: Name of the system
        normalize_stats: (mean, std) tuple for denormalization
    
    Returns:
        str: Path to saved CSV file
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Average over all samples
    gt_avg = np.mean(ground_truths, axis=0)  # [T, D]
    
    # Denormalize ground truth if stats provided
    if normalize_stats is not None and normalize_stats[0] is not None:
        mean, std = normalize_stats
        mean = np.array(mean)
        std = np.array(std)
        gt_avg = gt_avg * std + mean
    
    # Prepare data dictionary
    data = {}
    
    # Add time step
    data['Time_Step'] = range(len(gt_avg))
    
    # Add dimensions for ground truth average
    dims = gt_avg.shape[1]
    for dim in range(dims):
        data[f'GT_Avg_Dim_{dim+1}'] = gt_avg[:, dim]
    
    # Add averaged predictions for each model
    for model_name, predictions in predictions_dict.items():
        if predictions is None:
            continue
            
        pred_avg = np.mean(predictions, axis=0)  # [T, D]
        
        # Denormalize predictions if stats provided
        if normalize_stats is not None and normalize_stats[0] is not None:
            pred_avg = pred_avg * std + mean
        
        for dim in range(dims):
            data[f'{model_name}_Avg_Dim_{dim+1}'] = pred_avg[:, dim]
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    csv_path = os.path.join(save_path, f'all_models_averaged_predictions.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    
    return csv_path

def save_model_comparison_summary_csv(predictions_dict, ground_truths, save_path, system_name, normalize_stats=None):
    """
    Save model comparison summary with statistical measures
    
    Args:
        predictions_dict: Dictionary {model_name: predictions}
        ground_truths: [N, T, D] ground truth
        save_path: Directory to save CSV file
        system_name: Name of the system
        normalize_stats: (mean, std) tuple for denormalization
    
    Returns:
        str: Path to saved CSV file
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Import MetricsCalculator here to avoid circular imports
    from utils.metrics import MetricsCalculator
    
    gt = ground_truths.copy()
    
    # Denormalize ground truth if stats provided
    if normalize_stats is not None and normalize_stats[0] is not None:
        mean, std = normalize_stats
        mean = np.array(mean)
        std = np.array(std)
        gt = gt * std + mean
    
    # Calculate comparison statistics
    comparison_data = []
    
    for model_name, predictions in predictions_dict.items():
        if predictions is None:
            continue
            
        pred = predictions.copy()
        
        # Denormalize predictions if stats provided
        if normalize_stats is not None and normalize_stats[0] is not None:
            pred = pred * std + mean
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_all_metrics(gt, pred)
        
        # Calculate per-dimension metrics
        dims = gt.shape[2]
        for dim in range(dims):
            gt_dim = gt[:, :, dim]
            pred_dim = pred[:, :, dim]
            
            dim_metrics = MetricsCalculator.calculate_all_metrics(gt_dim, pred_dim)
            
            comparison_data.append({
                'Model': model_name,
                'Dimension': f'Dim_{dim+1}',
                'MSE': dim_metrics['MSE'],
                'RMSE': dim_metrics['RMSE'],
                'MAE': dim_metrics['MAE'],
                'MAPE': dim_metrics['MAPE'],
                'R2': dim_metrics['R2'],
                'Mean_GT': np.mean(gt_dim),
                'Std_GT': np.std(gt_dim),
                'Mean_Pred': np.mean(pred_dim),
                'Std_Pred': np.std(pred_dim)
            })
        
        # Overall metrics
        comparison_data.append({
            'Model': model_name,
            'Dimension': 'Overall',
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'R2': metrics['R2'],
            'Mean_GT': np.mean(gt),
            'Std_GT': np.std(gt),
            'Mean_Pred': np.mean(pred),
            'Std_Pred': np.std(pred)
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(save_path, f'model_comparison_summary.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    
    return csv_path

def save_all_results(results: Dict[str, Any], save_path: str, system_name: str):
    """
    Save all results in multiple formats (CSV, images)
    
    Args:
        results: Dictionary with model names as keys and result dictionaries as values
        save_path: Directory to save all files
        system_name: Name of the system
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Extract metrics
    metrics_dict = {}
    for model_name, result in results.items():
        if result is not None and 'metrics' in result:
            metrics_dict[model_name] = result['metrics']
    
    # Save overall metrics to CSV
    csv_path = save_metrics_to_csv(metrics_dict, save_path, 'metrics_summary.csv')
    if csv_path:
        print(f"Metrics summary saved to: {csv_path}")
    
    # Save stepwise metrics to CSV
    stepwise_csv = save_stepwise_metrics_to_csv(metrics_dict, save_path, 'stepwise_metrics.csv')
    if stepwise_csv:
        print(f"Stepwise metrics saved to: {stepwise_csv}")
    
    # Create comparison plots
    plot_path = plot_metrics_comparison(metrics_dict, save_path, system_name)
    if plot_path:
        print(f"Metrics comparison plot saved to: {plot_path}")
    
    # Create stepwise comparison plots
    plot_stepwise_comparison(metrics_dict, save_path, system_name)
    print(f"Stepwise comparison plots saved to: {save_path}")
    
    # Save predictions for each model
    for model_name, result in results.items():
        if result is not None and 'predictions' in result and 'ground_truths' in result:
            model_save_dir = os.path.join(save_path, model_name)
            os.makedirs(model_save_dir, exist_ok=True)
            
            # Save predictions to CSV
            pred_csv = save_predictions_to_csv(
                result['predictions'],
                result['ground_truths'],
                model_save_dir,
                model_name,
                sample_idx=0
            )
            if pred_csv:
                print(f"Predictions saved to: {pred_csv}")
            
            # Save robustness results if available
            if 'robustness' in result:
                rob_csv, rob_plot = save_robustness_results(
                    result['robustness'],
                    model_save_dir,
                    model_name
                )
                if rob_csv:
                    print(f"Robustness results saved to: {rob_csv}")
                if rob_plot:
                    print(f"Robustness plot saved to: {rob_plot}")
    
    # Create multi-model comparison plots and CSV files
    print("Creating multi-model comparisons...")
    try:
        # Import here to avoid circular imports
        from analysis.visualization import (
            plot_multi_model_phase_space, 
            plot_multi_model_predictions, 
            plot_multi_model_time_series
        )
        
        # Collect predictions and ground truths from all models
        predictions_dict = {}
        ground_truths = None
        normalize_stats = None
        
        for model_name, result in results.items():
            if result is not None and 'predictions' in result and 'ground_truths' in result:
                predictions_dict[model_name] = result['predictions']
                if ground_truths is None:  # Use the first ground truth (should be same for all)
                    ground_truths = result['ground_truths']
                if normalize_stats is None and 'normalize_stats' in result:
                    normalize_stats = result['normalize_stats']
        
        if len(predictions_dict) > 1 and ground_truths is not None:
            print(f"Found {len(predictions_dict)} models for comparison: {list(predictions_dict.keys())}")
            
            # 1. Multi-model phase space plots
            plot_multi_model_phase_space(
                predictions_dict, 
                ground_truths, 
                save_path, 
                system_name, 
                index=0, 
                normalize_stats=normalize_stats
            )
            print(f"✅ Multi-model phase space plots saved to: {save_path}")
            
            # 2. Multi-model prediction comparison plots
            plot_multi_model_predictions(
                predictions_dict, 
                ground_truths, 
                save_path, 
                system_name, 
                index=0, 
                normalize_stats=normalize_stats
            )
            print(f"✅ Multi-model prediction comparison plots saved to: {save_path}")
            
            # 3. Multi-model time series comparison plots
            plot_multi_model_time_series(
                predictions_dict, 
                ground_truths, 
                save_path, 
                system_name, 
                normalize_stats=normalize_stats
            )
            print(f"✅ Multi-model time series plots saved to: {save_path}")
            
            # 4. Save multi-model CSV files
            print("Saving multi-model CSV files...")
            
            # Sample predictions CSV
            sample_csv = save_multi_model_predictions_csv(
                predictions_dict, 
                ground_truths, 
                save_path, 
                system_name, 
                sample_idx=0, 
                normalize_stats=normalize_stats
            )
            print(f"✅ Sample predictions CSV saved to: {sample_csv}")
            
            # Averaged predictions CSV
            avg_csv = save_multi_model_averaged_csv(
                predictions_dict, 
                ground_truths, 
                save_path, 
                system_name, 
                normalize_stats=normalize_stats
            )
            print(f"✅ Averaged predictions CSV saved to: {avg_csv}")
            
            # Model comparison summary CSV
            summary_csv = save_model_comparison_summary_csv(
                predictions_dict, 
                ground_truths, 
                save_path, 
                system_name, 
                normalize_stats=normalize_stats
            )
            print(f"✅ Model comparison summary CSV saved to: {summary_csv}")
            
        else:
            print("Not enough models or data for multi-model comparison")
            print(f"Available models: {len(predictions_dict)}, Ground truth available: {ground_truths is not None}")
            
    except Exception as e:
        print(f"❌ Error creating multi-model comparisons: {e}")
        import traceback
        traceback.print_exc()

