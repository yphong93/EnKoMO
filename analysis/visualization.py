import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def plot_predictions(predictions, ground_truths, save_dir, system_name, index=0, normalize_stats=None):
    """
    Plot prediction vs ground truth for a single sample
    
    Args:
        predictions: [N, T, D] predictions
        ground_truths: [N, T, D] ground truth
        save_dir: Directory to save plots
        system_name: Name of the system
        index: Sample index to plot
        normalize_stats: (mean, std) tuple for denormalization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    pred = predictions[index].copy()
    gt = ground_truths[index].copy()
    
    # Denormalize if stats provided
    if normalize_stats is not None:
        mean, std = normalize_stats
        # Check if mean and std are not None (e.g., QuantileTransformer returns None)
        if mean is not None and std is not None:
            mean = np.array(mean)
            std = np.array(std)
            # Ensure shapes are compatible
            if mean.ndim == 1 and pred.ndim == 2:
                mean = mean.reshape(1, -1)
                std = std.reshape(1, -1)
            pred = pred * std + mean
            gt = gt * std + mean
    
    dims = pred.shape[1]
    
    plt.figure(figsize=(12, 4 * dims))
    for i in range(dims):
        plt.subplot(dims, 1, i+1)
        plt.plot(gt[:, i], 'k-', label='Ground Truth', linewidth=2)
        plt.plot(pred[:, i], 'r--', label='Prediction', linewidth=2)
        plt.title(f"{system_name} - Dimension {i+1}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pred_vs_gt_sample_{index}.png'), dpi=150)
    plt.close()

def plot_phase_space(predictions, ground_truths, save_dir, system_name, index=0, normalize_stats=None):
    """
    Plot phase space (trajectory) plots
    
    Args:
        predictions: [N, T, D] predictions
        ground_truths: [N, T, D] ground truth
        save_dir: Directory to save plots
        system_name: Name of the system
        index: Sample index to plot
        normalize_stats: (mean, std) tuple for denormalization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    pred = predictions[index].copy()
    gt = ground_truths[index].copy()
    
    # Denormalize if stats provided
    if normalize_stats is not None:
        mean, std = normalize_stats
        # Check if mean and std are not None (e.g., QuantileTransformer returns None)
        if mean is not None and std is not None:
            mean = np.array(mean)
            std = np.array(std)
            # Ensure shapes are compatible
            if mean.ndim == 1 and pred.ndim == 2:
                mean = mean.reshape(1, -1)
                std = std.reshape(1, -1)
            pred = pred * std + mean
            gt = gt * std + mean
    
    dim = pred.shape[1]
    
    if dim >= 2:
        # 2D phase space
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(gt[:, 0], gt[:, 1], 'k-', label='Ground Truth', linewidth=2, alpha=0.7)
        plt.plot(gt[0, 0], gt[0, 1], 'go', markersize=10, label='Start')
        plt.plot(gt[-1, 0], gt[-1, 1], 'rs', markersize=10, label='End')
        plt.xlabel('Dimension 0')
        plt.ylabel('Dimension 1')
        plt.title(f"{system_name} - Ground Truth Phase Space")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(pred[:, 0], pred[:, 1], 'r--', label='Prediction', linewidth=2, alpha=0.7)
        plt.plot(pred[0, 0], pred[0, 1], 'go', markersize=10, label='Start')
        plt.plot(pred[-1, 0], pred[-1, 1], 'rs', markersize=10, label='End')
        plt.xlabel('Dimension 0')
        plt.ylabel('Dimension 1')
        plt.title(f"{system_name} - Predicted Phase Space")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'phase_space_2d_sample_{index}.png'), dpi=150)
        plt.close()
    
    if dim >= 3:
        # 3D phase space
        fig = plt.figure(figsize=(12, 5))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', linewidth=2, alpha=0.7)
        ax1.scatter(gt[0, 0], gt[0, 1], gt[0, 2], c='green', s=100, label='Start')
        ax1.scatter(gt[-1, 0], gt[-1, 1], gt[-1, 2], c='red', s=100, label='End')
        ax1.set_xlabel('Dimension 0')
        ax1.set_ylabel('Dimension 1')
        ax1.set_zlabel('Dimension 2')
        ax1.set_title(f"{system_name} - Ground Truth 3D Phase Space")
        ax1.legend()
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'r--', linewidth=2, alpha=0.7)
        ax2.scatter(pred[0, 0], pred[0, 1], pred[0, 2], c='green', s=100, label='Start')
        ax2.scatter(pred[-1, 0], pred[-1, 1], pred[-1, 2], c='red', s=100, label='End')
        ax2.set_xlabel('Dimension 0')
        ax2.set_ylabel('Dimension 1')
        ax2.set_zlabel('Dimension 2')
        ax2.set_title(f"{system_name} - Predicted 3D Phase Space")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'phase_space_3d_sample_{index}.png'), dpi=150)
        plt.close()

def plot_time_series(predictions, ground_truths, save_dir, system_name, normalize_stats=None):
    """
    Plot time series for all samples (averaged)
    
    Args:
        predictions: [N, T, D] predictions
        ground_truths: [N, T, D] ground truth
        save_dir: Directory to save plots
        system_name: Name of the system
        normalize_stats: (mean, std) tuple for denormalization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    pred = predictions.copy()
    gt = ground_truths.copy()
    
    # Denormalize if stats provided
    if normalize_stats is not None:
        mean, std = normalize_stats
        # Check if mean and std are not None (e.g., QuantileTransformer returns None)
        if mean is not None and std is not None:
            mean = np.array(mean)
            std = np.array(std)
            # Ensure shapes are compatible
            if mean.ndim == 1 and pred.ndim == 2:
                mean = mean.reshape(1, -1)
                std = std.reshape(1, -1)
            pred = pred * std + mean
            gt = gt * std + mean
    
    # Average over samples
    pred_mean = np.mean(pred, axis=0)
    gt_mean = np.mean(gt, axis=0)
    pred_std = np.std(pred, axis=0)
    gt_std = np.std(gt, axis=0)
    
    dims = pred.shape[2]
    T = pred.shape[1]
    
    plt.figure(figsize=(12, 4 * dims))
    for i in range(dims):
        plt.subplot(dims, 1, i+1)
        plt.plot(gt_mean[:, i], 'k-', label='Ground Truth (mean)', linewidth=2)
        plt.fill_between(range(T), gt_mean[:, i] - gt_std[:, i], gt_mean[:, i] + gt_std[:, i], 
                        alpha=0.3, color='black')
        plt.plot(pred_mean[:, i], 'r--', label='Prediction (mean)', linewidth=2)
        plt.fill_between(range(T), pred_mean[:, i] - pred_std[:, i], pred_mean[:, i] + pred_std[:, i], 
                        alpha=0.3, color='red')
        plt.title(f"{system_name} - Dimension {i+1} (Averaged over samples)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'time_series_averaged.png'), dpi=150)
    plt.close()

def plot_multi_model_phase_space(predictions_dict, ground_truths, save_dir, system_name, index=0, normalize_stats=None, input_sequence=None):
    """
    Plot phase space comparison for all models
    
    Args:
        predictions_dict: Dictionary {model_name: predictions} where predictions are [N, T, D]
        ground_truths: [N, T, D] ground truth
        save_dir: Directory to save plots
        system_name: Name of the system
        index: Sample index to plot
        normalize_stats: (mean, std) tuple for denormalization
        input_sequence: [N, T, D] input sequence used for prediction (optional)
                        If provided, uses the last state as common start point
    """
    os.makedirs(save_dir, exist_ok=True)
    
    gt = ground_truths[index].copy()
    
    # Denormalize ground truth if stats provided
    if normalize_stats is not None and normalize_stats[0] is not None:
        mean, std = normalize_stats
        mean = np.array(mean)
        std = np.array(std)
        gt = gt * std + mean
    
    # Determine common start point
    # Option 1: Use input sequence's last state if provided
    # Option 2: Use GT's first value (actual prediction start point)
    if input_sequence is not None:
        input_last = input_sequence[index].copy()
        if normalize_stats is not None and normalize_stats[0] is not None:
            input_last = input_last * std + mean
        common_start_point = input_last[-1, :]  # Last state of input sequence
    else:
        common_start_point = gt[0, :]  # First value of ground truth (prediction start)
    
    dim = gt.shape[1]
    
    if dim >= 2:
        # 2D phase space comparison
        plt.figure(figsize=(15, 10))
        
        # Ground Truth
        plt.subplot(2, 3, 1)
        plt.plot(gt[:, 0], gt[:, 1], 'k-', label='Ground Truth', linewidth=2, alpha=0.8)
        # Use common start point for consistency
        plt.plot(common_start_point[0], common_start_point[1], 'go', markersize=8, label='Start (Common)')
        plt.plot(gt[-1, 0], gt[-1, 1], 'rs', markersize=8, label='End')
        plt.xlabel('Dimension 0')
        plt.ylabel('Dimension 1')
        plt.title(f"{system_name} - Ground Truth")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Model predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        plot_idx = 2
        
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            if predictions is None:
                continue
                
            pred = predictions[index].copy()
            
            # Denormalize predictions if stats provided
            if normalize_stats is not None and normalize_stats[0] is not None:
                pred = pred * std + mean
            
            color = colors[i % len(colors)]
            
            if plot_idx <= 6:  # Show up to 5 models + ground truth
                plt.subplot(2, 3, plot_idx)
                
                # Plot ground truth lightly in background
                plt.plot(gt[:, 0], gt[:, 1], 'k-', alpha=0.3, linewidth=1, label='Ground Truth')
                
                # Plot prediction prominently
                # Use common start point for all models
                plt.plot([common_start_point[0], pred[0, 0]], 
                        [common_start_point[1], pred[0, 1]], 
                        color=color, linestyle=':', linewidth=1, alpha=0.5)  # Connection line
                plt.plot(pred[:, 0], pred[:, 1], color=color, linestyle='--', 
                        label=f'{model_name}', linewidth=2, alpha=0.8)
                plt.plot(common_start_point[0], common_start_point[1], 'go', markersize=8, label='Start (Common)')
                plt.plot(pred[-1, 0], pred[-1, 1], 'rs', markersize=8, label='End')
                
                plt.xlabel('Dimension 0')
                plt.ylabel('Dimension 1')
                plt.title(f"{system_name} - {model_name}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'phase_space_all_models_sample_{index}.png'), dpi=150)
        plt.close()
        
        # Combined comparison plot
        plt.figure(figsize=(12, 8))
        
        # Plot ground truth
        plt.plot(gt[:, 0], gt[:, 1], 'k-', label='Ground Truth', linewidth=3, alpha=0.8)
        # Use common start point
        plt.plot(common_start_point[0], common_start_point[1], 'go', markersize=10, label='Start (Common)', zorder=5)
        plt.plot(gt[-1, 0], gt[-1, 1], 'rs', markersize=10, label='End', zorder=5)
        
        # Plot all model predictions
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            if predictions is None:
                continue
                
            pred = predictions[index].copy()
            
            # Denormalize predictions if stats provided
            if normalize_stats is not None and normalize_stats[0] is not None:
                pred = pred * std + mean
            
            color = colors[i % len(colors)]
            # Draw connection line from common start point to first prediction
            plt.plot([common_start_point[0], pred[0, 0]], 
                    [common_start_point[1], pred[0, 1]], 
                    color=color, linestyle=':', linewidth=1, alpha=0.5)
            plt.plot(pred[:, 0], pred[:, 1], color=color, linestyle='--', 
                    label=f'{model_name}', linewidth=2, alpha=0.7)
        
        plt.xlabel('Dimension 0')
        plt.ylabel('Dimension 1')
        plt.title(f"{system_name} - All Models Phase Space Comparison")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'phase_space_combined_sample_{index}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    elif dim >= 3:
        # 3D phase space comparison
        fig = plt.figure(figsize=(20, 12))
        
        # Ground Truth 3D
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', linewidth=2, alpha=0.8)
        ax1.scatter(common_start_point[0], common_start_point[1], common_start_point[2], 
                   color='green', s=100, label='Start (Common)')
        ax1.scatter(gt[-1, 0], gt[-1, 1], gt[-1, 2], color='red', s=100, label='End')
        ax1.set_xlabel('Dimension 0')
        ax1.set_ylabel('Dimension 1')
        ax1.set_zlabel('Dimension 2')
        ax1.set_title(f"{system_name} - Ground Truth 3D")
        ax1.legend()
        
        # Model predictions 3D
        plot_idx = 2
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            if predictions is None or plot_idx > 6:
                continue
                
            pred = predictions[index].copy()
            
            # Denormalize predictions if stats provided
            if normalize_stats is not None and normalize_stats[0] is not None:
                pred = pred * std + mean
            
            color = colors[i % len(colors)]
            
            ax = fig.add_subplot(2, 3, plot_idx, projection='3d')
            
            # Plot ground truth lightly in background
            ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', alpha=0.3, linewidth=1)
            
            # Plot prediction prominently
            # Draw connection line from common start point
            ax.plot([common_start_point[0], pred[0, 0]], 
                   [common_start_point[1], pred[0, 1]], 
                   [common_start_point[2], pred[0, 2]], 
                   color=color, linestyle=':', linewidth=1, alpha=0.5)
            ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color=color, 
                   linestyle='--', linewidth=2, alpha=0.8)
            ax.scatter(common_start_point[0], common_start_point[1], common_start_point[2], 
                      color='green', s=100, label='Start (Common)')
            ax.scatter(pred[-1, 0], pred[-1, 1], pred[-1, 2], color='red', s=100, label='End')
            
            ax.set_xlabel('Dimension 0')
            ax.set_ylabel('Dimension 1')
            ax.set_zlabel('Dimension 2')
            ax.set_title(f"{system_name} - {model_name} 3D")
            ax.legend()
            
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'phase_space_3d_all_models_sample_{index}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Combined 3D comparison
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot ground truth
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', linewidth=3, alpha=0.8, label='Ground Truth')
        ax.scatter(common_start_point[0], common_start_point[1], common_start_point[2], 
                  color='green', s=100, label='Start (Common)')
        ax.scatter(gt[-1, 0], gt[-1, 1], gt[-1, 2], color='red', s=100, label='End')
        
        # Plot all model predictions
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            if predictions is None:
                continue
                
            pred = predictions[index].copy()
            
            # Denormalize predictions if stats provided
            if normalize_stats is not None and normalize_stats[0] is not None:
                pred = pred * std + mean
            
            color = colors[i % len(colors)]
            # Draw connection line from common start point
            ax.plot([common_start_point[0], pred[0, 0]], 
                   [common_start_point[1], pred[0, 1]], 
                   [common_start_point[2], pred[0, 2]], 
                   color=color, linestyle=':', linewidth=1, alpha=0.5)
            ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color=color, 
                   linestyle='--', linewidth=2, alpha=0.7, label=f'{model_name}')
        
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_zlabel('Dimension 2')
        ax.set_title(f"{system_name} - All Models 3D Phase Space Comparison")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'phase_space_3d_combined_sample_{index}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

def plot_multi_model_predictions(predictions_dict, ground_truths, save_dir, system_name, index=0, normalize_stats=None):
    """
    Plot prediction vs ground truth comparison for all models
    
    Args:
        predictions_dict: Dictionary {model_name: predictions} where predictions are [N, T, D]
        ground_truths: [N, T, D] ground truth
        save_dir: Directory to save plots
        system_name: Name of the system
        index: Sample index to plot
        normalize_stats: (mean, std) tuple for denormalization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    gt = ground_truths[index].copy()
    
    # Denormalize ground truth if stats provided
    if normalize_stats is not None and normalize_stats[0] is not None:
        mean, std = normalize_stats
        mean = np.array(mean)
        std = np.array(std)
        gt = gt * std + mean
    
    dims = gt.shape[1]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Individual model comparison plots (similar to original pred_vs_gt)
    plt.figure(figsize=(15, 4 * dims))
    
    for dim in range(dims):
        plt.subplot(dims, 1, dim + 1)
        
        # Plot ground truth
        plt.plot(gt[:, dim], 'k-', label='Ground Truth', linewidth=3, alpha=0.8)
        
        # Plot all model predictions
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            if predictions is None:
                continue
                
            pred = predictions[index].copy()
            
            # Denormalize predictions if stats provided
            if normalize_stats is not None and normalize_stats[0] is not None:
                pred = pred * std + mean
            
            color = colors[i % len(colors)]
            plt.plot(pred[:, dim], color=color, linestyle='--', 
                    label=f'{model_name}', linewidth=2, alpha=0.7)
        
        plt.title(f"{system_name} - Dimension {dim+1} Comparison")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pred_vs_gt_all_models_sample_{index}.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Separate plots for each model (grid layout)
    num_models = len([k for k, v in predictions_dict.items() if v is not None])
    if num_models > 0:
        cols = min(3, num_models)
        rows = (num_models + cols - 1) // cols
        
        for dim in range(dims):
            plt.figure(figsize=(5 * cols, 4 * rows))
            
            plot_idx = 1
            for i, (model_name, predictions) in enumerate(predictions_dict.items()):
                if predictions is None:
                    continue
                    
                pred = predictions[index].copy()
                
                # Denormalize predictions if stats provided
                if normalize_stats is not None and normalize_stats[0] is not None:
                    pred = pred * std + mean
                
                plt.subplot(rows, cols, plot_idx)
                
                plt.plot(gt[:, dim], 'k-', label='Ground Truth', linewidth=2, alpha=0.8)
                color = colors[i % len(colors)]
                plt.plot(pred[:, dim], color=color, linestyle='--', 
                        label=f'{model_name}', linewidth=2, alpha=0.8)
                
                plt.title(f"{model_name} - Dim {dim+1}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                
                plot_idx += 1
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'pred_vs_gt_grid_dim{dim+1}_sample_{index}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()

def plot_multi_model_time_series(predictions_dict, ground_truths, save_dir, system_name, normalize_stats=None):
    """
    Plot averaged time series comparison for all models
    
    Args:
        predictions_dict: Dictionary {model_name: predictions} where predictions are [N, T, D]
        ground_truths: [N, T, D] ground truth
        save_dir: Directory to save plots
        system_name: Name of the system
        normalize_stats: (mean, std) tuple for denormalization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Average over all samples
    gt_avg = np.mean(ground_truths, axis=0)  # [T, D]
    
    # Denormalize ground truth if stats provided
    if normalize_stats is not None and normalize_stats[0] is not None:
        mean, std = normalize_stats
        mean = np.array(mean)
        std = np.array(std)
        gt_avg = gt_avg * std + mean
    
    dims = gt_avg.shape[1]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Combined averaged time series plot
    plt.figure(figsize=(15, 4 * dims))
    
    for dim in range(dims):
        plt.subplot(dims, 1, dim + 1)
        
        # Plot ground truth average
        plt.plot(gt_avg[:, dim], 'k-', label='Ground Truth (Avg)', linewidth=3, alpha=0.8)
        
        # Plot all model predictions (averaged)
        for i, (model_name, predictions) in enumerate(predictions_dict.items()):
            if predictions is None:
                continue
                
            pred_avg = np.mean(predictions, axis=0)  # [T, D]
            
            # Denormalize predictions if stats provided
            if normalize_stats is not None and normalize_stats[0] is not None:
                pred_avg = pred_avg * std + mean
            
            color = colors[i % len(colors)]
            plt.plot(pred_avg[:, dim], color=color, linestyle='--', 
                    label=f'{model_name} (Avg)', linewidth=2, alpha=0.7)
        
        plt.title(f"{system_name} - Dimension {dim+1} Averaged Time Series")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'time_series_all_models_averaged.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Individual model time series (grid layout)
    num_models = len([k for k, v in predictions_dict.items() if v is not None])
    if num_models > 0:
        cols = min(3, num_models)
        rows = (num_models + cols - 1) // cols
        
        for dim in range(dims):
            plt.figure(figsize=(5 * cols, 4 * rows))
            
            plot_idx = 1
            for i, (model_name, predictions) in enumerate(predictions_dict.items()):
                if predictions is None:
                    continue
                    
                pred_avg = np.mean(predictions, axis=0)  # [T, D]
                
                # Denormalize predictions if stats provided
                if normalize_stats is not None and normalize_stats[0] is not None:
                    pred_avg = pred_avg * std + mean
                
                plt.subplot(rows, cols, plot_idx)
                
                plt.plot(gt_avg[:, dim], 'k-', label='Ground Truth', linewidth=2, alpha=0.8)
                color = colors[i % len(colors)]
                plt.plot(pred_avg[:, dim], color=color, linestyle='--', 
                        label=f'{model_name}', linewidth=2, alpha=0.8)
                
                plt.title(f"{model_name} - Dim {dim+1} Averaged")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                
                plot_idx += 1
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'time_series_grid_dim{dim+1}_averaged.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()

