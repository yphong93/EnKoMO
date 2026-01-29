import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.metrics import MetricsCalculator

def robustness_test(model, test_data, test_labels, device, noise_levels=[0.0, 0.01, 0.05, 0.1, 0.2]):
    """
    Test model robustness to input noise
    
    Args:
        model: Trained model
        test_data: [N, T, D] test data
        test_labels: [N, T, D] ground truth
        device: torch device
        noise_levels: List of noise levels to test
    
    Returns:
        results: Dictionary of metrics for each noise level
    """
    model.eval()
    results = {}
    
    for noise_level in noise_levels:
        # Add noise to test data
        noise = torch.randn_like(test_data) * noise_level
        noisy_data = test_data + noise
        
        # Predict
        with torch.no_grad():
            outputs = model(noisy_data.to(device), pred_len=test_labels.shape[1])
            # All models return 5 values: (x_rec, x_dyn, x_pred, z, z_dyn)
            _, _, x_pred, _, _ = outputs
        
        if x_pred is not None:
            # Calculate metrics
            pred_np = x_pred.cpu().numpy()
            label_np = test_labels.numpy()
            
            metrics = MetricsCalculator.calculate_all_metrics(label_np, pred_np)
            results[noise_level] = metrics
        else:
            results[noise_level] = None
    
    return results

def perturbation_test(model, test_data, test_labels, device, perturbation_type='gaussian'):
    """
    Test model robustness to different types of perturbations
    
    Args:
        model: Trained model
        test_data: [N, T, D] test data
        test_labels: [N, T, D] ground truth
        device: torch device
        perturbation_type: 'gaussian', 'uniform', 'dropout'
    
    Returns:
        results: Dictionary of metrics for different perturbation levels
    """
    model.eval()
    results = {}
    
    perturbation_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    
    for level in perturbation_levels:
        # Apply perturbation
        if perturbation_type == 'gaussian':
            noise = torch.randn_like(test_data) * level
            perturbed_data = test_data + noise
        elif perturbation_type == 'uniform':
            noise = (torch.rand_like(test_data) - 0.5) * 2 * level
            perturbed_data = test_data + noise
        elif perturbation_type == 'dropout':
            mask = torch.rand_like(test_data) > level
            perturbed_data = test_data * mask
        else:
            perturbed_data = test_data
        
        # Predict
        with torch.no_grad():
            outputs = model(perturbed_data.to(device), pred_len=test_labels.shape[1])
            # All models return 5 values: (x_rec, x_dyn, x_pred, z, z_dyn)
            _, _, x_pred, _, _ = outputs
        
        if x_pred is not None:
            pred_np = x_pred.cpu().numpy()
            label_np = test_labels.numpy()
            
            metrics = MetricsCalculator.calculate_all_metrics(label_np, pred_np)
            results[f'{perturbation_type}_{level}'] = metrics
        else:
            results[f'{perturbation_type}_{level}'] = None
    
    return results

