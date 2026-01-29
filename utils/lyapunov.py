import numpy as np
import torch

def calculate_lyapunov_exponent(trajectory, dt=0.01, method='wolf'):
    """
    Calculate Lyapunov exponent from trajectory
    
    Args:
        trajectory: [T, D] array of states
        dt: Time step
        method: 'wolf' or 'rosenstein'
    
    Returns:
        lyap_exp: Largest Lyapunov exponent
    """
    if method == 'wolf':
        # Simplified Wolf algorithm
        # For accurate calculation, need to integrate along trajectory
        # Here we use a simplified approach based on divergence
        T, D = trajectory.shape
        
        # Calculate divergence rate
        divergences = []
        for i in range(1, T):
            dist = np.linalg.norm(trajectory[i] - trajectory[i-1])
            if dist > 0:
                divergences.append(np.log(dist + 1e-10))
        
        if len(divergences) > 0:
            lyap_exp = np.mean(divergences) / dt
        else:
            lyap_exp = 0.0
        
        return lyap_exp
    else:
        # Rosenstein method (simplified)
        # Find nearest neighbors and track divergence
        T, D = trajectory.shape
        min_neighbors = min(10, T // 10)
        
        divergences = []
        for i in range(T - min_neighbors):
            # Find nearest neighbor
            current = trajectory[i]
            distances = [np.linalg.norm(trajectory[j] - current) for j in range(i+1, min(i+min_neighbors, T))]
            if len(distances) > 0:
                min_dist = min(distances)
                if min_dist > 0:
                    divergences.append(np.log(min_dist + 1e-10))
        
        if len(divergences) > 0:
            lyap_exp = np.mean(divergences) / dt
        else:
            lyap_exp = 0.0
        
        return lyap_exp

def calculate_lyapunov_time(trajectory, dt=0.01, method='wolf'):
    """
    Calculate Lyapunov time (inverse of largest Lyapunov exponent)
    
    Args:
        trajectory: [T, D] array of states
        dt: Time step
        method: 'wolf' or 'rosenstein'
    
    Returns:
        lyap_time: Lyapunov time
    """
    lyap_exp = calculate_lyapunov_exponent(trajectory, dt, method)
    
    if lyap_exp > 0:
        lyap_time = 1.0 / lyap_exp
    else:
        lyap_time = np.inf
    
    return lyap_time

