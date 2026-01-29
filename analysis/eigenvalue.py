import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.autograd.functional import jacobian

def compute_jacobian_evolution_map(model, z):
    """
    Compute Jacobian of evolution map F(z) = z + Mamba(z)
    
    Args:
        model: EnKoMa model with evolution_map method
        z: [D] or [B, D] latent state
    
    Returns:
        J: [D, D] Jacobian matrix
    """
    if len(z.shape) == 2:
        z = z[0]  # Take first sample
    
    z = z.detach().requires_grad_(True)
    
    def evolution_fn(z_input):
        return model.evolution_map(z_input)
    
    try:
        J = jacobian(evolution_fn, z)
        if len(J.shape) == 1:
            J = J.unsqueeze(0)
        return J
    except Exception as e:
        print(f"Jacobian computation failed: {e}")
        return None

def analyze_eigenvalues(model, test_data, device, save_dir=None):
    """
    Analyze eigenvalues of the learned Koopman operator
    
    Args:
        model: Trained model
        test_data: [N, T, D] test data
        device: torch device
        save_dir: Directory to save results
    
    Returns:
        eigenvalues: List of eigenvalue arrays
    """
    model.eval()
    eigenvalues_list = []
    
    with torch.no_grad():
        # Encode test data
        if hasattr(model, 'encoder'):
            z = model.encoder(test_data.to(device))
        else:
            z = test_data.to(device)
        
        # Adaptive sampling for better coverage
        batch_size, seq_len, latent_dim = z.shape
        max_samples = 100  # Maximum number of sample points
        
        # Calculate sampling strategy
        total_points = batch_size * seq_len
        if total_points <= max_samples:
            # Use all points if dataset is small
            sample_indices = [(b, t) for b in range(batch_size) for t in range(seq_len)]
        else:
            # Uniform sampling across batches and time
            step_size = max(1, total_points // max_samples)
            sample_indices = []
            for i in range(0, total_points, step_size):
                b = i // seq_len
                t = i % seq_len
                if b < batch_size and t < seq_len:
                    sample_indices.append((b, t))
        
        # Compute eigenvalues for sampled points
        for b, t in sample_indices[:max_samples]:
            z_t = z[b, t, :]
            
            # Compute Jacobian
            J = compute_jacobian_evolution_map(model, z_t)
            
            if J is not None:
                # Compute eigenvalues
                J_np = J.detach().cpu().numpy()
                eigenvals = np.linalg.eigvals(J_np)
                eigenvalues_list.append(eigenvals)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot eigenvalue distribution
        all_eigenvals = np.concatenate(eigenvalues_list)
        
        plt.figure(figsize=(15, 5))
        
        # Complex plane distribution
        plt.subplot(1, 3, 1)
        plt.scatter(np.real(all_eigenvals), np.imag(all_eigenvals), 
                   alpha=0.5, s=10, c='blue')
        # Plot unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'r--', linewidth=2, label='Unit Circle')
        plt.xlabel('Real')
        plt.ylabel('Imaginary') 
        plt.title('Eigenvalue Distribution\n(Complex Plane)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Magnitude distribution
        plt.subplot(1, 3, 2)
        magnitudes = np.abs(all_eigenvals)
        plt.hist(magnitudes, bins=50, alpha=0.7, color='blue')
        plt.axvline(1.0, color='r', linestyle='--', linewidth=2, label='Stability Threshold')
        plt.xlabel('|Eigenvalue|')
        plt.ylabel('Frequency')
        plt.title('Eigenvalue Magnitude\nDistribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Stability analysis
        plt.subplot(1, 3, 3)
        stable_count = np.sum(magnitudes < 1.0)
        unstable_count = np.sum(magnitudes > 1.0)
        neutral_count = np.sum(np.abs(magnitudes - 1.0) < 1e-3)
        
        categories = ['Stable\n(|λ| < 1)', 'Unstable\n(|λ| > 1)', 'Neutral\n(|λ| ≈ 1)']
        counts = [stable_count, unstable_count, neutral_count]
        colors = ['green', 'red', 'orange']
        
        bars = plt.bar(categories, counts, color=colors, alpha=0.7)
        plt.ylabel('Count')
        plt.title('Stability Classification')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        total = len(all_eigenvals)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                    f'{count}\n({100*count/total:.1f}%)',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'eigenvalue_analysis.png'), dpi=150)
        plt.close()
    
    return eigenvalues_list

def global_jacobian_stitching(model, test_data, device, save_dir=None, num_samples=50):
    """
    Global Jacobian Stitching: Reconstruct global eigenvalue spectrum
    by stitching together local linearizations (Jacobians) along the trajectory.
    
    Args:
        model: Trained model
        test_data: [N, T, D] test data
        device: torch device
        save_dir: Directory to save results
        num_samples: Total number of points to sample across all trajectories
    
    Returns:
        global_eigenvalues: All eigenvalues collected along trajectory
    """
    model.eval()
    global_eigenvalues = []
    
    with torch.no_grad():
        # Encode test data
        if hasattr(model, 'encoder'):
            z = model.encoder(test_data.to(device))
        else:
            z = test_data.to(device)
        
        # Adaptive sampling: distribute samples across batches and time
        batch_size, seq_len, latent_dim = z.shape
        total_points = batch_size * seq_len
        
        # Sample points uniformly across the entire dataset
        if total_points <= num_samples:
            # Use all points if dataset is small
            sample_indices = [(b, t) for b in range(batch_size) for t in range(seq_len)]
        else:
            # Uniform sampling across batches and time
            step_size = max(1, total_points // num_samples)
            sample_indices = []
            for i in range(0, total_points, step_size):
                b = i // seq_len
                t = i % seq_len
                if b < batch_size and t < seq_len:
                    sample_indices.append((b, t))
        
        # Batch process Jacobians for efficiency
        sampled_points = []
        for b, t in sample_indices[:num_samples]:
            sampled_points.append(z[b, t, :])
        
        if sampled_points:
            # Process in smaller batches to avoid memory issues
            batch_size_jacobian = min(10, len(sampled_points))
            for i in range(0, len(sampled_points), batch_size_jacobian):
                batch_points = sampled_points[i:i+batch_size_jacobian]
                
                for z_t in batch_points:
                    # Compute Jacobian
                    J = compute_jacobian_evolution_map(model, z_t)
                    
                    if J is not None:
                        # EnKoMa evolution_map already includes residual connection
                        # F(z) = z + Mamba(z), so J is already J_total
                        J_total = J
                        
                        # Eigendecomposition
                        J_np = J_total.detach().cpu().numpy()
                        eigenvals = np.linalg.eigvals(J_np)
                        global_eigenvalues.extend(eigenvals)
    
    global_eigenvalues = np.array(global_eigenvalues)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Enhanced visualization for global jacobian stitching
        plt.figure(figsize=(18, 6))
        
        # Complex plane distribution
        plt.subplot(1, 3, 1)
        plt.scatter(np.real(global_eigenvalues), np.imag(global_eigenvalues), 
                   alpha=0.3, s=5, c='blue')
        # Plot unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2, label='Unit Circle')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.title('Global Eigenvalue Spectrum\n(Complex Plane)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Magnitude distribution  
        plt.subplot(1, 3, 2)
        magnitudes = np.abs(global_eigenvalues)
        plt.hist(magnitudes, bins=100, alpha=0.7, color='blue')
        plt.axvline(1.0, color='r', linestyle='--', linewidth=2, label='Stability Threshold')
        plt.xlabel('|Eigenvalue|')
        plt.ylabel('Frequency')
        plt.title('Global Eigenvalue Magnitude\nDistribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Spectral radius analysis
        plt.subplot(1, 3, 3)
        max_magnitude = np.max(magnitudes)
        mean_magnitude = np.mean(magnitudes)
        
        # Create radial histogram
        radial_bins = np.linspace(0, max(2.0, max_magnitude*1.1), 50)
        hist, _ = np.histogram(magnitudes, bins=radial_bins)
        bin_centers = (radial_bins[:-1] + radial_bins[1:]) / 2
        
        plt.plot(bin_centers, hist, 'b-', linewidth=2, label='Magnitude Distribution')
        plt.axvline(1.0, color='r', linestyle='--', linewidth=2, 
                   label=f'Stability Threshold')
        plt.axvline(max_magnitude, color='orange', linestyle=':', linewidth=2,
                   label=f'Spectral Radius = {max_magnitude:.3f}')
        plt.axvline(mean_magnitude, color='green', linestyle='-.', linewidth=2,
                   label=f'Mean Magnitude = {mean_magnitude:.3f}')
        
        plt.xlabel('|Eigenvalue|')
        plt.ylabel('Count')
        plt.title('Spectral Analysis\n(Global Dynamics)')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'global_jacobian_stitching.png'), dpi=150)
        plt.close()
        
        # Compute mode frequencies from eigenvalues
        dt = getattr(model.config, 'dt', 0.01)
        
        # For complex eigenvalues λ = re^(iθ), frequency = θ/(2πdt)
        # Only consider eigenvalues with significant imaginary parts
        complex_eigenvals = global_eigenvalues[np.abs(np.imag(global_eigenvalues)) > 1e-6]
        
        if len(complex_eigenvals) > 0:
            # Compute frequencies from phase angles
            angles = np.angle(complex_eigenvals)
            frequencies = angles / (2 * np.pi * dt)
            
            # Remove negative frequencies (keep only positive spectrum)
            positive_frequencies = frequencies[frequencies > 0]
            
            # Apply Nyquist limit
            nyquist_freq = 0.5 / dt
            valid_frequencies = positive_frequencies[positive_frequencies < nyquist_freq]
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            if len(valid_frequencies) > 0:
                plt.hist(valid_frequencies, bins=50, alpha=0.7, color='blue')
                plt.axvline(nyquist_freq, color='r', linestyle='--', 
                           linewidth=2, label=f'Nyquist Limit ({nyquist_freq:.1f} Hz)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Count')
            plt.title('Mode Frequencies (Positive Spectrum)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            # Plot eigenvalue magnitudes vs frequencies for complex modes
            magnitudes = np.abs(complex_eigenvals)
            all_frequencies = np.angle(complex_eigenvals) / (2 * np.pi * dt)
            plt.scatter(all_frequencies, magnitudes, alpha=0.5, s=10)
            plt.axhline(1.0, color='r', linestyle='--', linewidth=2, 
                       label='Stability Threshold')
            plt.axvline(0, color='k', linestyle='-', alpha=0.3)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('|Eigenvalue|')
            plt.title('Eigenvalue Magnitude vs Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        else:
            # No complex eigenvalues found
            plt.figure(figsize=(10, 5))
            plt.text(0.5, 0.5, 'No complex eigenvalues found\n(System may be over-damped)', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title('Mode Frequencies Analysis')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mode_frequencies.png'), dpi=150)
        plt.close()
    
    return global_eigenvalues

