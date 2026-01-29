import torch
import torch.nn as nn
import numpy as np
from torch.autograd.functional import jacobian

class SpectralRegularization(nn.Module):
    """
    Spectral Regularization for EnKoMa based on the paper:
    - Spectral stability loss
    - Power iteration for dominant eigenvalues
    - Regularization to ensure stable dynamics
    """
    
    def __init__(self, lambda_spectral=0.1, lambda_stability=0.01, max_iterations=10):
        super().__init__()
        self.lambda_spectral = lambda_spectral
        self.lambda_stability = lambda_stability
        self.max_iterations = max_iterations
    
    def power_iteration(self, A, num_iterations=10):
        """
        Compute dominant eigenvalue using power iteration
        
        Args:
            A: [D, D] matrix
            num_iterations: number of iterations
        
        Returns:
            eigenvalue: dominant eigenvalue
            eigenvector: corresponding eigenvector
        """
        D = A.shape[0]
        v = torch.randn(D, device=A.device, dtype=A.dtype)
        v = v / torch.norm(v)
        
        for _ in range(num_iterations):
            v = torch.mv(A, v)
            eigenvalue = torch.norm(v)
            if eigenvalue > 1e-8:
                v = v / eigenvalue
            else:
                eigenvalue = torch.tensor(0.0, device=A.device)
                break
        
        return eigenvalue, v
    
    def compute_jacobian_batch(self, model, z_batch, evolution_map_fn):
        """
        Compute Jacobian for a batch of latent states
        
        Args:
            model: EnKoMa model
            z_batch: [B, D] batch of latent states
            evolution_map_fn: evolution map function
        
        Returns:
            jacobians: [B, D, D] batch of Jacobian matrices
        """
        jacobians = []
        
        for i in range(z_batch.shape[0]):
            z_i = z_batch[i].detach().requires_grad_(True)
            
            try:
                # Compute Jacobian for single sample
                J = jacobian(evolution_map_fn, z_i)
                if len(J.shape) == 1:
                    J = torch.diag(J)
                jacobians.append(J)
            except Exception as e:
                # Fallback: finite difference approximation
                D = z_i.shape[0]
                J = torch.zeros(D, D, device=z_i.device, dtype=z_i.dtype)
                eps = 1e-6
                
                z_i_detached = z_i.detach()
                f_z = evolution_map_fn(z_i_detached)
                
                for j in range(D):
                    z_pert = z_i_detached.clone()
                    z_pert[j] += eps
                    f_z_pert = evolution_map_fn(z_pert)
                    J[:, j] = (f_z_pert - f_z) / eps
                
                jacobians.append(J)
        
        return torch.stack(jacobians)
    
    def spectral_stability_loss(self, jacobians):
        """
        Compute spectral stability loss to keep eigenvalues within unit circle
        
        Args:
            jacobians: [B, D, D] batch of Jacobian matrices
        
        Returns:
            loss: spectral stability loss
        """
        batch_size = jacobians.shape[0]
        stability_losses = []
        
        for i in range(batch_size):
            J = jacobians[i]
            
            # Use power iteration to get dominant eigenvalue
            eigenvalue, _ = self.power_iteration(J, self.max_iterations)
            
            # Penalty if eigenvalue magnitude > 1 (unstable)
            eigenvalue_mag = torch.abs(eigenvalue)
            stability_loss = torch.relu(eigenvalue_mag - 1.0) ** 2
            stability_losses.append(stability_loss)
        
        return torch.stack(stability_losses).mean()
    
    def spectral_radius_regularization(self, jacobians):
        """
        Regularization to keep spectral radius controlled
        
        Args:
            jacobians: [B, D, D] batch of Jacobian matrices
        
        Returns:
            loss: spectral radius regularization loss
        """
        batch_size = jacobians.shape[0]
        spectral_losses = []
        
        for i in range(batch_size):
            J = jacobians[i]
            
            # Frobenius norm as proxy for spectral radius
            frobenius_norm = torch.norm(J, p='fro')
            
            # Also compute power iteration for more accurate estimate
            dominant_eigenval, _ = self.power_iteration(J, self.max_iterations)
            dominant_eigenval_mag = torch.abs(dominant_eigenval)
            
            # Combined regularization
            spectral_loss = 0.7 * dominant_eigenval_mag + 0.3 * frobenius_norm / J.shape[0]
            spectral_losses.append(spectral_loss)
        
        return torch.stack(spectral_losses).mean()
    
    def forward(self, model, z_batch):
        """
        Compute total spectral regularization loss
        
        Args:
            model: EnKoMa model
            z_batch: [B, seq_len, D] batch of latent states
        
        Returns:
            total_loss: combined spectral loss
            components: dictionary with loss components
        """
        # Take the last timestep for Jacobian computation
        if len(z_batch.shape) == 3:
            z = z_batch[:, -1, :]  # [B, D]
        else:
            z = z_batch  # [B, D]
        
        # Define evolution map function for Jacobian computation
        def evolution_map_fn(z_input):
            return model.evolution_map(z_input)
        
        # Compute Jacobians
        jacobians = self.compute_jacobian_batch(model, z, evolution_map_fn)
        
        # Compute individual loss components
        stability_loss = self.spectral_stability_loss(jacobians)
        radius_loss = self.spectral_radius_regularization(jacobians)
        
        # Combined loss
        total_loss = (self.lambda_stability * stability_loss + 
                     self.lambda_spectral * radius_loss)
        
        components = {
            'stability_loss': stability_loss.item(),
            'radius_loss': radius_loss.item(),
            'total_spectral_loss': total_loss.item()
        }
        
        return total_loss, components

def create_spectral_regularizer(config):
    """
    Factory function to create spectral regularizer based on config
    
    Args:
        config: experiment configuration
    
    Returns:
        SpectralRegularization instance
    """
    lambda_spectral = getattr(config, 'lambda_spectral', 0.1)
    lambda_stability = getattr(config, 'lambda_stability', 0.01)
    max_iterations = getattr(config, 'spectral_max_iter', 10)
    
    return SpectralRegularization(
        lambda_spectral=lambda_spectral,
        lambda_stability=lambda_stability,
        max_iterations=max_iterations
    )
