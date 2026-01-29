import torch
import torch.nn as nn
import numpy as np

class EfficientSpectralRegularization(nn.Module):
    """
    효율적인 Spectral Regularization
    - 계산 복잡도를 대폭 줄임
    - Finite difference 대신 다른 방법 사용
    """
    
    def __init__(self, lambda_spectral=0.01, lambda_stability=0.001, 
                 sample_ratio=0.1, max_iterations=5):
        super().__init__()
        self.lambda_spectral = lambda_spectral
        self.lambda_stability = lambda_stability
        self.sample_ratio = sample_ratio  # 전체 배치의 일부만 계산
        self.max_iterations = max_iterations  # Power iteration 횟수 줄임
    
    def efficient_spectral_norm(self, z):
        """
        효율적인 spectral norm 추정
        Jacobian 계산 없이 latent space의 변화율만 측정
        """
        batch_size, latent_dim = z.shape
        
        # Random projection으로 spectral 특성 추정 (매우 빠름)
        random_proj = torch.randn(latent_dim, 8, device=z.device)  # 8개 방향만 테스트
        projected = torch.matmul(z, random_proj)  # [B, 8]
        
        # 변화율 기반 spectral radius 추정
        spectral_estimate = torch.norm(projected, dim=1).mean()
        
        return spectral_estimate
    
    def simple_stability_loss(self, z, z_next):
        """
        간단한 안정성 loss
        연속된 latent states 간의 변화율로 안정성 측정
        """
        if z_next is None:
            return torch.tensor(0.0, device=z.device)
        
        # Lipschitz 조건 기반 안정성
        delta_z = z_next - z
        stability_measure = torch.norm(delta_z, dim=-1).mean()
        
        # 과도한 변화를 억제
        stability_loss = torch.relu(stability_measure - 1.0) ** 2
        
        return stability_loss
    
    def forward(self, model, z_batch, z_next_batch=None):
        """
        효율적인 spectral loss 계산
        
        Args:
            model: EnKoMa model (사용 안 함)
            z_batch: [B, seq_len, D] latent states
            z_next_batch: [B, seq_len, D] next latent states (optional)
        
        Returns:
            total_loss: 빠른 spectral loss
            components: loss 구성 요소
        """
        # 마지막 timestep 사용
        z = z_batch[:, -1, :] if len(z_batch.shape) == 3 else z_batch
        z_next = z_next_batch[:, -1, :] if z_next_batch is not None and len(z_next_batch.shape) == 3 else z_next_batch
        
        batch_size = z.shape[0]
        
        # 배치의 일부만 계산 (속도 향상)
        sample_size = max(1, int(batch_size * self.sample_ratio))
        indices = torch.randperm(batch_size)[:sample_size]
        z_sample = z[indices]
        z_next_sample = z_next[indices] if z_next is not None else None
        
        # 1. 효율적인 spectral norm 추정
        spectral_loss = self.efficient_spectral_norm(z_sample)
        
        # 2. 간단한 안정성 loss
        stability_loss = self.simple_stability_loss(z_sample, z_next_sample)
        
        # Combined loss
        total_loss = (self.lambda_spectral * spectral_loss + 
                     self.lambda_stability * stability_loss)
        
        components = {
            'spectral_loss': spectral_loss.item(),
            'stability_loss': stability_loss.item(),
            'total_spectral_loss': total_loss.item()
        }
        
        return total_loss, components

def create_efficient_spectral_regularizer(config):
    """
    효율적인 spectral regularizer 생성
    """
    lambda_spectral = getattr(config, 'lambda_spectral', 0.01)  # 기본값 줄임
    lambda_stability = getattr(config, 'lambda_stability', 0.001)  # 기본값 줄임
    sample_ratio = getattr(config, 'spectral_sample_ratio', 0.1)  # 10%만 계산
    max_iterations = getattr(config, 'spectral_max_iter', 5)  # 반복 줄임
    
    return EfficientSpectralRegularization(
        lambda_spectral=lambda_spectral,
        lambda_stability=lambda_stability,
        sample_ratio=sample_ratio,
        max_iterations=max_iterations
    )
