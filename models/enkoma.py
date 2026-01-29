import torch
import torch.nn as nn
from .components import Encoder, Decoder, MambaBlock
from .spectral_loss import SpectralRegularization, create_spectral_regularizer

class EnKoMa(nn.Module):
    """
    EnKoMa: Enhanced Koopman via Mamba
    State-dependent Koopman operator using Mamba's selective state space model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=64,
            dropout=0.1 if config.seq_len >= 200 else 0.0
        )
        
        self.decoder = Decoder(
            latent_dim=config.latent_dim,
            output_dim=config.input_dim,
            hidden_dim=64,
            dropout=0.1 if config.seq_len >= 200 else 0.0
        )
        
        # Dynamics Module (Mamba) - State-dependent Koopman operator
        self.dynamics_blocks = nn.ModuleList([
            MambaBlock(d_model=config.latent_dim, d_state=32)
            for _ in range(2)
        ])
        
        self.dynamics_norm = nn.LayerNorm(config.latent_dim)
        
        # Spectral Regularization (논문의 핵심 안정화 기법)
        self.use_spectral_loss = getattr(config, 'use_spectral_loss', True)
        if self.use_spectral_loss:
            self.spectral_regularizer = create_spectral_regularizer(config)
    
    def evolution_map(self, z):
        """
        Full Evolution Map: F(z) = z + Mamba(z)
        This implements the residual structure from the EnKoMa paper.
        
        Args:
            z: [B, T, D] or [B, D] or [D] latent tensor
        
        Returns:
            F(z): [same shape as z] evolved latent state
        """
        original_shape = z.shape
        if len(z.shape) == 1:
            z = z.unsqueeze(0).unsqueeze(0)
        elif len(z.shape) == 2:
            z = z.unsqueeze(1)
        
        # Apply Mamba blocks
        mamba_out = z
        for block in self.dynamics_blocks:
            mamba_out = mamba_out + block(mamba_out)
        mamba_out = self.dynamics_norm(mamba_out)
        
        # Extract Mamba(z) component
        mamba_z = mamba_out - z
        
        # Residual structure: F(z) = z + Mamba(z)
        result = z + mamba_z
        
        # Reshape back to original shape
        if len(original_shape) == 1:
            result = result.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            result = result.squeeze(1)
        
        return result
        
    def forward(self, x, pred_len=0):
        # x: [B, seq_len, input_dim]
        z = self.encoder(x)
        
        # Static Reconstruction
        x_rec = self.decoder(z)
        
        # Dynamics Propagation in Latent Space
        z_dyn = z
        for block in self.dynamics_blocks:
            z_dyn = z_dyn + block(z_dyn)
        z_dyn = self.dynamics_norm(z_dyn)
        
        # Dynamic Reconstruction (One-step ahead prediction)
        x_dyn = self.decoder(z_dyn)
        
        # Multi-step Prediction
        x_pred = None
        if pred_len > 0:
            batch_size, seq_len, _ = z.shape
            curr_z = z if self.training else z.detach()
            pred_zs = []
            
            for _ in range(pred_len):
                context = curr_z[:, -seq_len:, :]
                
                dyn_out = context
                for block in self.dynamics_blocks:
                    dyn_out = dyn_out + block(dyn_out)
                dyn_out = self.dynamics_norm(dyn_out)
                
                z_next = dyn_out[:, -1:, :]
                pred_zs.append(z_next)
                curr_z = torch.cat([curr_z, z_next], dim=1)
            
            pred_zs = torch.cat(pred_zs, dim=1)
            x_pred = self.decoder(pred_zs)

        return x_rec, x_dyn, x_pred, z, z_dyn
    
    def compute_spectral_loss(self, z):
        """
        Compute spectral regularization loss for stability
        
        Args:
            z: [B, seq_len, D] latent states
        
        Returns:
            spectral_loss: spectral regularization loss
            components: dictionary with loss breakdown
        """
        if not self.use_spectral_loss or not hasattr(self, 'spectral_regularizer'):
            return torch.tensor(0.0, device=z.device), {}
        
        return self.spectral_regularizer(self, z)
    
    def compute_total_loss(self, x_rec, x_dyn, x_pred, x_true, z, 
                          alpha_rec=1.0, alpha_pred=1.0, alpha_spectral=0.1, 
                          epoch=0, spectral_interval=10):
        """
        Compute total loss including periodic spectral regularization
        
        Args:
            x_rec: reconstructed data
            x_dyn: one-step dynamics prediction
            x_pred: multi-step prediction (can be None)
            x_true: ground truth data
            z: latent states
            alpha_rec: reconstruction loss weight
            alpha_pred: prediction loss weight  
            alpha_spectral: spectral loss weight
            epoch: current training epoch
            spectral_interval: compute spectral loss every N epochs
        
        Returns:
            total_loss: combined loss
            loss_components: dictionary with individual losses
        """
        mse_loss = nn.MSELoss()
        
        # Reconstruction loss - compare with input sequence part only
        input_len = x_rec.shape[1]  # Should be seq_len
        rec_loss = mse_loss(x_rec, x_true[:, :input_len, :])
        
        # One-step prediction loss
        if x_dyn.shape[1] > 1:
            # Compare dynamics prediction with next timesteps in input sequence
            dyn_len = min(x_dyn.shape[1], input_len)
            pred_loss = mse_loss(x_dyn[:, 1:dyn_len, :], x_true[:, 1:dyn_len, :])
        else:
            pred_loss = torch.tensor(0.0, device=x_rec.device)
        
        # Multi-step prediction loss
        multistep_loss = torch.tensor(0.0, device=x_rec.device)
        if x_pred is not None:
            # Compare with future ground truth if available
            pred_len = x_pred.shape[1]
            if x_true.shape[1] > x_rec.shape[1]:
                future_true = x_true[:, -pred_len:, :]
                multistep_loss = mse_loss(x_pred, future_true)
        
        # Spectral regularization loss (computed periodically)
        spectral_loss = torch.tensor(0.0, device=x_rec.device)
        spectral_components = {}
        
        # Only compute spectral loss every N epochs (performance optimization)
        if epoch % spectral_interval == 0:
            spectral_loss, spectral_components = self.compute_spectral_loss(z)
        
        # Combined loss
        total_loss = (alpha_rec * rec_loss + 
                     alpha_pred * (pred_loss + multistep_loss) + 
                     alpha_spectral * spectral_loss)
        
        loss_components = {
            'reconstruction_loss': rec_loss.item(),
            'prediction_loss': pred_loss.item(),
            'multistep_loss': multistep_loss.item(),
            'spectral_loss': spectral_loss.item() if isinstance(spectral_loss, torch.Tensor) else spectral_loss,
            'total_loss': total_loss.item()
        }
        loss_components.update(spectral_components)
        
        return total_loss, loss_components

