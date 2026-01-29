import torch
import torch.nn as nn

class LinearModel(nn.Module):
    """
    Simple Linear model for time series forecasting
    Baseline model using linear transformations
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.seq_len = config.seq_len
        
        # Flatten input for linear layers
        self.flatten_dim = self.seq_len * self.input_dim
        
        # Linear layers for reconstruction
        self.reconstruction_net = nn.Sequential(
            nn.Linear(self.flatten_dim, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, self.flatten_dim)
        )
        
        # Linear layers for prediction
        self.prediction_net = nn.Sequential(
            nn.Linear(self.flatten_dim, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, self.input_dim)  # Predict one step
        )
        
        # Latent representation layer
        self.latent_net = nn.Sequential(
            nn.Linear(self.flatten_dim, config.latent_dim),
            nn.ReLU()
        )
        
    def forward(self, x, pred_len=0):
        """
        x: [B, T, D]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Flatten input
        x_flat = x.reshape(batch_size, -1)  # [B, T*D]
        
        # Reconstruction
        x_rec_flat = self.reconstruction_net(x_flat)  # [B, T*D]
        x_rec = x_rec_flat.reshape(batch_size, seq_len, input_dim)  # [B, T, D]
        
        # Dynamic reconstruction (same as reconstruction for linear model)
        x_dyn = x_rec
        
        # Latent representation
        z_flat = self.latent_net(x_flat)  # [B, latent_dim]
        z = z_flat.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, latent_dim]
        z_dyn = z
        
        # Prediction
        x_pred = None
        if pred_len > 0:
            preds = []
            current_input = x_flat
            
            for _ in range(pred_len):
                # Predict one step
                pred_step = self.prediction_net(current_input)  # [B, D]
                preds.append(pred_step.unsqueeze(1))  # [B, 1, D]
                
                # Update input: shift and add prediction
                # Simple approach: use last (seq_len-1) steps + prediction
                current_seq = current_input.reshape(batch_size, seq_len, input_dim)
                new_seq = torch.cat([current_seq[:, 1:, :], pred_step.unsqueeze(1)], dim=1)
                current_input = new_seq.reshape(batch_size, -1)
            
            x_pred = torch.cat(preds, dim=1)  # [B, pred_len, D]
        
        return x_rec, x_dyn, x_pred, z, z_dyn
