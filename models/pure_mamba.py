import torch
import torch.nn as nn
from .components import MambaBlock

class PureMamba(nn.Module):
    """
    Pure Mamba Model (No Autoencoder, operates on input dimension directly)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.input_proj = nn.Linear(config.input_dim, config.latent_dim)
        self.layers = nn.ModuleList([
            MambaBlock(config.latent_dim, d_state=32)
            for _ in range(2)
        ])
        self.output_proj = nn.Linear(config.latent_dim, config.input_dim)
        
    def forward(self, x, pred_len=0):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        
        x_rec = self.output_proj(h)
        
        x_pred = None
        if pred_len > 0:
            curr_x = x
            preds = []
            for _ in range(pred_len):
                h_ctx = self.input_proj(curr_x[:, -self.seq_len:, :])
                for layer in self.layers:
                    h_ctx = layer(h_ctx)
                next_x = self.output_proj(h_ctx[:, -1:, :])
                preds.append(next_x)
                curr_x = torch.cat([curr_x, next_x], dim=1)
            
            x_pred = torch.cat(preds, dim=1)
            
        return x_rec, x_rec, x_pred, None, None

