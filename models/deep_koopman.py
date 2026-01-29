import torch
import torch.nn as nn
from .components import Encoder, Decoder

class DeepKoopman(nn.Module):
    """
    Deep Koopman Autoencoder
    Uses a fixed linear matrix K for latent dynamics: z_{t+1} = K * z_t
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
        # Fixed linear Koopman operator
        self.K = nn.Linear(config.latent_dim, config.latent_dim, bias=False)
        
    def forward(self, x, pred_len=0):
        # x: [B, T, D]
        z = self.encoder(x)
        
        # Linear evolution: z_{t+1} = K * z_t
        z_dyn = self.K(z)
        
        x_rec = self.decoder(z)
        x_dyn = self.decoder(z_dyn)
        
        x_pred = None
        if pred_len > 0:
            z_curr = z[:, -1, :]
            preds = []
            for _ in range(pred_len):
                z_next = self.K(z_curr)
                preds.append(z_next.unsqueeze(1))
                z_curr = z_next
            
            z_pred_seq = torch.cat(preds, dim=1)
            x_pred = self.decoder(z_pred_seq)
            
        return x_rec, x_dyn, x_pred, z, z_dyn

