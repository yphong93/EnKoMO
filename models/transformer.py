import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    """
    Transformer model for time series forecasting
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.d_model = config.latent_dim
        self.nhead = 8
        self.num_layers = 4
        self.dropout = 0.1
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.input_dim)
        
        # Prediction head
        self.prediction_head = nn.Linear(self.d_model, self.input_dim)
        
    def forward(self, x, pred_len=0):
        """
        x: [B, T, D]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x_proj = self.input_projection(x)  # [B, T, d_model]
        
        # Add positional encoding
        x_pos = self.pos_encoder(x_proj.transpose(0, 1)).transpose(0, 1)  # [B, T, d_model]
        
        # Transformer encoding
        encoded = self.transformer_encoder(x_pos)  # [B, T, d_model]
        
        # Reconstruction
        x_rec = self.output_projection(encoded)  # [B, T, D]
        
        # Dynamic reconstruction (same as reconstruction)
        x_dyn = x_rec
        
        # Prediction
        x_pred = None
        if pred_len > 0:
            # Use last encoded state for prediction
            last_state = encoded[:, -1:, :]  # [B, 1, d_model]
            
            preds = []
            current_state = last_state
            
            for _ in range(pred_len):
                # Predict next step
                pred_step = self.prediction_head(current_state)  # [B, 1, D]
                preds.append(pred_step)
                
                # Update state (simple approach: project prediction back to latent space)
                current_state = self.input_projection(pred_step)
            
            x_pred = torch.cat(preds, dim=1)  # [B, pred_len, D]
        
        # Latent representations
        z = encoded  # [B, T, d_model]
        z_dyn = z  # Same as z for transformer
        
        return x_rec, x_dyn, x_pred, z, z_dyn
