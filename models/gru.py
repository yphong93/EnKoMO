import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    GRU (Gated Recurrent Unit) model for time series forecasting
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.latent_dim
        self.num_layers = 2
        self.dropout = 0.1
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.reconstruction_head = nn.Linear(self.hidden_dim, self.input_dim)
        self.prediction_head = nn.Linear(self.hidden_dim, self.input_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x, pred_len=0):
        """
        x: [B, T, D]
        """
        batch_size, seq_len, _ = x.shape
        
        # GRU forward pass
        gru_out, hidden = self.gru(x)  # gru_out: [B, T, hidden_dim], hidden: [num_layers, B, hidden_dim]
        
        # Apply dropout
        gru_out = self.dropout_layer(gru_out)
        
        # Reconstruction
        x_rec = self.reconstruction_head(gru_out)  # [B, T, D]
        
        # Dynamic reconstruction (same as reconstruction for GRU)
        x_dyn = x_rec
        
        # Prediction
        x_pred = None
        if pred_len > 0:
            preds = []
            current_hidden = hidden
            current_input = x[:, -1:, :]  # Last time step
            
            for _ in range(pred_len):
                # One step forward
                gru_step_out, current_hidden = self.gru(current_input, current_hidden)
                gru_step_out = self.dropout_layer(gru_step_out)
                
                # Predict
                pred_step = self.prediction_head(gru_step_out)  # [B, 1, D]
                preds.append(pred_step)
                
                # Use prediction as next input
                current_input = pred_step
            
            x_pred = torch.cat(preds, dim=1)  # [B, pred_len, D]
        
        # Latent representations
        z = gru_out  # [B, T, hidden_dim]
        z_dyn = z  # Same as z for GRU
        
        return x_rec, x_dyn, x_pred, z, z_dyn
