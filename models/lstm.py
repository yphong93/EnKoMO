import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM-based sequence prediction model
    """
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.latent_dim,
            num_layers=2,
            batch_first=True
        )
        self.head = nn.Linear(config.latent_dim, config.input_dim)
        
    def forward(self, x, pred_len=0):
        out, (h, c) = self.lstm(x)
        x_rec = self.head(out)
        
        x_pred = None
        if pred_len > 0:
            preds = []
            h_t = h
            c_t = c
            last_x = x[:, -1:, :]
            
            for _ in range(pred_len):
                out_t, (h_t, c_t) = self.lstm(last_x, (h_t, c_t))
                pred_t = self.head(out_t)
                preds.append(pred_t)
                last_x = pred_t
            
            x_pred = torch.cat(preds, dim=1)
            
        return x_rec, x_rec, x_pred, None, None

