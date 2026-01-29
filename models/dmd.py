import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd, pinv

class DMD(nn.Module):
    """
    Improved Dynamic Mode Decomposition (DMD)
    Classical linear operator method for dynamical systems
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = min(config.latent_dim, config.seq_len - 1)
        self.K = None  # Koopman operator
        self.fitted = False
        self.device = None
        
    def fit_dmd_global(self, train_loader):
        """
        Fit DMD operator from entire training dataset
        """
        X_list = []
        Y_list = []
        
        # Collect all training data
        for batch_x, batch_y in train_loader:
            batch_size, seq_len, input_dim = batch_x.shape
            if seq_len > 1:
                # Create time-shifted pairs
                X_batch = batch_x[:, :-1, :].reshape(-1, input_dim)
                Y_batch = batch_x[:, 1:, :].reshape(-1, input_dim)
                X_list.append(X_batch.cpu().numpy())
                Y_list.append(Y_batch.cpu().numpy())
        
        if len(X_list) == 0:
            return
        
        # Concatenate all data
        X_all = np.vstack(X_list)
        Y_all = np.vstack(Y_list)
        
        # Compute DMD
        try:
            # Use SVD for robust computation
            U, s, Vh = svd(X_all.T, full_matrices=False)
            
            # Truncate to rank (avoid numerical issues)
            rank = min(self.rank, len(s), X_all.shape[1])
            
            # Filter out small singular values
            threshold = 1e-10
            valid_indices = s > threshold
            if np.sum(valid_indices) < rank:
                rank = np.sum(valid_indices)
            
            if rank == 0:
                # Fallback to identity
                self.K = torch.eye(X_all.shape[1]).float()
            else:
                U = U[:, :rank]
                s = s[:rank]
                Vh = Vh[:rank, :]
                
                # Compute DMD operator with regularization
                S_inv = np.diag(1.0 / s)
                A = Y_all.T @ Vh.T @ S_inv @ U.T
                
                self.K = torch.from_numpy(A.T).float()
            
            self.fitted = True
            
        except Exception as e:
            print(f"DMD fitting failed: {e}. Using identity matrix.")
            self.K = torch.eye(X_all.shape[1]).float()
            self.fitted = True
    
    def forward(self, x, pred_len=0):
        """
        x: [B, T, D]
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        if self.device is None:
            self.device = device
        
        # Move K to correct device
        if self.K is not None and self.K.device != device:
            self.K = self.K.to(device)
        
        # If not fitted yet, use simple local fitting as fallback
        if not self.fitted or self.K is None:
            self.fit_local_dmd(x)
        
        # Reconstruct using DMD operator
        if self.K is not None:
            x_flat = x.reshape(-1, input_dim)
            try:
                x_rec_flat = torch.matmul(x_flat, self.K.T)
                x_rec = x_rec_flat.reshape(batch_size, seq_len, input_dim)
            except:
                x_rec = x  # Fallback
        else:
            x_rec = x
        
        # Dynamic reconstruction (same as reconstruction for DMD)
        x_dyn = x_rec
        
        # Prediction
        x_pred = None
        if pred_len > 0 and self.K is not None:
            preds = []
            x_curr = x[:, -1, :]  # Last time step
            
            for _ in range(pred_len):
                try:
                    x_next = torch.matmul(x_curr, self.K.T)
                    preds.append(x_next.unsqueeze(1))
                    x_curr = x_next
                except:
                    # Fallback: repeat last state
                    preds.append(x_curr.unsqueeze(1))
            
            x_pred = torch.cat(preds, dim=1)
        
        return x_rec, x_dyn, x_pred, None, None
    
    def fit_local_dmd(self, x):
        """
        Local DMD fitting for single batch (fallback method)
        """
        batch_size, seq_len, input_dim = x.shape
        
        if seq_len <= 1:
            self.K = torch.eye(input_dim).float().to(x.device)
            return
        
        try:
            # Use all samples in batch
            X = x[:, :-1, :].reshape(-1, input_dim)
            Y = x[:, 1:, :].reshape(-1, input_dim)
            
            X_np = X.detach().cpu().numpy()
            Y_np = Y.detach().cpu().numpy()
            
            # Use pseudo-inverse for robustness
            X_pinv = pinv(X_np, rcond=1e-6)
            A = Y_np.T @ X_pinv.T
            
            self.K = torch.from_numpy(A.T).float().to(x.device)
            
        except Exception as e:
            print(f"Local DMD fitting failed: {e}. Using identity matrix.")
            self.K = torch.eye(input_dim).float().to(x.device)

