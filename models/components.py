import torch
import torch.nn as nn

# Try to import Mamba, fall back to pure PyTorch implementation if not found
try:
    from mamba_ssm import Mamba
    HAS_MAMBA_LIB = True
except ImportError:
    HAS_MAMBA_LIB = False
    print("Mamba library not found. Using Pure PyTorch implementation.")

class PureMambaLayer(nn.Module):
    """Pure PyTorch implementation of Mamba layer."""
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.x_proj = nn.Linear(d_model, d_state + d_model + d_model)
        self.dt_proj = nn.Linear(d_state, d_model, bias=True)
        
        nn.init.constant_(self.dt_proj.weight, 0.001)
        nn.init.constant_(self.dt_proj.bias, 0.1)
        A_init = torch.linspace(0.5, 2.0, d_state).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_model) * 0.01)
        self.act = nn.SiLU()

    def forward(self, x):
        batch, seq, _ = x.shape
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.chunk(2, dim=-1)
        x_in = self.act(x_in)
        
        ssm_params = self.x_proj(x_in)
        delta, B, C = torch.split(ssm_params, [self.d_state, self.d_model, self.d_model], dim=-1)
        delta = torch.nn.functional.softplus(self.dt_proj(delta))
        delta = torch.clamp(delta, min=1e-5, max=0.1)
        A = -torch.exp(self.A_log)
        A = torch.clamp(A, min=-5.0, max=-0.01)
        
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)
        y_list = []
        for t in range(seq):
            dt_t = delta[:, t, :].unsqueeze(-1)
            B_t = B[:, t, :].unsqueeze(-1)
            C_t = C[:, t, :].unsqueeze(-1)
            x_t = x_in[:, t, :].unsqueeze(-1)
            dA = torch.exp(A * dt_t)
            dA = torch.clamp(dA, min=1e-6, max=1.0)
            dB = B_t * dt_t
            h = h * dA + dB * x_t
            h = torch.clamp(h, min=-100, max=100)
            y_t = torch.sum(h * C_t, dim=-1) + self.D * x_t.squeeze(-1)
            y_list.append(y_t)
        y = torch.stack(y_list, dim=1)
        return self.out_proj(y * self.act(res))

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        if HAS_MAMBA_LIB:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        else:
            self.mamba = PureMambaLayer(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
            
    def forward(self, x): 
        return self.norm(self.mamba(x))

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.hidden1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hidden2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        h = self.input_proj(x)
        h = h + self.hidden1(h)
        h = h + self.hidden2(h)
        z = self.output_proj(h)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        self.hidden1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hidden2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, z):
        h = self.input_proj(z)
        h = h + self.hidden1(h)
        h = h + self.hidden2(h)
        x = self.output_proj(h)
        return x

