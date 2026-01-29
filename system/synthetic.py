import numpy as np
import torch
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler

class SyntheticSystem:
    """
    Synthetic dynamical systems for benchmarking
    Supports: Lorenz, VanDerPol, Duffing, Burgers, KuramotoSivashinsky
    """
    
    @staticmethod
    def lorenz(state, t, sigma=10, rho=28, beta=8/3):
        """Lorenz system"""
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    
    @staticmethod
    def van_der_pol(state, t, mu=1.0):
        """Van der Pol oscillator"""
        x, y = state
        return [y, mu * (1 - x**2) * y - x]
    
    @staticmethod
    def duffing(state, t, alpha=-1.0, beta=1.0, delta=0.3, gamma=0.5, omega=1.2):
        """Duffing oscillator"""
        x, y = state
        return [y, -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)]
    
    @staticmethod
    def burgers_rhs(u, t, k, nu, N):
        """
        Burgers equation: u_t + u*u_x = nu*u_xx
        Using Fourier spectral method (periodic BC)
        
        Args:
            u: [N] state vector in physical space
            t: time
            k: [N] wavenumber array
            nu: viscosity
            N: number of grid points
        
        Returns:
            du/dt: [N] time derivative
        """
        # Transform to Fourier space
        u_hat = np.fft.fft(u)
        
        # Compute u_x in Fourier space: ik * u_hat
        u_x_hat = 1j * k * u_hat
        u_x = np.real(np.fft.ifft(u_x_hat))
        
        # Nonlinear term: u * u_x
        nonlinear = u * u_x
        
        # Transform nonlinear term to Fourier space
        nonlinear_hat = np.fft.fft(nonlinear)
        
        # Diffusion term: -nu * k^2 * u_hat
        diffusion_hat = -nu * k**2 * u_hat
        
        # Total RHS in Fourier space
        du_hat_dt = -nonlinear_hat + diffusion_hat
        
        # Transform back to physical space
        du_dt = np.real(np.fft.ifft(du_hat_dt))
        
        return du_dt
    
    @staticmethod
    def kuramoto_sivashinsky_rhs(u, t, k, L, N):
        """
        Kuramoto-Sivashinsky equation: u_t + u_xxxx + u_xx + u*u_x = 0
        Using Fourier spectral method (periodic BC)
        
        Args:
            u: [N] state vector in physical space
            t: time
            k: [N] wavenumber array
            L: domain length parameter
            N: number of grid points
        
        Returns:
            du/dt: [N] time derivative
        """
        # Transform to Fourier space
        u_hat = np.fft.fft(u)
        
        # Compute u_x in Fourier space: ik * u_hat
        u_x_hat = 1j * k * u_hat
        u_x = np.real(np.fft.ifft(u_x_hat))
        
        # Nonlinear term: u * u_x
        nonlinear = u * u_x
        nonlinear_hat = np.fft.fft(nonlinear)
        
        # Linear terms: -u_xxxx - u_xx = -(ik)^4 * u_hat - (ik)^2 * u_hat
        # k^4 and k^2 terms
        linear_hat = -(k**4) * u_hat - (k**2) * u_hat
        
        # Total RHS in Fourier space
        du_hat_dt = -nonlinear_hat + linear_hat
        
        # Transform back to physical space
        du_dt = np.real(np.fft.ifft(du_hat_dt))
        
        return du_dt
    
    @classmethod
    def generate_trajectory(cls, system_name, init_state, t_span, dt, system_params=None):
        """
        Generate trajectory for a given system
        
        Args:
            system_name: 'Lorenz', 'VanDerPol', 'Duffing', 'Burgers', 'KuramotoSivashinsky'
            init_state: Initial state vector
            t_span: Total time span
            dt: Time step
            system_params: Dictionary of system parameters
        
        Returns:
            t: Time array
            trajectory: [T, D] array of states
        """
        t = np.arange(0, t_span, dt)
        
        if system_name == 'Lorenz':
            params = {'sigma': 10, 'rho': 28, 'beta': 8/3}
            if system_params:
                params.update(system_params)
            func = lambda state, t: cls.lorenz(state, t, **params)
            init = init_state if init_state else [1.0, 1.0, 1.0]
            trajectory = odeint(func, init, t)
            
        elif system_name == 'VanDerPol':
            params = {'mu': 1.0}
            if system_params:
                params.update(system_params)
            func = lambda state, t: cls.van_der_pol(state, t, **params)
            init = init_state if init_state else [2.0, 0.0]
            trajectory = odeint(func, init, t)
            
        elif system_name == 'Duffing':
            params = {'alpha': -1.0, 'beta': 1.0, 'delta': 0.3, 'gamma': 0.5, 'omega': 1.2}
            if system_params:
                params.update(system_params)
            func = lambda state, t: cls.duffing(state, t, **params)
            init = init_state if init_state else [0.0, 0.0]
            trajectory = odeint(func, init, t)
            
        elif system_name == 'Burgers':
            # Burgers equation using Fourier spectral method
            N = system_params.get('N', 128) if system_params else 128
            nu = system_params.get('nu', 0.01) if system_params else 0.01
            L = system_params.get('L', 2.0 * np.pi) if system_params else 2.0 * np.pi
            
            # Domain: [0, L] with periodic BC
            x = np.linspace(0, L, N, endpoint=False)
            dx = L / N
            
            # Wavenumbers for Fourier transform
            k = 2 * np.pi * np.fft.fftfreq(N, dx)
            
            # Initial condition
            if init_state is None:
                # Gaussian pulse or sinusoidal
                init_type = system_params.get('init_type', 'sinusoidal') if system_params else 'sinusoidal'
                if init_type == 'sinusoidal':
                    u0 = np.sin(2 * np.pi * x / L)
                elif init_type == 'gaussian':
                    center = L / 2
                    width = L / 10
                    u0 = np.exp(-((x - center) / width) ** 2)
                else:
                    u0 = np.random.randn(N) * 0.1
            else:
                u0 = np.array(init_state)
                if len(u0) != N:
                    raise ValueError(f"Initial state length {len(u0)} != N={N}")
            
            # RHS function
            def rhs(t, u):
                return cls.burgers_rhs(u, t, k, nu, N)
            
            # Solve using RK45
            sol = solve_ivp(rhs, [0, t_span], u0, t_eval=t, method='RK45', rtol=1e-6, atol=1e-8)
            trajectory = sol.y.T  # [T, N]
            
        elif system_name == 'KuramotoSivashinsky':
            # Kuramoto-Sivashinsky equation using Fourier spectral method
            N = system_params.get('N', 128) if system_params else 128
            L = system_params.get('L', 16.0 * np.pi) if system_params else 16.0 * np.pi
            
            # Domain: [0, L] with periodic BC
            x = np.linspace(0, L, N, endpoint=False)
            dx = L / N
            
            # Wavenumbers for Fourier transform
            k = 2 * np.pi * np.fft.fftfreq(N, dx)
            
            # Initial condition
            if init_state is None:
                # Random perturbation
                u0 = np.random.randn(N) * 0.1
                # Add some low-frequency components
                for i in range(1, 5):
                    u0 += 0.1 * np.sin(2 * np.pi * i * x / L)
            else:
                u0 = np.array(init_state)
                if len(u0) != N:
                    raise ValueError(f"Initial state length {len(u0)} != N={N}")
            
            # RHS function
            def rhs(t, u):
                return cls.kuramoto_sivashinsky_rhs(u, t, k, L, N)
            
            # Solve using RK45
            sol = solve_ivp(rhs, [0, t_span], u0, t_eval=t, method='RK45', rtol=1e-6, atol=1e-8)
            trajectory = sol.y.T  # [T, N]
            
        else:
            raise ValueError(f"Unknown system: {system_name}")
        
        return t, trajectory
    
    @classmethod
    def get_data(cls, system_name, config, mode='train', seed=None):
        """
        Generate data for training/testing
        
        Args:
            system_name: 'Lorenz', 'VanDerPol', 'Duffing', 'Burgers', or 'KuramotoSivashinsky'
            config: Configuration object with system parameters
            mode: 'train' or 'test'
            seed: Random seed
        
        Returns:
            data: [N, seq_len+pred_len, D] tensor
            scaler: Fitted StandardScaler for normalization
        
        Note:
            For Burgers and KuramotoSivashinsky, system_params.N must match config.input_dim
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Determine time span and initial conditions
        if mode == 'train':
            t_span = config.train_time
            if system_name == 'Lorenz':
                init_state = [1.0, 1.0, 1.0]
            elif system_name == 'VanDerPol':
                init_state = [2.0, 0.0]
            elif system_name == 'Duffing':
                init_state = [0.0, 0.0]
            elif system_name == 'Burgers':
                init_state = None  # Will be generated based on init_type
            elif system_name == 'KuramotoSivashinsky':
                init_state = None  # Will be generated randomly
        else:
            t_span = config.test_time
            if system_name == 'Lorenz':
                init_state = [2.0, 3.0, 5.0]
            elif system_name == 'VanDerPol':
                init_state = [1.5, 0.5]
            elif system_name == 'Duffing':
                init_state = [0.5, 0.0]
            elif system_name == 'Burgers':
                init_state = None  # Will be generated based on init_type
            elif system_name == 'KuramotoSivashinsky':
                init_state = None  # Will be generated randomly
        
        # Get system parameters from config
        system_params = getattr(config, 'system_params', {})
        
        # For high-dimensional systems, ensure N matches input_dim
        if system_name in ['Burgers', 'KuramotoSivashinsky']:
            if 'N' not in system_params:
                # Use input_dim as N if not specified
                system_params = system_params.copy()
                system_params['N'] = config.input_dim
            # Verify N matches input_dim
            if system_params.get('N') != config.input_dim:
                raise ValueError(
                    f"For {system_name}: system_params.N ({system_params.get('N')}) "
                    f"must match config.input_dim ({config.input_dim})"
                )
        
        # Generate full trajectory
        t, trajectory = cls.generate_trajectory(
            system_name, init_state, t_span, config.dt, system_params
        )
        
        # Verify trajectory dimension matches input_dim
        if trajectory.shape[1] != config.input_dim:
            raise ValueError(
                f"Trajectory dimension {trajectory.shape[1]} does not match "
                f"config.input_dim {config.input_dim} for {system_name}"
            )
        
        # Add noise if specified
        if hasattr(config, 'noise_level') and config.noise_level > 0:
            noise = np.random.normal(0, config.noise_level, trajectory.shape)
            trajectory = trajectory + noise
        
        # Normalize
        scaler = StandardScaler()
        trajectory_scaled = scaler.fit_transform(trajectory)
        
        # Create sequences
        seq_len = config.seq_len
        pred_len = config.pred_len
        total_len = seq_len + pred_len
        
        num_samples = len(trajectory_scaled) - total_len + 1
        if num_samples <= 0:
            raise ValueError(f"Trajectory too short: {len(trajectory_scaled)} < {total_len}")
        
        sequences = []
        for i in range(num_samples):
            seq = trajectory_scaled[i:i+total_len]
            sequences.append(seq)
        
        data = np.array(sequences)
        data_tensor = torch.from_numpy(data).float()
        
        return data_tensor, scaler

