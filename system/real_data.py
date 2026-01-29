import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from scipy import signal
from scipy.stats import zscore
import os

class RealDataSystem:
    """
    Real-world time series data loader
    Supports: ETT, SST, AirQuality, NASA Bearing, EnergyConsumption
    """
    
    @staticmethod
    def detrend(data, method='linear'):
        """
        Remove trend from data with enhanced methods
        
        Args:
            data: [T, D] array
            method: 'linear', 'polynomial', 'seasonal', 'diff', or 'none'
        
        Returns:
            detrended_data: [T, D] array
            trend: trend component (for reconstruction)
        """
        if method == 'none':
            return data, np.zeros_like(data)
            
        T, D = data.shape
        trend = np.zeros_like(data)
        detrended = np.zeros_like(data)
        
        if method == 'linear':
            # Linear detrending
            x = np.arange(T)
            for d in range(D):
                coeffs = np.polyfit(x, data[:, d], 1)
                trend[:, d] = np.polyval(coeffs, x)
                detrended[:, d] = data[:, d] - trend[:, d]
            return detrended, trend
            
        elif method == 'polynomial':
            # Polynomial detrending (degree 2)
            x = np.arange(T)
            for d in range(D):
                coeffs = np.polyfit(x, data[:, d], 2)
                trend[:, d] = np.polyval(coeffs, x)
                detrended[:, d] = data[:, d] - trend[:, d]
            return detrended, trend
            
        elif method == 'seasonal':
            # Seasonal detrending (moving average)
            window_size = min(12, T // 4)  # Adaptive window
            for d in range(D):
                if window_size > 1:
                    # Use pandas rolling for better handling
                    series = pd.Series(data[:, d])
                    moving_avg = series.rolling(window=window_size, center=True).mean()
                    moving_avg = moving_avg.bfill().ffill()
                    trend[:, d] = moving_avg.values
                    detrended[:, d] = data[:, d] - trend[:, d]
                else:
                    # Fallback to linear for short series
                    x = np.arange(T)
                    coeffs = np.polyfit(x, data[:, d], 1)
                    trend[:, d] = np.polyval(coeffs, x)
                    detrended[:, d] = data[:, d] - trend[:, d]
            return detrended, trend
            
        elif method == 'diff':
            # First difference (returns one less time step)
            detrended = np.diff(data, axis=0)
            trend = data[:-1] - detrended
            return detrended, trend
            
        else:
            # Default to linear
            x = np.arange(T)
            for d in range(D):
                coeffs = np.polyfit(x, data[:, d], 1)
                trend[:, d] = np.polyval(coeffs, x)
                detrended[:, d] = data[:, d] - trend[:, d]
            return detrended, trend
    
    @classmethod
    def load_ett_data(cls, data_path, dataset='ETTh1', mode='train', split_ratio=0.7):
        """
        Load ETT (Electricity Transformer Temperature) dataset
        
        Data format: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
        - date: timestamp
        - HUFL, HULL, MUFL, MULL, LUFL, LULL, OT: numeric feature columns
        
        Args:
            data_path: Path to data directory
            dataset: Dataset name (ETTh1, ETTh2, ETTm1, ETTm2)
            mode: 'train' or 'test'
            split_ratio: Train/test split ratio
        
        Returns:
            data: [T, D] array where D=7 (7 numeric features)
        """
        file_path = os.path.join(data_path, f'{dataset}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ETT data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # ETT typically has columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
        # Use all numeric columns (date is string, so it's automatically excluded)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # If no numeric columns found, try selecting specific columns
        if len(numeric_cols) == 0:
            # Try specific ETT column names
            expected_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
            numeric_cols = [col for col in expected_cols if col in df.columns]
        
        if len(numeric_cols) == 0:
            raise ValueError(f"No numeric columns found in ETT dataset {dataset}. Available columns: {df.columns.tolist()}")
        
        data = df[numeric_cols].values
        
        # Convert to float, handling any string values
        data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').values
        
        # Remove any NaN values (forward fill then backward fill)
        df_data = pd.DataFrame(data)
        df_data = df_data.ffill().bfill()
        data = df_data.values
        
        # Remove any remaining NaN rows
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]
        
        if len(data) == 0:
            raise ValueError("No valid data after cleaning NaN values")
        
        # Split train/test
        split_idx = int(len(data) * split_ratio)
        if mode == 'train':
            data = data[:split_idx]
        else:
            data = data[split_idx:]
        
        return data
    
    @classmethod
    def load_sst_data(cls, data_path, mode='train', split_ratio=0.7, use_anomaly=False):
        """
        Load NOAA Sea Surface Temperature (SST) data from sst.csv
        
        Data format: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
        - YR: Year
        - MON: Month
        - NINO1+2, NINO3, NINO4, NINO3.4: Temperature values for different NINO regions
        - ANOM: Anomaly values (deviations from normal)
        
        Args:
            data_path: Path to data directory
            mode: 'train' or 'test'
            split_ratio: Train/test split ratio
            use_anomaly: If True, use ANOM (anomaly) values; if False, use temperature values
        
        Returns:
            data: [T, D] array where D=4 (4 NINO regions)
        """
        file_path = os.path.join(data_path, 'sst.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"SST data not found: {file_path}")
        
        # Read CSV with space-separated values (multiple spaces)
        try:
            df = pd.read_csv(file_path, sep=r'\s+', engine='python')
        except:
            # Fallback: try with default separator
            df = pd.read_csv(file_path)
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Extract NINO region data
        if use_anomaly:
            # Use anomaly values (ANOM columns)
            # Column names might be: 'ANOM', 'ANOM.1', 'ANOM.2', 'ANOM.3' or similar
            anom_cols = [col for col in df.columns if 'ANOM' in col.upper()]
            if len(anom_cols) >= 4:
                # Take first 4 ANOM columns
                data_cols = anom_cols[:4]
            else:
                # Fallback: use temperature values
                use_anomaly = False
        
        if not use_anomaly:
            # Use temperature values (NINO columns)
            # Column names: NINO1+2, NINO3, NINO4, NINO3.4
            nino_cols = []
            for col_name in ['NINO1+2', 'NINO3', 'NINO4', 'NINO3.4']:
                # Try exact match first
                if col_name in df.columns:
                    nino_cols.append(col_name)
                else:
                    # Try to find similar column (handle spaces, case)
                    matching = [c for c in df.columns if col_name.replace('+', '').replace('.', '') in c.replace('+', '').replace('.', '').replace(' ', '')]
                    if matching:
                        nino_cols.append(matching[0])
                    else:
                        # If still not found, try to find by position
                        # Based on file structure: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
                        pass
            
            # If we couldn't find columns by name, use position-based extraction
            if len(nino_cols) < 4:
                # Columns are: YR, MON, NINO1+2, ANOM, NINO3, ANOM, NINO4, ANOM, NINO3.4, ANOM
                # Temperature columns are at indices: 2, 4, 6, 8
                all_cols = df.columns.tolist()
                if len(all_cols) >= 9:
                    nino_cols = [all_cols[2], all_cols[4], all_cols[6], all_cols[8]]
                else:
                    # Use all numeric columns except YR and MON
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if 'YR' in numeric_cols:
                        numeric_cols.remove('YR')
                    if 'MON' in numeric_cols:
                        numeric_cols.remove('MON')
                    # Take first 4 numeric columns
                    nino_cols = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            
            data_cols = nino_cols[:4]  # Ensure we have exactly 4 columns
        
        # Extract data
        data_list = []
        for col in data_cols:
            if col in df.columns:
                values = df[col].values
                # Convert to float, handling any string values
                values = pd.to_numeric(values, errors='coerce')
                data_list.append(values)
            else:
                raise ValueError(f"Column '{col}' not found in SST data. Available columns: {df.columns.tolist()}")
        
        # Stack into [T, D] array
        if len(data_list) > 0:
            data = np.column_stack(data_list)
        else:
            raise ValueError("No valid data columns found in SST file")
        
        # Convert to DataFrame for easier manipulation
        df_data = pd.DataFrame(data)
        
        # Remove outliers using IQR method for each column (similar to AirQuality)
        for col_idx in range(df_data.shape[1]):
            Q1 = df_data.iloc[:, col_idx].quantile(0.25)
            Q3 = df_data.iloc[:, col_idx].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            
            # Cap outliers instead of removing them to preserve data length
            df_data.iloc[:, col_idx] = df_data.iloc[:, col_idx].clip(lower_bound, upper_bound)
        
        # Remove any NaN values (forward fill then backward fill)
        df_data = df_data.ffill().bfill()
        data = df_data.values
        
        # Remove any remaining NaN rows
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]
        
        if len(data) == 0:
            raise ValueError("No valid data after cleaning NaN values")
        
        # Split train/test
        split_idx = int(len(data) * split_ratio)
        if mode == 'train':
            data = data[:split_idx]
        else:
            data = data[split_idx:]
        
        return data
    
    @classmethod
    def load_air_quality_data(cls, data_path, mode='train', split_ratio=0.7):
        """
        Load Air Quality data from air_quality_dataset.csv
        
        Data format: PM2.5, PM10, NOx, NO2, SO2, VOCs, CO, CO2, CH4, 
                     Temperature, Humidity, Wind_Direction, Location_Type, Source_Label
        
        Args:
            data_path: Path to data directory
            mode: 'train' or 'test'
            split_ratio: Train/test split ratio
        
        Returns:
            data: [T, D] array where D=12 (12 numeric features)
        """
        file_path = os.path.join(data_path, 'air_quality_dataset.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Air Quality data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Extract numeric columns (exclude Location_Type and Source_Label)
        # Numeric columns: PM2.5, PM10, NOx, NO2, SO2, VOCs, CO, CO2, CH4, 
        #                  Temperature, Humidity, Wind_Direction
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude any non-numeric columns that might have been read as numeric
        # We want: PM2.5, PM10, NOx, NO2, SO2, VOCs, CO, CO2, CH4, Temperature, Humidity, Wind_Direction
        # Expected column names (case-insensitive)
        expected_cols = ['PM2.5', 'PM10', 'NOx', 'NO2', 'SO2', 'VOCs', 'CO', 'CO2', 'CH4', 
                        'Temperature', 'Humidity', 'Wind_Direction']
        
        # Find matching columns
        data_cols = []
        for exp_col in expected_cols:
            # Try exact match first
            if exp_col in df.columns:
                data_cols.append(exp_col)
            else:
                # Try case-insensitive match
                matching = [c for c in df.columns if c.upper() == exp_col.upper()]
                if matching:
                    data_cols.append(matching[0])
        
        # If we couldn't find all expected columns, use all numeric columns
        if len(data_cols) < 12:
            # Use all numeric columns (should be 12)
            data_cols = numeric_cols[:12] if len(numeric_cols) >= 12 else numeric_cols
        
        # Extract data
        data = df[data_cols].values
        
        # Convert to float, handling any string values
        data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').values
        
        # Enhanced data cleaning and preprocessing for AirQuality
        df_data = pd.DataFrame(data)
        
        # Remove outliers using IQR method for each column
        for col_idx in range(df_data.shape[1]):
            Q1 = df_data.iloc[:, col_idx].quantile(0.25)
            Q3 = df_data.iloc[:, col_idx].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            
            # Cap outliers instead of removing them to preserve data length
            df_data.iloc[:, col_idx] = df_data.iloc[:, col_idx].clip(lower_bound, upper_bound)
        
        # Fill NaN values with more sophisticated method
        # First forward fill, then backward fill, then interpolate remaining
        df_data = df_data.ffill().bfill()
        df_data = df_data.interpolate(method='linear', limit_direction='both')
        
        # If still NaN, fill with median
        for col_idx in range(df_data.shape[1]):
            if df_data.iloc[:, col_idx].isna().any():
                median_val = df_data.iloc[:, col_idx].median()
                df_data.iloc[:, col_idx] = df_data.iloc[:, col_idx].fillna(median_val)
        
        data = df_data.values
        
        # Remove any remaining NaN rows (should be none now)
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]
        
        if len(data) == 0:
            raise ValueError("No valid data after cleaning NaN values")
        
        # Split train/test
        split_idx = int(len(data) * split_ratio)
        if mode == 'train':
            data = data[:split_idx]
        else:
            data = data[split_idx:]
        
        return data
    
    @classmethod
    def load_nasa_bearing_data(cls, data_path, bearing_id=1, mode='train', split_ratio=0.7):
        """
        Load NASA Bearing Dataset (IMS) - Updated to use generated CSV
        
        Args:
            data_path: Path to bearing_1.csv file or directory containing it
            bearing_id: Bearing ID (1-4) or 'all' for all bearings
            mode: 'train' or 'test'
            split_ratio: Train/test split ratio
        """
        # Handle both file path and directory path
        if os.path.isdir(data_path):
            file_path = os.path.join(data_path, f'bearing_{bearing_id}.csv')
        else:
            file_path = data_path
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NASA Bearing data not found: {file_path}")
        
        print(f"Loading NASA Bearing data from: {file_path}")
        
        try:
            # Try pandas first
            df = pd.read_csv(file_path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(file_path, nrows=0).columns else False)
            
            # Select bearing-specific columns
            if bearing_id == 1:
                target_cols = ['Bearing1_X', 'Bearing1_Y']
            elif bearing_id == 2:
                target_cols = ['Bearing2_X', 'Bearing2_Y']
            elif bearing_id == 3:
                target_cols = ['Bearing3_X', 'Bearing3_Y']
            elif bearing_id == 4:
                target_cols = ['Bearing4_X', 'Bearing4_Y']
            else:
                # Use all bearing columns
                target_cols = [col for col in df.columns if 'Bearing' in col]
            
            # Extract selected columns
            available_cols = [col for col in target_cols if col in df.columns]
            if not available_cols:
                # Fallback to all numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                data = df[numeric_cols].values
            else:
                data = df[available_cols].values
            
            print(f"Selected columns: {available_cols if available_cols else 'all numeric'}")
            
        except ImportError:
            # Fallback to basic CSV reading without pandas
            print("Warning: pandas not available, using basic CSV reading")
            data = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                header = lines[0].strip().split(',')
                
                # Select columns based on bearing_id
                if bearing_id == 1:
                    target_cols = ['Bearing1_X', 'Bearing1_Y']
                elif bearing_id == 2:
                    target_cols = ['Bearing2_X', 'Bearing2_Y']
                elif bearing_id == 3:
                    target_cols = ['Bearing3_X', 'Bearing3_Y']
                elif bearing_id == 4:
                    target_cols = ['Bearing4_X', 'Bearing4_Y']
                else:
                    target_cols = [col for col in header if 'Bearing' in col and col != 'timestamp']
                
                # Find column indices
                col_indices = [header.index(col) for col in target_cols if col in header]
                
                # Read data
                for line in lines[1:]:
                    values = line.strip().split(',')
                    if len(values) > max(col_indices):
                        try:
                            row = [float(values[i]) for i in col_indices]
                            data.append(row)
                        except ValueError:
                            continue
            
            data = np.array(data)
        
        print(f"Loaded data shape: {data.shape}")
        
        # Train/test split
        split_idx = int(len(data) * split_ratio)
        if mode == 'train':
            result = data[:split_idx]
        else:
            result = data[split_idx:]
        
        print(f"Split data shape ({mode}): {result.shape}")
        return result
    
    @classmethod
    def load_energy_consumption_data(cls, data_path, mode='train', split_ratio=0.7, include_target=True):
        """
        Load Energy Consumption data from Energy_consumption_dataset.csv
        
        Data format: Month, Hour, DayOfWeek, Holiday, Temperature, Humidity, 
                     SquareFootage, Occupancy, HVACUsage, LightingUsage, 
                     RenewableEnergy, EnergyConsumption
        
        Args:
            data_path: Path to data directory
            mode: 'train' or 'test'
            split_ratio: Train/test split ratio
            include_target: If True, include EnergyConsumption in features; 
                          If False, use it only as target (for prediction tasks)
        
        Returns:
            data: [T, D] array where D=7-12 depending on include_target
        """
        file_path = os.path.join(data_path, 'Energy_consumption_dataset.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Energy Consumption data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Define feature columns (numeric columns for prediction)
        # Exclude categorical columns: DayOfWeek, Holiday, HVACUsage, LightingUsage
        # Exclude time columns: Month, Hour (these are temporal features, not prediction features)
        # Use numeric features: Temperature, Humidity, SquareFootage, Occupancy, RenewableEnergy
        # Optionally include EnergyConsumption if include_target=True
        
        numeric_features = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 
                           'RenewableEnergy', 'EnergyConsumption']
        
        # Exclude Month and Hour from features (they are temporal, not predictive features)
        exclude_cols = ['Month', 'Hour']
        
        # Find matching columns (case-insensitive)
        data_cols = []
        for feat in numeric_features:
            # Skip if in exclude list
            if feat in exclude_cols:
                continue
            # Try exact match first
            if feat in df.columns:
                data_cols.append(feat)
            else:
                # Try case-insensitive match
                matching = [c for c in df.columns if c.upper() == feat.upper()]
                if matching:
                    data_cols.append(matching[0])
        
        # If we couldn't find all expected columns, use all numeric columns (excluding Month and Hour)
        if len(data_cols) < len([f for f in numeric_features if f not in exclude_cols]):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove Month and Hour
            numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
            # Take the expected number of features
            expected_count = len([f for f in numeric_features if f not in exclude_cols])
            data_cols = numeric_cols[:expected_count] if len(numeric_cols) >= expected_count else numeric_cols
        
        # If include_target=False, exclude EnergyConsumption
        if not include_target:
            data_cols = [col for col in data_cols if 'EnergyConsumption' not in col.upper()]
        
        # Extract data
        data = df[data_cols].values
        
        # Convert to float, handling any string values
        data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').values
        
        # Enhanced data cleaning and preprocessing
        df_data = pd.DataFrame(data)
        
        # Remove outliers using IQR method for each column
        for col_idx in range(df_data.shape[1]):
            Q1 = df_data.iloc[:, col_idx].quantile(0.25)
            Q3 = df_data.iloc[:, col_idx].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            
            # Cap outliers instead of removing them to preserve data length
            df_data.iloc[:, col_idx] = df_data.iloc[:, col_idx].clip(lower_bound, upper_bound)
        
        # Fill NaN values
        df_data = df_data.ffill().bfill()
        df_data = df_data.interpolate(method='linear', limit_direction='both')
        
        # If still NaN, fill with median
        for col_idx in range(df_data.shape[1]):
            if df_data.iloc[:, col_idx].isna().any():
                median_val = df_data.iloc[:, col_idx].median()
                df_data.iloc[:, col_idx] = df_data.iloc[:, col_idx].fillna(median_val)
        
        data = df_data.values
        
        # Remove any remaining NaN rows
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]
        
        if len(data) == 0:
            raise ValueError("No valid data after cleaning NaN values")
        
        # Split train/test
        split_idx = int(len(data) * split_ratio)
        if mode == 'train':
            data = data[:split_idx]
        else:
            data = data[split_idx:]
        
        return data
    
    @classmethod
    def get_data(cls, data_type, config, mode='train', data_path=None, scaler=None, detrend_info_train=None, **kwargs):
        """
        Get real-world data with preprocessing
        
        Args:
            data_type: 'ETT', 'SST', 'AirQuality', 'NASA_Bearing', 'EnergyConsumption'
            config: Configuration object
            mode: 'train' or 'test'
            data_path: Path to data directory
            scaler: Pre-fitted scaler (for test data). If None, fit new scaler.
            detrend_info_train: Detrending info from training (for test data consistency)
            **kwargs: Additional arguments (e.g., dataset='ETTh1' for ETT)
        
        Returns:
            data: [N, seq_len+pred_len, D] tensor
            scaler: Fitted scaler (same as input if provided)
            detrend_info: Information about detrending (if applied)
        """
        if data_path is None:
            data_path = getattr(config, 'data_path', './data/real_data')
        
        # Load raw data
        if data_type == 'ETT':
            dataset = kwargs.get('dataset', 'ETTh1')
            raw_data = cls.load_ett_data(data_path, dataset, mode, split_ratio=0.7)
        elif data_type == 'SST':
            use_anomaly = kwargs.get('use_anomaly', False)
            raw_data = cls.load_sst_data(data_path, mode, split_ratio=0.7, use_anomaly=use_anomaly)
        elif data_type == 'AirQuality':
            raw_data = cls.load_air_quality_data(data_path, mode, split_ratio=0.7)
        elif data_type == 'NASA_Bearing':
            bearing_id = kwargs.get('bearing_id', 1)
            raw_data = cls.load_nasa_bearing_data(data_path, bearing_id, mode, split_ratio=0.7)
        elif data_type == 'EnergyConsumption':
            include_target = kwargs.get('include_target', True)
            raw_data = cls.load_energy_consumption_data(data_path, mode, split_ratio=0.7, include_target=include_target)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Enhanced preprocessing pipeline
        use_detrend = getattr(config, 'use_detrend', True)
        detrend_info = None
        if use_detrend:
            # Use more sophisticated detrending for different data types
            detrend_method = kwargs.get('detrend_method', 
                                       'polynomial' if data_type in ['AirQuality', 'SST', 'EnergyConsumption'] else 'linear')
            # For test data, use same detrending method but don't reuse trend
            # (detrending should be independent per dataset split)
            detrended_data, trend = cls.detrend(raw_data, method=detrend_method)
            detrend_info = {'trend': trend, 'method': detrend_method}
            processed_data = detrended_data
        else:
            processed_data = raw_data
        
        # Enhanced normalization based on data characteristics
        # IMPORTANT: For test data, use pre-fitted scaler from training!
        if scaler is None:
            # Train mode: fit new scaler
            normalization_method = kwargs.get('normalization_method', 
                                             'robust' if data_type in ['AirQuality', 'EnergyConsumption'] else 'standard')
            
            if normalization_method == 'robust':
                # RobustScaler is less sensitive to outliers (good for AirQuality data)
                scaler = RobustScaler()
            elif normalization_method == 'quantile':
                # Quantile normalization for heavily skewed data
                scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(processed_data)))
            elif normalization_method == 'minmax':
                # MinMax normalization for bounded data
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                # Standard normalization (default)
                scaler = StandardScaler()
            
            normalized_data = scaler.fit_transform(processed_data)
        else:
            # Test mode: use pre-fitted scaler (CRITICAL FIX!)
            normalized_data = scaler.transform(processed_data)
        
        # Optional smoothing for noisy data (useful for AirQuality, SST, and EnergyConsumption)
        apply_smoothing = kwargs.get('apply_smoothing', False)
        if apply_smoothing and data_type in ['AirQuality', 'SST', 'EnergyConsumption']:
            from scipy.ndimage import gaussian_filter1d
            sigma = kwargs.get('smoothing_sigma', 0.5)
            for d in range(normalized_data.shape[1]):
                normalized_data[:, d] = gaussian_filter1d(normalized_data[:, d], sigma=sigma)
        
        # Create sequences
        seq_len = config.seq_len
        pred_len = config.pred_len
        total_len = seq_len + pred_len
        
        num_samples = len(normalized_data) - total_len + 1
        if num_samples <= 0:
            raise ValueError(f"Data too short: {len(normalized_data)} < {total_len}")
        
        sequences = []
        for i in range(num_samples):
            seq = normalized_data[i:i+total_len]
            sequences.append(seq)
        
        data = np.array(sequences)
        data_tensor = torch.from_numpy(data).float()
        
        return data_tensor, scaler, detrend_info

