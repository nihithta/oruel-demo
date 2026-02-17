"""
Advanced Analytics Module
Provides deep temporal, statistical, and physics-based analysis of GPU cluster metrics
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class TemporalAnalyzer:
    """Advanced time-series analysis for GPU metrics"""
    
    @staticmethod
    def detect_anomalies(data, contamination=0.1):
        """Detect anomalies using Isolation Forest"""
        if len(data) < 10:
            return []
        
        data_clean = np.array(data).reshape(-1, 1)
        data_clean = data_clean[~np.isnan(data_clean).any(axis=1)]
        
        if len(data_clean) < 10:
            return []
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(data_clean)
        
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        return anomaly_indices
    
    @staticmethod
    def calculate_trend(data, timestamps=None):
        """Calculate linear trend and acceleration"""
        data_clean = np.array(data)
        valid_mask = ~np.isnan(data_clean)
        
        if valid_mask.sum() < 3:
            return {'slope': 0, 'r_squared': 0, 'trend': 'stable', 'acceleration': 0}
        
        y = data_clean[valid_mask]
        x = np.arange(len(y))
        
        # Linear regression
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        
        slope = model.coef_[0]
        predictions = model.predict(x.reshape(-1, 1))
        r_squared = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2))
        
        # Calculate acceleration (second derivative)
        if len(y) > 5:
            acceleration = np.gradient(np.gradient(y))
            avg_acceleration = np.mean(acceleration)
        else:
            avg_acceleration = 0
        
        # Classify trend
        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'slope': float(slope),
            'r_squared': float(r_squared),
            'trend': trend,
            'acceleration': float(avg_acceleration)
        }
    
    @staticmethod
    def detect_periodicity(data):
        """Detect periodic patterns using FFT"""
        data_clean = np.array(data)
        valid_mask = ~np.isnan(data_clean)
        
        if valid_mask.sum() < 20:
            return {'has_periodicity': False, 'dominant_period': None, 'strength': 0}
        
        y = data_clean[valid_mask]
        
        # Detrend
        y_detrended = signal.detrend(y)
        
        # FFT
        fft = np.fft.fft(y_detrended)
        power = np.abs(fft) ** 2
        freqs = np.fft.fftfreq(len(y))
        
        # Find dominant frequency (excluding DC component)
        positive_freqs = freqs[1:len(freqs)//2]
        positive_power = power[1:len(power)//2]
        
        if len(positive_power) == 0:
            return {'has_periodicity': False, 'dominant_period': None, 'strength': 0}
        
        dominant_idx = np.argmax(positive_power)
        dominant_freq = positive_freqs[dominant_idx]
        dominant_power = positive_power[dominant_idx]
        
        # Calculate strength as ratio of dominant peak to average
        avg_power = np.mean(positive_power)
        strength = dominant_power / avg_power if avg_power > 0 else 0
        
        has_periodicity = strength > 3  # Threshold for significant periodicity
        
        return {
            'has_periodicity': bool(has_periodicity),
            'dominant_period': float(1 / abs(dominant_freq)) if abs(dominant_freq) > 0 else None,
            'strength': float(strength)
        }
    
    @staticmethod
    def calculate_volatility(data, window=10):
        """Calculate rolling volatility (standard deviation)"""
        data_clean = pd.Series(data).dropna()
        
        if len(data_clean) < window:
            return {'current': 0, 'average': 0, 'max': 0}
        
        rolling_std = data_clean.rolling(window=window).std()
        
        return {
            'current': float(rolling_std.iloc[-1]) if not pd.isna(rolling_std.iloc[-1]) else 0,
            'average': float(rolling_std.mean()),
            'max': float(rolling_std.max())
        }


class PhysicsAnalyzer:
    """Physics-based analysis of GPU behavior"""
    
    @staticmethod
    def analyze_thermal_dynamics(temp_data, power_data):
        """Analyze thermal behavior and heat dissipation"""
        temp = np.array(temp_data)
        power = np.array(power_data)
        
        valid_mask = ~(np.isnan(temp) | np.isnan(power))
        
        if valid_mask.sum() < 5:
            return {
                'thermal_efficiency': 0,
                'heat_dissipation_rate': 0,
                'thermal_correlation': 0,
                'thermal_stability': 'unknown'
            }
        
        temp_clean = temp[valid_mask]
        power_clean = power[valid_mask]
        
        # Thermal efficiency (temp per watt)
        if len(power_clean) > 0 and power_clean.mean() > 0:
            thermal_efficiency = temp_clean.mean() / power_clean.mean()
        else:
            thermal_efficiency = 0
        
        # Heat dissipation rate (temperature change per unit time)
        temp_diff = np.diff(temp_clean)
        heat_dissipation_rate = np.mean(np.abs(temp_diff))
        
        # Correlation between power and temperature
        if len(temp_clean) > 2:
            correlation = np.corrcoef(temp_clean, power_clean)[0, 1]
        else:
            correlation = 0
        
        # Thermal stability
        temp_std = np.std(temp_clean)
        if temp_std < 2:
            stability = 'excellent'
        elif temp_std < 5:
            stability = 'good'
        elif temp_std < 10:
            stability = 'moderate'
        else:
            stability = 'poor'
        
        return {
            'thermal_efficiency': float(thermal_efficiency),
            'heat_dissipation_rate': float(heat_dissipation_rate),
            'thermal_correlation': float(correlation) if not np.isnan(correlation) else 0,
            'thermal_stability': stability,
            'avg_temp': float(temp_clean.mean()),
            'temp_std': float(temp_std)
        }
    
    @staticmethod
    def analyze_power_efficiency(power_data, utilization_data):
        """Analyze power consumption efficiency"""
        power = np.array(power_data)
        util = np.array(utilization_data)
        
        valid_mask = ~(np.isnan(power) | np.isnan(util))
        
        if valid_mask.sum() < 5:
            return {
                'power_efficiency': 0,
                'idle_power': 0,
                'peak_power': 0,
                'efficiency_score': 0
            }
        
        power_clean = power[valid_mask]
        util_clean = util[valid_mask]
        
        # Idle power (power when utilization < 5%)
        idle_mask = util_clean < 5
        if idle_mask.sum() > 0:
            idle_power = power_clean[idle_mask].mean()
        else:
            idle_power = power_clean.min()
        
        # Peak power
        peak_power = power_clean.max()
        
        # Power efficiency (utilization per watt)
        active_mask = util_clean > 10
        if active_mask.sum() > 0:
            power_efficiency = util_clean[active_mask].mean() / power_clean[active_mask].mean()
        else:
            power_efficiency = 0
        
        # Efficiency score (0-100): how well power scales with utilization
        if len(util_clean) > 2 and np.std(util_clean) > 0:
            correlation = np.corrcoef(power_clean, util_clean)[0, 1]
            efficiency_score = max(0, min(100, correlation * 100))
        else:
            efficiency_score = 0
        
        return {
            'power_efficiency': float(power_efficiency),
            'idle_power': float(idle_power),
            'peak_power': float(peak_power),
            'efficiency_score': float(efficiency_score) if not np.isnan(efficiency_score) else 0
        }
    
    @staticmethod
    def analyze_workload_characteristics(util_data, memory_data):
        """Analyze workload patterns and compute intensity"""
        util = np.array(util_data)
        memory = np.array(memory_data)
        
        valid_mask = ~(np.isnan(util) | np.isnan(memory))
        
        if valid_mask.sum() < 5:
            return {
                'workload_type': 'unknown',
                'compute_intensity': 0,
                'memory_intensity': 0,
                'balance_score': 0
            }
        
        util_clean = util[valid_mask]
        memory_clean = memory[valid_mask]
        
        avg_util = util_clean.mean()
        avg_memory = memory_clean.mean()
        
        # Classify workload type
        if avg_util > 70 and avg_memory > 70:
            workload_type = 'compute_intensive'
        elif avg_memory > 70 and avg_util < 50:
            workload_type = 'memory_intensive'
        elif avg_util > 50 and avg_memory < 50:
            workload_type = 'compute_bound'
        elif avg_util < 30 and avg_memory < 30:
            workload_type = 'idle'
        else:
            workload_type = 'balanced'
        
        # Compute intensity
        compute_intensity = avg_util
        memory_intensity = avg_memory
        
        # Balance score: how well compute and memory are utilized together
        if avg_util > 10 and avg_memory > 10:
            balance_score = 100 - abs(avg_util - avg_memory)
        else:
            balance_score = 0
        
        return {
            'workload_type': workload_type,
            'compute_intensity': float(compute_intensity),
            'memory_intensity': float(memory_intensity),
            'balance_score': float(balance_score)
        }


class StatisticalAnalyzer:
    """Advanced statistical analysis"""
    
    @staticmethod
    def calculate_distribution_stats(data):
        """Calculate detailed distribution statistics"""
        data_clean = np.array(data)
        data_clean = data_clean[~np.isnan(data_clean)]
        
        if len(data_clean) < 3:
            return {}
        
        return {
            'mean': float(np.mean(data_clean)),
            'median': float(np.median(data_clean)),
            'std': float(np.std(data_clean)),
            'variance': float(np.var(data_clean)),
            'skewness': float(stats.skew(data_clean)),
            'kurtosis': float(stats.kurtosis(data_clean)),
            'min': float(np.min(data_clean)),
            'max': float(np.max(data_clean)),
            'q25': float(np.percentile(data_clean, 25)),
            'q75': float(np.percentile(data_clean, 75)),
            'iqr': float(np.percentile(data_clean, 75) - np.percentile(data_clean, 25))
        }
    
    @staticmethod
    def detect_outliers(data, method='iqr'):
        """Detect outliers using IQR or Z-score method"""
        data_clean = np.array(data)
        data_clean = data_clean[~np.isnan(data_clean)]
        
        if len(data_clean) < 5:
            return {'outlier_indices': [], 'outlier_count': 0, 'outlier_percentage': 0}
        
        if method == 'iqr':
            q25 = np.percentile(data_clean, 25)
            q75 = np.percentile(data_clean, 75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = (data_clean < lower_bound) | (data_clean > upper_bound)
        else:  # z-score
            z_scores = np.abs(stats.zscore(data_clean))
            outliers = z_scores > 3
        
        outlier_indices = np.where(outliers)[0].tolist()
        
        return {
            'outlier_indices': outlier_indices,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': float(len(outlier_indices) / len(data_clean) * 100)
        }
    
    @staticmethod
    def calculate_correlations(df, metrics):
        """Calculate correlation matrix for multiple metrics"""
        correlation_data = {}
        
        for metric in metrics:
            if metric in df.columns:
                correlation_data[metric] = df[metric].fillna(0).values
        
        if len(correlation_data) < 2:
            return {}
        
        # Create DataFrame and calculate correlations
        corr_df = pd.DataFrame(correlation_data)
        corr_matrix = corr_df.corr()
        
        return corr_matrix.to_dict()


class PredictiveAnalyzer:
    """Predictive analytics and failure prediction"""
    
    @staticmethod
    def predict_failure_risk(health_history, trend_data):
        """Predict GPU failure risk based on historical patterns"""
        
        # Simple rule-based + trend-based prediction
        risk_score = 0
        risk_factors = []
        
        # Historical health trend
        if trend_data.get('trend') == 'decreasing' and trend_data.get('slope', 0) < -0.5:
            risk_score += 30
            risk_factors.append("Declining health trend")
        
        # Acceleration (rapid deterioration)
        if trend_data.get('acceleration', 0) < -0.1:
            risk_score += 20
            risk_factors.append("Accelerating degradation")
        
        # Recent health scores
        if len(health_history) > 0:
            recent_avg = np.mean(health_history[-5:])
            if recent_avg < 50:
                risk_score += 40
                risk_factors.append("Consistently low health scores")
            elif recent_avg < 70:
                risk_score += 20
                risk_factors.append("Below-average health")
        
        # Risk classification
        if risk_score > 70:
            risk_level = 'critical'
        elif risk_score > 40:
            risk_level = 'high'
        elif risk_score > 20:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': min(100, risk_score),
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }
    
    @staticmethod
    def forecast_metric(data, steps=10):
        """Simple linear forecast for metric values"""
        data_clean = np.array(data)
        valid_mask = ~np.isnan(data_clean)
        
        if valid_mask.sum() < 5:
            return {'forecast': [], 'confidence': 0}
        
        y = data_clean[valid_mask]
        x = np.arange(len(y))
        
        # Linear regression
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)
        
        # Forecast
        future_x = np.arange(len(y), len(y) + steps)
        forecast = model.predict(future_x.reshape(-1, 1))
        
        # Calculate confidence (inverse of residual variance)
        predictions = model.predict(x.reshape(-1, 1))
        residuals = y - predictions
        confidence = max(0, 100 - np.std(residuals) * 10)
        
        return {
            'forecast': forecast.tolist(),
            'confidence': float(confidence)
        }


class AdvancedModels:
    """Additional sophisticated models for deeper analysis"""
    
    @staticmethod
    def detect_change_points(data, threshold=2.0):
        """Detect significant change points in time series using cumulative sum"""
        data_clean = np.array(data)
        valid_mask = ~np.isnan(data_clean)
        
        if valid_mask.sum() < 10:
            return {'change_points': [], 'count': 0}
        
        y = data_clean[valid_mask]
        
        # Standardize
        mean = np.mean(y)
        std = np.std(y)
        if std == 0:
            return {'change_points': [], 'count': 0}
        
        standardized = (y - mean) / std
        
        # CUSUM algorithm
        cusum_pos = np.zeros(len(standardized))
        cusum_neg = np.zeros(len(standardized))
        
        for i in range(1, len(standardized)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + standardized[i])
            cusum_neg[i] = min(0, cusum_neg[i-1] + standardized[i])
        
        # Detect change points
        change_points = []
        for i in range(len(cusum_pos)):
            if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                change_points.append(i)
        
        return {
            'change_points': change_points,
            'count': len(change_points),
            'cusum_positive': cusum_pos.tolist(),
            'cusum_negative': cusum_neg.tolist()
        }
    
    @staticmethod
    def calculate_entropy(data):
        """Calculate Shannon entropy of the distribution"""
        data_clean = np.array(data)
        valid_mask = ~np.isnan(data_clean)
        
        if valid_mask.sum() < 5:
            return 0
        
        y = data_clean[valid_mask]
        
        # Create histogram
        hist, _ = np.histogram(y, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return float(entropy)
    
    @staticmethod
    def calculate_hurst_exponent(data):
        """Calculate Hurst exponent to measure long-term memory"""
        data_clean = np.array(data)
        valid_mask = ~np.isnan(data_clean)
        
        if valid_mask.sum() < 20:
            return 0.5
        
        y = data_clean[valid_mask]
        
        # Remove trend
        y = signal.detrend(y)
        
        # Calculate R/S statistic
        lags = range(2, min(len(y)//2, 20))
        rs_values = []
        
        for lag in lags:
            # Split into segments
            segments = len(y) // lag
            if segments == 0:
                continue
            
            rs_seg = []
            for i in range(segments):
                seg = y[i*lag:(i+1)*lag]
                if len(seg) == 0:
                    continue
                
                # Mean and cumulative deviation
                mean_seg = np.mean(seg)
                cum_dev = np.cumsum(seg - mean_seg)
                
                # Range
                r = np.max(cum_dev) - np.min(cum_dev)
                
                # Standard deviation
                s = np.std(seg)
                
                if s > 0:
                    rs_seg.append(r / s)
            
            if len(rs_seg) > 0:
                rs_values.append(np.mean(rs_seg))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Linear regression on log-log plot
        log_lags = np.log(list(lags[:len(rs_values)]))
        log_rs = np.log(rs_values)
        
        if len(log_lags) > 0 and np.std(log_lags) > 0:
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            return float(np.clip(hurst, 0, 1))
        
        return 0.5
    
    @staticmethod
    def calculate_degradation_rate(data, timestamps=None):
        """Calculate rate of performance degradation"""
        data_clean = np.array(data)
        valid_mask = ~np.isnan(data_clean)
        
        if valid_mask.sum() < 10:
            return {'rate': 0, 'projected_failure': None, 'severity': 'unknown'}
        
        y = data_clean[valid_mask]
        x = np.arange(len(y))
        
        # Fit exponential decay model: y = a * exp(-b * x) + c
        from scipy.optimize import curve_fit
        
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        try:
            params, _ = curve_fit(exp_decay, x, y, p0=[y[0]-y[-1], 0.01, y[-1]], maxfev=5000)
            a, b, c = params
            
            # Degradation rate
            rate = float(b)
            
            # Project when it might reach critical threshold (e.g., 50% of initial)
            critical_threshold = y[0] * 0.5
            if rate > 0 and a > 0:
                projected_steps = -np.log((critical_threshold - c) / a) / rate if (critical_threshold - c) / a > 0 else None
            else:
                projected_steps = None
            
            # Severity classification
            if rate > 0.1:
                severity = 'critical'
            elif rate > 0.01:
                severity = 'high'
            elif rate > 0.001:
                severity = 'moderate'
            else:
                severity = 'low'
            
            return {
                'rate': rate,
                'projected_failure_steps': float(projected_steps) if projected_steps else None,
                'severity': severity,
                'model_params': {'a': float(a), 'b': float(b), 'c': float(c)}
            }
        except:
            # Fallback to linear
            slope = np.polyfit(x, y, 1)[0]
            return {
                'rate': float(abs(slope)),
                'projected_failure_steps': None,
                'severity': 'unknown',
                'model_params': {}
            }
    
    @staticmethod
    def calculate_stability_index(data):
        """Calculate overall stability index (0-100)"""
        data_clean = np.array(data)
        valid_mask = ~np.isnan(data_clean)
        
        if valid_mask.sum() < 5:
            return 50
        
        y = data_clean[valid_mask]
        
        # Multiple stability metrics
        stability_score = 100
        
        # 1. Coefficient of variation
        cv = np.std(y) / np.mean(y) if np.mean(y) > 0 else 0
        if cv > 0.5:
            stability_score -= 30
        elif cv > 0.3:
            stability_score -= 15
        
        # 2. Number of trend reversals
        diffs = np.diff(y)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        reversal_rate = sign_changes / len(y)
        if reversal_rate > 0.3:
            stability_score -= 20
        elif reversal_rate > 0.2:
            stability_score -= 10
        
        # 3. Range as percentage of mean
        range_pct = (np.max(y) - np.min(y)) / np.mean(y) if np.mean(y) > 0 else 0
        if range_pct > 1.0:
            stability_score -= 20
        elif range_pct > 0.5:
            stability_score -= 10
        
        return max(0, min(100, stability_score))


class ClusterAnalyzer:
    """Cross-GPU and cluster-wide analysis"""
    
    @staticmethod
    def find_similar_gpus(gpu_data_list, n_clusters=3):
        """Cluster GPUs by behavior similarity using K-means"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        if len(gpu_data_list) < n_clusters:
            return {'clusters': [], 'labels': []}
        
        # Extract features for clustering
        features = []
        gpu_ids = []
        
        for gpu_id, data in enumerate(gpu_data_list):
            if len(data) > 0:
                features.append([
                    np.mean(data),
                    np.std(data),
                    np.max(data),
                    np.min(data)
                ])
                gpu_ids.append(gpu_id)
        
        if len(features) < n_clusters:
            return {'clusters': [], 'labels': []}
        
        # Standardize and cluster
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # Group by cluster
        clusters = {}
        for gpu_id, label in zip(gpu_ids, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(gpu_id)
        
        return {
            'clusters': clusters,
            'labels': labels.tolist(),
            'gpu_ids': gpu_ids
        }
    
    @staticmethod
    def calculate_cluster_imbalance(gpu_data_list):
        """Measure load imbalance across GPUs"""
        utilizations = []
        
        for data in gpu_data_list:
            if len(data) > 0:
                utilizations.append(np.mean(data))
        
        if len(utilizations) < 2:
            return {'imbalance_score': 0, 'coefficient_of_variation': 0}
        
        # Coefficient of variation
        cv = np.std(utilizations) / np.mean(utilizations) if np.mean(utilizations) > 0 else 0
        
        # Imbalance score (0-100, higher is worse)
        imbalance_score = min(100, cv * 100)
        
        return {
            'imbalance_score': float(imbalance_score),
            'coefficient_of_variation': float(cv),
            'utilization_range': float(np.max(utilizations) - np.min(utilizations)),
            'mean_utilization': float(np.mean(utilizations)),
            'std_utilization': float(np.std(utilizations))
        }
    
    @staticmethod
    def detect_synchronization(gpu_data1, gpu_data2):
        """Detect if two GPUs are synchronized in behavior"""
        if len(gpu_data1) < 10 or len(gpu_data2) < 10:
            return {'synchronized': False, 'lag': 0, 'correlation': 0}
        
        # Cross-correlation to find lag
        correlation = signal.correlate(gpu_data1, gpu_data2, mode='same')
        lag = np.argmax(correlation) - len(gpu_data1)//2
        max_corr = np.max(correlation) / (np.std(gpu_data1) * np.std(gpu_data2) * len(gpu_data1))
        
        synchronized = abs(max_corr) > 0.7 and abs(lag) < 5
        
        return {
            'synchronized': bool(synchronized),
            'lag': int(lag),
            'correlation': float(max_corr)
        }


class AdvancedAnalyzer:
    """Main interface for all advanced analytics"""
    
    def __init__(self):
        self.temporal = TemporalAnalyzer()
        self.physics = PhysicsAnalyzer()
        self.statistical = StatisticalAnalyzer()
        self.predictive = PredictiveAnalyzer()
        self.advanced = AdvancedModels()
        self.cluster = ClusterAnalyzer()
    
    def analyze_gpu_comprehensive(self, df, gpu_id):
        """Perform comprehensive analysis on a single GPU"""
        gpu_prefix = f'system.gpu.{gpu_id}'
        
        results = {
            'gpu_id': gpu_id,
            'temporal': {},
            'physics': {},
            'statistical': {},
            'predictive': {}
        }
        
        # Extract metrics
        util_col = f'{gpu_prefix}.gpu'
        temp_col = f'{gpu_prefix}.temp'
        power_col = f'{gpu_prefix}.powerWatts'
        memory_col = f'{gpu_prefix}.memoryAllocated'
        
        util_data = df[util_col].fillna(0).values if util_col in df.columns else []
        temp_data = df[temp_col].fillna(0).values if temp_col in df.columns else []
        power_data = df[power_col].fillna(0).values if power_col in df.columns else []
        memory_data = df[memory_col].fillna(0).values if memory_col in df.columns else []
        
        # Temporal analysis
        if len(util_data) > 0:
            results['temporal']['utilization'] = {
                'trend': self.temporal.calculate_trend(util_data),
                'anomalies': self.temporal.detect_anomalies(util_data),
                'periodicity': self.temporal.detect_periodicity(util_data),
                'volatility': self.temporal.calculate_volatility(util_data)
            }
        
        if len(temp_data) > 0:
            results['temporal']['temperature'] = {
                'trend': self.temporal.calculate_trend(temp_data),
                'anomalies': self.temporal.detect_anomalies(temp_data),
                'volatility': self.temporal.calculate_volatility(temp_data)
            }
        
        # Physics analysis
        if len(temp_data) > 0 and len(power_data) > 0:
            results['physics']['thermal'] = self.physics.analyze_thermal_dynamics(
                temp_data, power_data
            )
        
        if len(power_data) > 0 and len(util_data) > 0:
            results['physics']['power'] = self.physics.analyze_power_efficiency(
                power_data, util_data
            )
        
        if len(util_data) > 0 and len(memory_data) > 0:
            results['physics']['workload'] = self.physics.analyze_workload_characteristics(
                util_data, memory_data
            )
        
        # Statistical analysis
        if len(util_data) > 0:
            results['statistical']['utilization'] = self.statistical.calculate_distribution_stats(util_data)
            results['statistical']['utilization_outliers'] = self.statistical.detect_outliers(util_data)
        if len(temp_data) > 0:
            results['statistical']['temperature'] = self.statistical.calculate_distribution_stats(temp_data)
            results['statistical']['temperature_outliers'] = self.statistical.detect_outliers(temp_data)
        if len(power_data) > 0:
            results['statistical']['power'] = self.statistical.calculate_distribution_stats(power_data)
        
        # Advanced models
        results['advanced'] = {}
        if len(util_data) > 0:
            results['advanced']['utilization'] = {
                'change_points': self.advanced.detect_change_points(util_data),
                'entropy': self.advanced.calculate_entropy(util_data),
                'hurst_exponent': self.advanced.calculate_hurst_exponent(util_data),
                'stability_index': self.advanced.calculate_stability_index(util_data),
                'degradation': self.advanced.calculate_degradation_rate(util_data)
            }
        
        if len(temp_data) > 0:
            results['advanced']['temperature'] = {
                'change_points': self.advanced.detect_change_points(temp_data),
                'entropy': self.advanced.calculate_entropy(temp_data),
                'stability_index': self.advanced.calculate_stability_index(temp_data),
                'degradation': self.advanced.calculate_degradation_rate(temp_data)
            }
        
        if len(power_data) > 0:
            results['advanced']['power'] = {
                'stability_index': self.advanced.calculate_stability_index(power_data),
                'entropy': self.advanced.calculate_entropy(power_data)
            }
        
        return results
