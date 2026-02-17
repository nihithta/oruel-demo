"""
Model Interface for Patented GPU Health Prediction Model
Handles loading and inference of the in-house model
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime


class GPUHealthModel:
    """Interface for GPU health prediction model"""
    
    def __init__(self, model_path: str = "placeholder.pt"):
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        
        # Try to load the model if it exists
        self._load_model()
    
    def _load_model(self):
        """Load the PyTorch model if available"""
        if os.path.exists(self.model_path):
            try:
                import torch
                self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
                self.model.eval()
                self.model_loaded = True
                print(f"Successfully loaded model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using fallback rule-based model")
                self.model_loaded = False
        else:
            print(f"Model file {self.model_path} not found. Using fallback rule-based model")
            self.model_loaded = False
    
    def predict(self, gpu_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run prediction on GPU metrics
        
        Input: Dictionary of DCGM metrics for a single GPU
        Output: Prediction results including RUL, health, actions, causes, etc.
        """
        if self.model_loaded:
            return self._model_predict(gpu_metrics)
        else:
            return self._fallback_predict(gpu_metrics)
    
    def _model_predict(self, gpu_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Use actual PyTorch model for prediction"""
        try:
            import torch
            
            # Prepare input features (adapt this to your model's expected input)
            features = self._extract_features(gpu_metrics)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Parse model output (adapt this to your model's output format)
            result = self._parse_model_output(output, gpu_metrics)
            
            return result
            
        except Exception as e:
            print(f"Error during model inference: {e}")
            return self._fallback_predict(gpu_metrics)
    
    def _extract_features(self, metrics: Dict[str, Any]) -> List[float]:
        """Extract feature vector from DCGM metrics"""
        # Define feature order (adapt to your model's training)
        feature_names = [
            'gpu_utilization',
            'gpu_temp_c',
            'memory_temp_c',
            'power_draw_w',
            'fb_memory_used_mb',
            'memory_utilization',
            'sm_clock_mhz',
            'memory_clock_mhz',
            'ecc_sbe_volatile_total',
            'ecc_dbe_volatile_total',
            'nvlink_crc_errors',
            'pcie_replay_counter',
            'clock_throttle_reasons',
            'xid_errors'
        ]
        
        features = []
        for feature in feature_names:
            value = metrics.get(feature, 0)
            # Handle any potential None or string values
            try:
                features.append(float(value))
            except (TypeError, ValueError):
                features.append(0.0)
        
        return features
    
    def _parse_model_output(self, output, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Parse model output tensor into structured prediction"""
        # This is a placeholder - adapt to your model's actual output format
        # Assuming output is a tensor with multiple values
        
        try:
            # Convert tensor to numpy
            output_np = output.cpu().numpy()[0]
            
            # Example output interpretation (adapt to your model)
            # Assuming: [RUL_hours, health_score, anomaly_score, ...]
            
            rul_hours = float(output_np[0]) if len(output_np) > 0 else 1000
            health_score = float(output_np[1]) if len(output_np) > 1 else 80
            anomaly_score = float(output_np[2]) if len(output_np) > 2 else 0.1
            
            # Build structured output
            result = {
                'rul_hours': max(0, rul_hours),
                'rul_days': max(0, rul_hours / 24),
                'health_score': np.clip(health_score, 0, 100),
                'anomaly_score': np.clip(anomaly_score, 0, 1),
                'status': self._classify_health(health_score),
                'actions': self._recommend_actions(health_score, rul_hours, metrics),
                'causes': self._identify_causes(metrics, anomaly_score),
                'risk_level': self._assess_risk(rul_hours, health_score),
                'predictions': {
                    'failure_probability_24h': self._calc_failure_prob(rul_hours, health_score, 24),
                    'failure_probability_7d': self._calc_failure_prob(rul_hours, health_score, 168),
                    'failure_probability_30d': self._calc_failure_prob(rul_hours, health_score, 720)
                },
                'metrics_percentiles': self._calculate_percentiles(metrics),
                'timestamp': datetime.now().isoformat(),
                'model_version': 'pytorch_model'
            }
            
            return result
            
        except Exception as e:
            print(f"Error parsing model output: {e}")
            return self._fallback_predict(metrics)
    
    def _fallback_predict(self, gpu_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based prediction when model is not available"""
        
        # Extract key metrics
        gpu_temp = gpu_metrics.get('gpu_temp_c', 0)
        gpu_util = gpu_metrics.get('gpu_utilization', 0)
        power = gpu_metrics.get('power_draw_w', 0)
        memory_util = gpu_metrics.get('memory_utilization', 0)
        ecc_sbe = gpu_metrics.get('ecc_sbe_volatile_total', 0)
        ecc_dbe = gpu_metrics.get('ecc_dbe_volatile_total', 0)
        xid_errors = gpu_metrics.get('xid_errors', 0)
        throttle = gpu_metrics.get('clock_throttle_reasons', 0)
        pcie_replay = gpu_metrics.get('pcie_replay_counter', 0)
        nvlink_errors = gpu_metrics.get('nvlink_crc_errors', 0) + gpu_metrics.get('nvlink_recovery_errors', 0)
        
        # Calculate health score
        health_score = 100.0
        issues = []
        
        # Temperature impact
        if gpu_temp > 85:
            health_score -= 30
            issues.append("Critical temperature")
        elif gpu_temp > 80:
            health_score -= 15
            issues.append("High temperature")
        elif gpu_temp > 75:
            health_score -= 5
            issues.append("Elevated temperature")
        
        # ECC errors (critical)
        if ecc_dbe > 0:
            health_score -= 40
            issues.append(f"Double-bit ECC errors: {ecc_dbe}")
        
        if ecc_sbe > 10:
            health_score -= 20
            issues.append(f"High single-bit ECC errors: {ecc_sbe}")
        elif ecc_sbe > 0:
            health_score -= 5
            issues.append(f"Single-bit ECC errors: {ecc_sbe}")
        
        # XID errors (hardware errors)
        if xid_errors > 0:
            health_score -= 35
            issues.append(f"XID hardware errors: {xid_errors}")
        
        # Throttling
        if throttle > 0:
            # Check if it's thermal throttling (more serious)
            thermal_throttle = (throttle & ((1 << 5) | (1 << 6))) > 0
            if thermal_throttle:
                health_score -= 15
                issues.append("Thermal throttling")
            else:
                health_score -= 5
                issues.append("Performance throttling")
        
        # PCIe issues
        if pcie_replay > 100:
            health_score -= 15
            issues.append(f"High PCIe replay count: {pcie_replay}")
        elif pcie_replay > 50:
            health_score -= 5
            issues.append(f"Elevated PCIe replay count: {pcie_replay}")
        
        # NVLink errors
        if nvlink_errors > 0:
            health_score -= 25
            issues.append(f"NVLink errors: {nvlink_errors}")
        
        # Low utilization (might indicate issues)
        if gpu_util < 5 and power > 100:
            health_score -= 10
            issues.append("Unexpectedly low utilization")
        
        health_score = max(0, min(100, health_score))
        
        # Calculate RUL based on health score and degradation rate
        # Simple heuristic: higher degradation = lower RUL
        if health_score > 90:
            rul_hours = 8760  # ~1 year
        elif health_score > 80:
            rul_hours = 4380  # ~6 months
        elif health_score > 70:
            rul_hours = 2190  # ~3 months
        elif health_score > 50:
            rul_hours = 720   # ~30 days
        elif health_score > 30:
            rul_hours = 168   # ~7 days
        else:
            rul_hours = 24    # ~1 day - critical
        
        # Adjust RUL based on specific issues
        if ecc_dbe > 0 or xid_errors > 0:
            rul_hours = min(rul_hours, 168)  # Max 7 days if critical errors
        
        # Calculate anomaly score
        anomaly_score = (100 - health_score) / 100.0
        
        # Build result
        result = {
            'rul_hours': rul_hours,
            'rul_days': rul_hours / 24,
            'health_score': health_score,
            'anomaly_score': anomaly_score,
            'status': self._classify_health(health_score),
            'actions': self._recommend_actions(health_score, rul_hours, gpu_metrics),
            'causes': issues if issues else ['No issues detected'],
            'risk_level': self._assess_risk(rul_hours, health_score),
            'predictions': {
                'failure_probability_24h': self._calc_failure_prob(rul_hours, health_score, 24),
                'failure_probability_7d': self._calc_failure_prob(rul_hours, health_score, 168),
                'failure_probability_30d': self._calc_failure_prob(rul_hours, health_score, 720)
            },
            'metrics_percentiles': self._calculate_percentiles(gpu_metrics),
            'timestamp': datetime.now().isoformat(),
            'model_version': 'rule_based_fallback'
        }
        
        return result
    
    def _classify_health(self, health_score: float) -> str:
        """Classify health status"""
        if health_score >= 90:
            return "Excellent"
        elif health_score >= 80:
            return "Good"
        elif health_score >= 70:
            return "Fair"
        elif health_score >= 50:
            return "Warning"
        elif health_score >= 30:
            return "Critical"
        else:
            return "Failing"
    
    def _assess_risk(self, rul_hours: float, health_score: float) -> str:
        """Assess overall risk level"""
        if rul_hours < 24 or health_score < 30:
            return "Critical"
        elif rul_hours < 168 or health_score < 50:
            return "High"
        elif rul_hours < 720 or health_score < 70:
            return "Moderate"
        else:
            return "Low"
    
    def _recommend_actions(self, health_score: float, rul_hours: float, metrics: Dict[str, Any]) -> List[str]:
        """Recommend maintenance actions"""
        actions = []
        
        temp = metrics.get('gpu_temp_c', 0)
        ecc_dbe = metrics.get('ecc_dbe_volatile_total', 0)
        xid = metrics.get('xid_errors', 0)
        
        if ecc_dbe > 0 or xid > 0:
            actions.append("URGENT: Schedule immediate replacement - hardware failure detected")
        
        if rul_hours < 24:
            actions.append("URGENT: Remove from production immediately")
        elif rul_hours < 168:
            actions.append("Schedule replacement within 7 days")
        elif rul_hours < 720:
            actions.append("Schedule maintenance check within 30 days")
        
        if temp > 85:
            actions.append("Check cooling system - critical temperature")
        elif temp > 80:
            actions.append("Verify cooling efficiency")
        
        if health_score < 50:
            actions.append("Reduce workload until maintenance")
        
        if metrics.get('throttle_reasons', 0) > 0:
            actions.append("Investigate throttling causes")
        
        if not actions:
            actions.append("Continue normal operation - no action needed")
        
        return actions
    
    def _identify_causes(self, metrics: Dict[str, Any], anomaly_score: float) -> List[str]:
        """Identify root causes of issues"""
        causes = []
        
        temp = metrics.get('gpu_temp_c', 0)
        ecc_sbe = metrics.get('ecc_sbe_volatile_total', 0)
        ecc_dbe = metrics.get('ecc_dbe_volatile_total', 0)
        xid = metrics.get('xid_errors', 0)
        throttle = metrics.get('clock_throttle_reasons', 0)
        
        if ecc_dbe > 0:
            causes.append("Memory hardware failure (uncorrectable errors)")
        
        if xid > 0:
            causes.append("GPU hardware malfunction (XID errors)")
        
        if ecc_sbe > 10:
            causes.append("Memory degradation (correctable errors increasing)")
        
        if temp > 85:
            causes.append("Thermal overload - cooling insufficient")
        elif temp > 80:
            causes.append("Thermal stress - check cooling")
        
        if throttle & (1 << 6):  # HW thermal throttling
            causes.append("Hardware thermal protection activated")
        
        if throttle & (1 << 2):  # SW power cap
            causes.append("Power limit throttling")
        
        if not causes:
            if anomaly_score > 0.5:
                causes.append("Anomalous behavior detected - cause under investigation")
            else:
                causes.append("Normal operation")
        
        return causes
    
    def _calc_failure_prob(self, rul_hours: float, health_score: float, horizon_hours: float) -> float:
        """Calculate failure probability within time horizon"""
        # Simple exponential model
        if rul_hours <= 0:
            return 1.0
        
        # Probability increases as we approach RUL
        time_factor = min(1.0, horizon_hours / rul_hours)
        health_factor = (100 - health_score) / 100
        
        # Combined probability
        prob = time_factor * 0.7 + health_factor * 0.3
        
        return min(1.0, max(0.0, prob))
    
    def _calculate_percentiles(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate where metrics fall in typical distributions
        (Would normally use historical data, using heuristics here)
        """
        # Define typical ranges for key metrics (based on H100 specs)
        ranges = {
            'gpu_temp_c': (30, 85),
            'power_draw_w': (50, 700),
            'gpu_utilization': (0, 100),
            'memory_utilization': (0, 100),
            'sm_clock_mhz': (500, 1980),
        }
        
        percentiles = {}
        for metric, (min_val, max_val) in ranges.items():
            value = metrics.get(metric, 0)
            # Simple linear percentile estimate
            if max_val > min_val:
                percentile = ((value - min_val) / (max_val - min_val)) * 100
                percentiles[metric] = max(0, min(100, percentile))
        
        return percentiles
    
    def batch_predict(self, gpu_metrics_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run predictions on multiple GPUs"""
        results = []
        for metrics in gpu_metrics_list:
            result = self.predict(metrics)
            results.append(result)
        return results


# Singleton instance
_model_instance = None

def get_model_instance(model_path: str = "placeholder.pt") -> GPUHealthModel:
    """Get or create model singleton"""
    global _model_instance
    if _model_instance is None:
        _model_instance = GPUHealthModel(model_path)
    return _model_instance


if __name__ == '__main__':
    # Test the model interface
    print("Testing Model Interface...")
    
    model = GPUHealthModel()
    
    # Test with sample metrics
    sample_metrics = {
        'gpu_utilization': 85.5,
        'gpu_temp_c': 78.2,
        'memory_temp_c': 72.0,
        'power_draw_w': 450.5,
        'fb_memory_used_mb': 65000,
        'memory_utilization': 81.25,
        'sm_clock_mhz': 1800,
        'memory_clock_mhz': 2400,
        'ecc_sbe_volatile_total': 2,
        'ecc_dbe_volatile_total': 0,
        'nvlink_crc_errors': 0,
        'pcie_replay_counter': 5,
        'clock_throttle_reasons': 0,
        'xid_errors': 0
    }
    
    print("\n=== Healthy GPU Test ===")
    result = model.predict(sample_metrics)
    print(json.dumps(result, indent=2))
    
    # Test with degraded GPU
    degraded_metrics = sample_metrics.copy()
    degraded_metrics['gpu_temp_c'] = 88
    degraded_metrics['ecc_sbe_volatile_total'] = 45
    degraded_metrics['ecc_dbe_volatile_total'] = 2
    degraded_metrics['clock_throttle_reasons'] = (1 << 6)  # HW thermal throttling
    
    print("\n=== Degraded GPU Test ===")
    result = model.predict(degraded_metrics)
    print(json.dumps(result, indent=2))
