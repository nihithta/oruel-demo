"""
DCGM Prometheus Metrics Ingestion Layer
Scrapes and processes NVIDIA DCGM metrics from Prometheus endpoints
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd


class DCGMPrometheusClient:
    """Client to scrape DCGM metrics from Prometheus"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.query_url = f"{prometheus_url}/api/v1/query"
        self.range_query_url = f"{prometheus_url}/api/v1/query_range"
        
        # DCGM metric mappings
        self.dcgm_metrics = {
            # GPU Utilization
            'DCGM_FI_DEV_GPU_UTIL': 'gpu_utilization',
            'DCGM_FI_DEV_MEM_COPY_UTIL': 'memory_copy_utilization',
            'DCGM_FI_DEV_ENC_UTIL': 'encoder_utilization',
            'DCGM_FI_DEV_DEC_UTIL': 'decoder_utilization',
            
            # Temperature
            'DCGM_FI_DEV_GPU_TEMP': 'gpu_temp_c',
            'DCGM_FI_DEV_MEMORY_TEMP': 'memory_temp_c',
            
            # Power
            'DCGM_FI_DEV_POWER_USAGE': 'power_draw_w',
            'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION': 'total_energy_consumption',
            
            # Memory
            'DCGM_FI_DEV_FB_FREE': 'fb_memory_free_mb',
            'DCGM_FI_DEV_FB_USED': 'fb_memory_used_mb',
            'DCGM_FI_DEV_FB_TOTAL': 'fb_memory_total_mb',
            
            # Clock Frequencies
            'DCGM_FI_DEV_SM_CLOCK': 'sm_clock_mhz',
            'DCGM_FI_DEV_MEM_CLOCK': 'memory_clock_mhz',
            
            # PCIe
            'DCGM_FI_DEV_PCIE_TX_THROUGHPUT': 'pcie_tx_throughput_kbps',
            'DCGM_FI_DEV_PCIE_RX_THROUGHPUT': 'pcie_rx_throughput_kbps',
            'DCGM_FI_DEV_PCIE_REPLAY_COUNTER': 'pcie_replay_counter',
            
            # NVLink
            'DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL': 'nvlink_bandwidth_total',
            'DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL': 'nvlink_crc_errors',
            'DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL': 'nvlink_recovery_errors',
            
            # ECC/Memory Errors
            'DCGM_FI_DEV_ECC_SBE_VOL_TOTAL': 'ecc_sbe_volatile_total',
            'DCGM_FI_DEV_ECC_DBE_VOL_TOTAL': 'ecc_dbe_volatile_total',
            'DCGM_FI_DEV_ECC_SBE_AGG_TOTAL': 'ecc_sbe_aggregate_total',
            'DCGM_FI_DEV_ECC_DBE_AGG_TOTAL': 'ecc_dbe_aggregate_total',
            
            # Throttling
            'DCGM_FI_DEV_CLOCK_THROTTLE_REASONS': 'clock_throttle_reasons',
            
            # XID Errors
            'DCGM_FI_DEV_XID_ERRORS': 'xid_errors',
            
            # Compute Processes
            'DCGM_FI_DEV_COMPUTE_PIDS': 'compute_pids',
            'DCGM_FI_DEV_GRAPHICS_PIDS': 'graphics_pids',
        }
    
    def query_instant(self, metric: str) -> Dict[str, Any]:
        """Query instant value of a metric"""
        try:
            response = requests.get(
                self.query_url,
                params={'query': metric},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error querying {metric}: {e}")
            return {'status': 'error', 'data': {'result': []}}
    
    def query_range(self, metric: str, start: int, end: int, step: str = '15s') -> Dict[str, Any]:
        """Query range of values for a metric"""
        try:
            response = requests.get(
                self.range_query_url,
                params={
                    'query': metric,
                    'start': start,
                    'end': end,
                    'step': step
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error querying range for {metric}: {e}")
            return {'status': 'error', 'data': {'result': []}}
    
    def get_all_gpu_metrics(self) -> pd.DataFrame:
        """Fetch all DCGM metrics for all GPUs"""
        all_metrics = []
        
        for dcgm_metric, friendly_name in self.dcgm_metrics.items():
            result = self.query_instant(dcgm_metric)
            
            if result['status'] == 'success':
                for series in result['data']['result']:
                    labels = series.get('metric', {})
                    value = series.get('value', [None, None])
                    
                    metric_data = {
                        'timestamp': datetime.fromtimestamp(value[0]) if value[0] else datetime.now(),
                        'metric_name': friendly_name,
                        'value': float(value[1]) if value[1] else 0.0,
                        'gpu_id': labels.get('gpu', labels.get('UUID', 'unknown')),
                        'hostname': labels.get('instance', labels.get('hostname', 'unknown')),
                        'gpu_model': labels.get('modelName', 'unknown'),
                        'gpu_uuid': labels.get('UUID', 'unknown')
                    }
                    all_metrics.append(metric_data)
        
        if not all_metrics:
            return pd.DataFrame()
        
        return pd.DataFrame(all_metrics)
    
    def get_gpu_health_snapshot(self) -> Dict[str, Any]:
        """Get current health snapshot of all GPUs"""
        df = self.get_all_gpu_metrics()
        
        if df.empty:
            return {'gpus': [], 'timestamp': datetime.now().isoformat()}
        
        # Pivot to get one row per GPU
        pivot_df = df.pivot_table(
            index=['hostname', 'gpu_id', 'gpu_model', 'gpu_uuid'],
            columns='metric_name',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        gpus = []
        for _, row in pivot_df.iterrows():
            gpu_data = {
                'hostname': row['hostname'],
                'gpu_id': row['gpu_id'],
                'gpu_model': row['gpu_model'],
                'gpu_uuid': row['gpu_uuid'],
                'metrics': {col: row[col] for col in pivot_df.columns if col not in ['hostname', 'gpu_id', 'gpu_model', 'gpu_uuid']}
            }
            gpus.append(gpu_data)
        
        return {
            'gpus': gpus,
            'timestamp': datetime.now().isoformat(),
            'total_gpus': len(gpus)
        }
    
    def get_topology_info(self) -> Dict[str, Any]:
        """
        Get GPU topology information from DCGM
        This includes NVLink connections, PCIe topology, etc.
        """
        # Query NVLink connectivity
        nvlink_query = 'DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL'
        result = self.query_instant(nvlink_query)
        
        topology = {
            'nodes': [],
            'links': [],
            'timestamp': datetime.now().isoformat()
        }
        
        if result['status'] == 'success':
            # Parse NVLink connections
            for series in result['data']['result']:
                labels = series.get('metric', {})
                
                gpu_id = labels.get('gpu', 'unknown')
                hostname = labels.get('instance', 'unknown')
                
                # Look for link information in labels
                link = labels.get('link', None)
                
                if link is not None:
                    topology['links'].append({
                        'source_gpu': gpu_id,
                        'source_host': hostname,
                        'link_id': link,
                        'bandwidth': float(series.get('value', [None, 0])[1])
                    })
        
        return topology
    
    def get_historical_metrics(self, hours: int = 24) -> pd.DataFrame:
        """Get historical metrics for the past N hours"""
        end_time = int(time.time())
        start_time = end_time - (hours * 3600)
        
        all_data = []
        
        for dcgm_metric, friendly_name in self.dcgm_metrics.items():
            result = self.query_range(dcgm_metric, start_time, end_time, step='1m')
            
            if result['status'] == 'success':
                for series in result['data']['result']:
                    labels = series.get('metric', {})
                    values = series.get('values', [])
                    
                    for timestamp, value in values:
                        metric_data = {
                            'timestamp': datetime.fromtimestamp(timestamp),
                            'metric_name': friendly_name,
                            'value': float(value) if value else 0.0,
                            'gpu_id': labels.get('gpu', labels.get('UUID', 'unknown')),
                            'hostname': labels.get('instance', labels.get('hostname', 'unknown'))
                        }
                        all_data.append(metric_data)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.DataFrame(all_data)


class DCGMDataProcessor:
    """Process and transform DCGM data for analysis"""
    
    @staticmethod
    def calculate_memory_utilization(fb_used_mb: float, fb_total_mb: float) -> float:
        """Calculate memory utilization percentage"""
        if fb_total_mb > 0:
            return (fb_used_mb / fb_total_mb) * 100
        return 0.0
    
    @staticmethod
    def detect_throttling(throttle_reasons: int) -> Dict[str, bool]:
        """Decode throttle reasons bitmask"""
        # DCGM throttle reasons bitmask
        reasons = {
            'gpu_idle': bool(throttle_reasons & (1 << 0)),
            'clocks_setting': bool(throttle_reasons & (1 << 1)),
            'sw_power_cap': bool(throttle_reasons & (1 << 2)),
            'hw_slowdown': bool(throttle_reasons & (1 << 3)),
            'sync_boost': bool(throttle_reasons & (1 << 4)),
            'sw_thermal': bool(throttle_reasons & (1 << 5)),
            'hw_thermal': bool(throttle_reasons & (1 << 6)),
            'hw_power_brake': bool(throttle_reasons & (1 << 7)),
            'display_clock': bool(throttle_reasons & (1 << 8))
        }
        return reasons
    
    @staticmethod
    def assess_gpu_health(metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess GPU health based on DCGM metrics"""
        health_score = 100
        issues = []
        
        # Temperature checks
        gpu_temp = metrics.get('gpu_temp_c', 0)
        if gpu_temp > 85:
            health_score -= 30
            issues.append(f"Critical GPU temperature: {gpu_temp}°C")
        elif gpu_temp > 80:
            health_score -= 15
            issues.append(f"High GPU temperature: {gpu_temp}°C")
        
        # Memory errors
        ecc_dbe = metrics.get('ecc_dbe_volatile_total', 0)
        ecc_sbe = metrics.get('ecc_sbe_volatile_total', 0)
        
        if ecc_dbe > 0:
            health_score -= 40
            issues.append(f"Double-bit ECC errors detected: {ecc_dbe}")
        
        if ecc_sbe > 10:
            health_score -= 20
            issues.append(f"High single-bit ECC errors: {ecc_sbe}")
        
        # XID errors
        xid_errors = metrics.get('xid_errors', 0)
        if xid_errors > 0:
            health_score -= 35
            issues.append(f"XID errors detected: {xid_errors}")
        
        # PCIe replay counter
        pcie_replay = metrics.get('pcie_replay_counter', 0)
        if pcie_replay > 100:
            health_score -= 15
            issues.append(f"High PCIe replay counter: {pcie_replay}")
        
        # NVLink errors
        nvlink_errors = metrics.get('nvlink_crc_errors', 0) + metrics.get('nvlink_recovery_errors', 0)
        if nvlink_errors > 0:
            health_score -= 25
            issues.append(f"NVLink errors detected: {nvlink_errors}")
        
        # Throttling
        throttle_reasons = int(metrics.get('clock_throttle_reasons', 0))
        if throttle_reasons > 0:
            throttle_info = DCGMDataProcessor.detect_throttling(throttle_reasons)
            active_throttles = [k for k, v in throttle_info.items() if v and k not in ['gpu_idle', 'clocks_setting']]
            if active_throttles:
                health_score -= 10
                issues.append(f"Throttling active: {', '.join(active_throttles)}")
        
        health_score = max(0, health_score)
        
        status = 'Healthy' if health_score > 80 else 'Warning' if health_score > 50 else 'Critical'
        
        return {
            'health_score': health_score,
            'status': status,
            'issues': issues
        }


if __name__ == '__main__':
    # Test the ingestion layer
    client = DCGMPrometheusClient()
    
    print("Fetching GPU health snapshot...")
    snapshot = client.get_gpu_health_snapshot()
    print(json.dumps(snapshot, indent=2, default=str))
    
    print("\nFetching topology info...")
    topology = client.get_topology_info()
    print(json.dumps(topology, indent=2))
