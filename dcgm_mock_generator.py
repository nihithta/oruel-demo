"""
Mock DCGM Data Generator
Generates realistic DCGM metrics for testing without actual GPUs
Can run as standalone Prometheus exporter or generate CSV files
"""

import random
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
from flask import Flask, Response


class MockGPU:
    """Simulates a single GPU with realistic behavior"""
    
    def __init__(self, gpu_id: str, hostname: str, model: str = "NVIDIA H100", cluster_id: int = 0):
        self.gpu_id = gpu_id
        self.hostname = hostname
        self.model = model
        self.cluster_id = cluster_id
        self.uuid = f"GPU-{gpu_id}-{random.randint(1000, 9999)}"
        
        # Base characteristics
        self.max_temp = 85
        self.max_power = 700  # H100 TDP
        self.total_memory = 80000  # 80GB for H100
        
        # Simulation state
        self.time_offset = random.uniform(0, 100)
        self.workload_pattern = random.choice(['stable', 'bursty', 'degrading', 'idle'])
        self.health_degradation = 0  # Slowly increases over time
        self.has_ecc_errors = random.random() < 0.05  # 5% chance of ECC errors
        self.has_throttling = random.random() < 0.1  # 10% chance of throttling
        self.nvlink_partner = None  # Set externally for NVLink simulation
        
    def get_utilization(self, t: float) -> float:
        """Get GPU utilization based on workload pattern"""
        if self.workload_pattern == 'idle':
            return random.uniform(0, 15)
        elif self.workload_pattern == 'stable':
            base = 75 + 10 * np.sin(0.1 * (t + self.time_offset))
            return max(0, min(100, base + random.uniform(-5, 5)))
        elif self.workload_pattern == 'bursty':
            if np.sin(0.5 * (t + self.time_offset)) > 0.5:
                return random.uniform(80, 100)
            else:
                return random.uniform(10, 30)
        elif self.workload_pattern == 'degrading':
            base = 90 - (self.health_degradation * 10)
            return max(20, base + random.uniform(-10, 10))
        return 50
    
    def get_temperature(self, utilization: float, t: float) -> float:
        """Get temperature based on utilization and thermal dynamics"""
        # Temperature follows utilization with some lag and noise
        base_temp = 30 + (utilization / 100) * 55
        thermal_noise = 3 * np.sin(0.3 * (t + self.time_offset))
        
        if self.has_throttling:
            base_temp += 10  # Throttling GPUs run hotter
        
        return max(30, min(self.max_temp + 5, base_temp + thermal_noise + random.uniform(-2, 2)))
    
    def get_power(self, utilization: float) -> float:
        """Get power draw based on utilization"""
        idle_power = 50
        active_power = self.max_power * (utilization / 100)
        return idle_power + active_power + random.uniform(-20, 20)
    
    def get_memory_used(self, utilization: float) -> float:
        """Get memory usage"""
        if self.workload_pattern == 'idle':
            return random.uniform(100, 1000)
        else:
            base_usage = self.total_memory * (0.3 + 0.6 * (utilization / 100))
            return base_usage + random.uniform(-1000, 1000)
    
    def get_clock_speeds(self, utilization: float) -> Dict[str, float]:
        """Get SM and memory clock speeds"""
        max_sm_clock = 1980  # H100 boost clock
        max_mem_clock = 2619  # H100 memory clock
        
        if self.has_throttling:
            sm_clock = max_sm_clock * 0.7 + random.uniform(-50, 50)
            mem_clock = max_mem_clock * 0.8 + random.uniform(-100, 100)
        else:
            sm_clock = max_sm_clock * (0.5 + 0.5 * utilization / 100) + random.uniform(-50, 50)
            mem_clock = max_mem_clock * (0.7 + 0.3 * utilization / 100) + random.uniform(-100, 100)
        
        return {
            'sm_clock_mhz': max(500, sm_clock),
            'memory_clock_mhz': max(800, mem_clock)
        }
    
    def get_ecc_errors(self, t: float) -> Dict[str, int]:
        """Get ECC error counts"""
        if self.has_ecc_errors:
            # Errors accumulate over time
            base_sbe = int(t / 10) + random.randint(0, 5)
            base_dbe = random.randint(0, 2) if random.random() < 0.01 else 0
        else:
            base_sbe = random.randint(0, 2) if random.random() < 0.01 else 0
            base_dbe = 0
        
        return {
            'ecc_sbe_volatile_total': base_sbe,
            'ecc_dbe_volatile_total': base_dbe,
            'ecc_sbe_aggregate_total': base_sbe * 2,
            'ecc_dbe_aggregate_total': base_dbe
        }
    
    def get_nvlink_metrics(self) -> Dict[str, float]:
        """Get NVLink metrics"""
        # Assume 8 NVLink connections at ~25 GB/s each
        total_bandwidth = random.uniform(150000, 200000)  # MB/s
        
        crc_errors = random.randint(0, 5) if random.random() < 0.02 else 0
        recovery_errors = random.randint(0, 2) if random.random() < 0.01 else 0
        
        return {
            'nvlink_bandwidth_total': total_bandwidth,
            'nvlink_crc_errors': crc_errors,
            'nvlink_recovery_errors': recovery_errors
        }
    
    def get_pcie_metrics(self) -> Dict[str, float]:
        """Get PCIe metrics"""
        tx_throughput = random.uniform(1000, 50000)  # KB/s
        rx_throughput = random.uniform(1000, 50000)
        replay_counter = random.randint(0, 10) if random.random() < 0.05 else 0
        
        return {
            'pcie_tx_throughput_kbps': tx_throughput,
            'pcie_rx_throughput_kbps': rx_throughput,
            'pcie_replay_counter': replay_counter
        }
    
    def get_throttle_reasons(self) -> int:
        """Get throttle reasons bitmask"""
        if not self.has_throttling:
            return 0
        
        # Simulate different throttle reasons
        reasons = 0
        if random.random() < 0.5:
            reasons |= (1 << 5)  # SW thermal throttling
        if random.random() < 0.3:
            reasons |= (1 << 6)  # HW thermal throttling
        if random.random() < 0.2:
            reasons |= (1 << 2)  # SW power cap
        
        return reasons
    
    def get_xid_errors(self) -> int:
        """Get XID error count"""
        if random.random() < 0.01:  # 1% chance
            return random.randint(1, 5)
        return 0
    
    def get_all_metrics(self, t: float) -> Dict[str, Any]:
        """Get all metrics for this GPU at time t"""
        utilization = self.get_utilization(t)
        temperature = self.get_temperature(utilization, t)
        power = self.get_power(utilization)
        memory_used = self.get_memory_used(utilization)
        clocks = self.get_clock_speeds(utilization)
        ecc = self.get_ecc_errors(t)
        nvlink = self.get_nvlink_metrics()
        pcie = self.get_pcie_metrics()
        
        metrics = {
            'gpu_utilization': utilization,
            'memory_copy_utilization': random.uniform(0, 30),
            'encoder_utilization': random.uniform(0, 20),
            'decoder_utilization': random.uniform(0, 20),
            'gpu_temp_c': temperature,
            'memory_temp_c': temperature - random.uniform(5, 10),
            'power_draw_w': power,
            'total_energy_consumption': power * t / 3600,  # kWh
            'fb_memory_free_mb': self.total_memory - memory_used,
            'fb_memory_used_mb': memory_used,
            'fb_memory_total_mb': self.total_memory,
            'memory_utilization': (memory_used / self.total_memory) * 100,
            'sm_clock_mhz': clocks['sm_clock_mhz'],
            'memory_clock_mhz': clocks['memory_clock_mhz'],
            'clock_throttle_reasons': self.get_throttle_reasons(),
            'xid_errors': self.get_xid_errors(),
            'compute_pids': random.randint(0, 8),
            'graphics_pids': 0,
        }
        
        metrics.update(ecc)
        metrics.update(nvlink)
        metrics.update(pcie)
        
        # Simulate health degradation over time
        if random.random() < 0.001:
            self.health_degradation += 0.1
        
        return metrics


class DCGMClusterSimulator:
    """Simulates an entire GPU cluster"""
    
    def __init__(self, num_gpus: int = 1000, num_clusters: int = 10):
        self.num_gpus = num_gpus
        self.num_clusters = num_clusters
        self.gpus: List[MockGPU] = []
        self.start_time = time.time()
        
        self._generate_cluster()
    
    def _generate_cluster(self):
        """Generate cluster topology"""
        gpus_per_cluster = self.num_gpus // self.num_clusters
        gpus_per_node = 8  # Standard 8-GPU nodes
        
        for cluster_id in range(self.num_clusters):
            cluster_gpus = gpus_per_cluster if cluster_id < self.num_clusters - 1 else \
                          self.num_gpus - (cluster_id * gpus_per_cluster)
            
            num_nodes = (cluster_gpus + gpus_per_node - 1) // gpus_per_node
            
            for node_id in range(num_nodes):
                hostname = f"cluster{cluster_id:02d}-node{node_id:03d}"
                
                node_gpus_count = min(gpus_per_node, cluster_gpus - (node_id * gpus_per_node))
                
                for gpu_idx in range(node_gpus_count):
                    gpu_id = f"{cluster_id * gpus_per_cluster + node_id * gpus_per_node + gpu_idx}"
                    gpu = MockGPU(gpu_id, hostname, cluster_id=cluster_id)
                    self.gpus.append(gpu)
        
        # Set up NVLink partnerships (GPUs on same node)
        gpu_by_host = {}
        for gpu in self.gpus:
            if gpu.hostname not in gpu_by_host:
                gpu_by_host[gpu.hostname] = []
            gpu_by_host[gpu.hostname].append(gpu)
        
        # Create NVLink mesh within each node
        for host_gpus in gpu_by_host.values():
            for i, gpu in enumerate(host_gpus):
                # Each GPU connects to 2-3 neighbors in the node
                partners = []
                if i > 0:
                    partners.append(host_gpus[i-1].gpu_id)
                if i < len(host_gpus) - 1:
                    partners.append(host_gpus[i+1].gpu_id)
                gpu.nvlink_partner = partners
        
        print(f"Generated cluster with {len(self.gpus)} GPUs across {len(gpu_by_host)} nodes")
    
    def get_topology(self) -> Dict[str, Any]:
        """Get cluster topology structure"""
        nodes = {}
        links = []
        
        for gpu in self.gpus:
            if gpu.hostname not in nodes:
                nodes[gpu.hostname] = {
                    'hostname': gpu.hostname,
                    'cluster_id': gpu.cluster_id,
                    'gpus': []
                }
            
            nodes[gpu.hostname]['gpus'].append({
                'gpu_id': gpu.gpu_id,
                'gpu_uuid': gpu.uuid,
                'model': gpu.model
            })
            
            # Add NVLink connections
            if gpu.nvlink_partner:
                for partner_id in gpu.nvlink_partner:
                    links.append({
                        'source': gpu.gpu_id,
                        'target': partner_id,
                        'type': 'nvlink',
                        'bandwidth': random.uniform(150, 200)  # GB/s
                    })
        
        # Add inter-node links (simulated network fabric)
        node_list = list(nodes.keys())
        for i, node1 in enumerate(node_list):
            # Connect to a few nearby nodes in the cluster
            cluster_id = nodes[node1]['cluster_id']
            for node2 in node_list[i+1:i+4]:
                if nodes[node2]['cluster_id'] == cluster_id:
                    # Use first GPU from each node as representative
                    gpu1 = nodes[node1]['gpus'][0]['gpu_id']
                    gpu2 = nodes[node2]['gpus'][0]['gpu_id']
                    links.append({
                        'source': gpu1,
                        'target': gpu2,
                        'type': 'network',
                        'bandwidth': random.uniform(100, 200)  # GB/s Infiniband
                    })
        
        return {
            'nodes': list(nodes.values()),
            'links': links,
            'clusters': self.num_clusters,
            'total_gpus': len(self.gpus)
        }
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get current snapshot of all GPU metrics"""
        t = time.time() - self.start_time
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'gpus': []
        }
        
        for gpu in self.gpus:
            metrics = gpu.get_all_metrics(t)
            metrics['gpu_id'] = gpu.gpu_id
            metrics['hostname'] = gpu.hostname
            metrics['gpu_uuid'] = gpu.uuid
            metrics['gpu_model'] = gpu.model
            metrics['cluster_id'] = gpu.cluster_id
            
            snapshot['gpus'].append(metrics)
        
        return snapshot
    
    def generate_prometheus_metrics(self) -> str:
        """Generate Prometheus exposition format metrics"""
        t = time.time() - self.start_time
        output = []
        
        # Group metrics by type
        metric_groups = {}
        
        for gpu in self.gpus:
            metrics = gpu.get_all_metrics(t)
            
            for metric_name, value in metrics.items():
                # Convert to DCGM metric name
                dcgm_name = f"DCGM_FI_DEV_{metric_name.upper()}"
                
                if dcgm_name not in metric_groups:
                    metric_groups[dcgm_name] = []
                
                metric_groups[dcgm_name].append({
                    'gpu': gpu.gpu_id,
                    'hostname': gpu.hostname,
                    'UUID': gpu.uuid,
                    'modelName': gpu.model,
                    'value': value
                })
        
        # Format as Prometheus metrics
        for metric_name, samples in metric_groups.items():
            output.append(f"# HELP {metric_name} DCGM metric")
            output.append(f"# TYPE {metric_name} gauge")
            
            for sample in samples:
                labels = ','.join([f'{k}="{v}"' for k, v in sample.items() if k != 'value'])
                output.append(f'{metric_name}{{{labels}}} {sample["value"]}')
        
        return '\n'.join(output)
    
    def generate_historical_csv(self, hours: int = 24, interval_sec: int = 60) -> pd.DataFrame:
        """Generate historical data as CSV"""
        data = []
        num_samples = (hours * 3600) // interval_sec
        
        for i in range(num_samples):
            t = i * interval_sec
            timestamp = datetime.now() - timedelta(seconds=(num_samples - i) * interval_sec)
            
            # Sample subset of GPUs to keep file size reasonable
            sample_gpus = random.sample(self.gpus, min(100, len(self.gpus)))
            
            for gpu in sample_gpus:
                metrics = gpu.get_all_metrics(t)
                metrics['timestamp'] = timestamp
                metrics['gpu_id'] = gpu.gpu_id
                metrics['hostname'] = gpu.hostname
                metrics['gpu_uuid'] = gpu.uuid
                
                data.append(metrics)
        
        return pd.DataFrame(data)


# Flask app for Prometheus exporter
app = Flask(__name__)
simulator = None


@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    global simulator
    if simulator is None:
        simulator = DCGMClusterSimulator(num_gpus=1000, num_clusters=10)
    
    metrics_output = simulator.generate_prometheus_metrics()
    return Response(metrics_output, mimetype='text/plain')


@app.route('/snapshot')
def snapshot():
    """JSON snapshot endpoint"""
    global simulator
    if simulator is None:
        simulator = DCGMClusterSimulator(num_gpus=1000, num_clusters=10)
    
    return simulator.get_snapshot()


@app.route('/topology')
def topology():
    """Topology endpoint"""
    global simulator
    if simulator is None:
        simulator = DCGMClusterSimulator(num_gpus=1000, num_clusters=10)
    
    return simulator.get_topology()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DCGM Mock Data Generator')
    parser.add_argument('--mode', choices=['server', 'csv', 'test'], default='test',
                       help='Run mode: server (Prometheus exporter), csv (generate CSV), or test')
    parser.add_argument('--gpus', type=int, default=1000, help='Number of GPUs to simulate')
    parser.add_argument('--clusters', type=int, default=10, help='Number of clusters')
    parser.add_argument('--output', type=str, default='mock_dcgm_data.csv', help='Output CSV file')
    parser.add_argument('--hours', type=int, default=24, help='Hours of historical data to generate')
    parser.add_argument('--port', type=int, default=9091, help='Port for Prometheus exporter')
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        print(f"Starting DCGM Mock Prometheus Exporter on port {args.port}")
        print(f"Simulating {args.gpus} GPUs in {args.clusters} clusters")
        print(f"Metrics endpoint: http://localhost:{args.port}/metrics")
        print(f"Snapshot endpoint: http://localhost:{args.port}/snapshot")
        print(f"Topology endpoint: http://localhost:{args.port}/topology")
        
        simulator = DCGMClusterSimulator(num_gpus=args.gpus, num_clusters=args.clusters)
        app.run(host='0.0.0.0', port=args.port, debug=False)
    
    elif args.mode == 'csv':
        print(f"Generating {args.hours} hours of historical data for {args.gpus} GPUs...")
        sim = DCGMClusterSimulator(num_gpus=args.gpus, num_clusters=args.clusters)
        df = sim.generate_historical_csv(hours=args.hours)
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output} ({len(df)} rows)")
    
    else:  # test mode
        print("Running test mode...")
        sim = DCGMClusterSimulator(num_gpus=20, num_clusters=2)
        
        print("\n=== Topology ===")
        topo = sim.get_topology()
        print(f"Nodes: {len(topo['nodes'])}")
        print(f"Links: {len(topo['links'])}")
        print(f"Sample node: {json.dumps(topo['nodes'][0], indent=2)}")
        
        print("\n=== Snapshot ===")
        snapshot = sim.get_snapshot()
        print(f"Total GPUs: {len(snapshot['gpus'])}")
        print(f"Sample GPU metrics: {json.dumps(snapshot['gpus'][0], indent=2, default=str)}")
        
        print("\n=== Prometheus Metrics (first 50 lines) ===")
        prom_metrics = sim.generate_prometheus_metrics()
        print('\n'.join(prom_metrics.split('\n')[:50]))
