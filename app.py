"""
Cluster Health Monitoring MVP with Advanced Analytics
GPU Datacenter Observability with Predictive Maintenance
"""
import os
import glob
import pandas as pd
import numpy as np
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from advanced_analytics import AdvancedAnalyzer
from model_interface import get_model_instance
from dcgm_mock_generator import DCGMClusterSimulator
import random
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize components
advanced_analyzer = AdvancedAnalyzer()
model = get_model_instance()
mock_simulator = None  # Lazy init

# Path to CSV files
CSV_DIR = 'system_metrics'

def get_simulator():
    """Get or create mock simulator"""
    global mock_simulator
    if mock_simulator is None:
        mock_simulator = DCGMClusterSimulator(num_gpus=1000, num_clusters=10)
    return mock_simulator

def load_all_metrics():
    """Load all CSV files from system_metrics folder"""
    csv_files = glob.glob(os.path.join(CSV_DIR, '*.csv'))
    all_data = {}
    
    for csv_file in csv_files:
        node_name = os.path.basename(csv_file).replace('.csv', '')
        try:
            df = pd.read_csv(csv_file)
            all_data[node_name] = df
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return all_data

def analyze_gpu_health(df, gpu_id):
    """Analyze health metrics for a specific GPU"""
    health_issues = []
    health_score = 100
    
    # Check if CSV has gpu_id column (flat format)
    if 'gpu_id' in df.columns:
        # Get unique GPU IDs and create mapping to indices
        unique_gpus = sorted(df['gpu_id'].unique())
        
        # Map integer gpu_id to actual string GPU ID
        if gpu_id < len(unique_gpus):
            actual_gpu_id = unique_gpus[gpu_id]
            gpu_df = df[df['gpu_id'] == actual_gpu_id]
        else:
            return {'health_score': 50, 'status': 'Unknown', 'issues': ['GPU not present in data']}
        
        if len(gpu_df) == 0:
            return {'health_score': 50, 'status': 'Unknown', 'issues': ['No data available']}
        
        # Check temperature
        if 'gpu_temp_c' in gpu_df.columns:
            temps = gpu_df['gpu_temp_c'].dropna()
            if len(temps) > 0:
                avg_temp = temps.mean()
                max_temp = temps.max()
                if max_temp > 85:
                    health_issues.append(f"Critical temp: {max_temp:.1f}°C")
                    health_score -= 30
                elif max_temp > 80:
                    health_issues.append(f"High temp: {max_temp:.1f}°C")
                    health_score -= 15
        
        # Check GPU utilization
        if 'gpu_utilization' in gpu_df.columns:
            utilization = gpu_df['gpu_utilization'].dropna()
            if len(utilization) > 0:
                avg_util = utilization.mean()
                if avg_util < 10:
                    health_issues.append(f"Low utilization: {avg_util:.1f}%")
                    health_score -= 20
        
        # Check power anomalies
        if 'power_draw_w' in gpu_df.columns:
            power = gpu_df['power_draw_w'].dropna()
            if len(power) > 0:
                if power.std() > 50:  # High variance in power
                    health_issues.append("Unstable power draw")
                    health_score -= 10
    
    # Fallback to old format
    else:
        gpu_prefix = f'system.gpu.{gpu_id}'
        
        # Check temperature
        temp_col = f'{gpu_prefix}.temp'
        if temp_col in df.columns:
            temps = df[temp_col].dropna()
            if len(temps) > 0:
                avg_temp = temps.mean()
                max_temp = temps.max()
                if max_temp > 85:
                    health_issues.append(f"Critical temp: {max_temp}°C")
                    health_score -= 30
                elif max_temp > 80:
                    health_issues.append(f"High temp: {max_temp}°C")
                    health_score -= 15
        
        # Check memory errors
        corrected_col = f'{gpu_prefix}.correctedMemoryErrors'
        uncorrected_col = f'{gpu_prefix}.uncorrectedMemoryErrors'
        
        if corrected_col in df.columns:
            corrected_errors = df[corrected_col].dropna().sum()
            if corrected_errors > 0:
                health_issues.append(f"{int(corrected_errors)} corrected memory errors")
                health_score -= 10
        
        if uncorrected_col in df.columns:
            uncorrected_errors = df[uncorrected_col].dropna().sum()
            if uncorrected_errors > 0:
                health_issues.append(f"{int(uncorrected_errors)} UNCORRECTED memory errors")
                health_score -= 40
        
        # Check GPU utilization
        gpu_util_col = f'{gpu_prefix}.gpu'
        if gpu_util_col in df.columns:
            utilization = df[gpu_util_col].dropna()
            if len(utilization) > 0:
                avg_util = utilization.mean()
                if avg_util < 10:
                    health_issues.append(f"Low utilization: {avg_util:.1f}%")
                    health_score -= 20
        
        # Check power anomalies
        power_col = f'{gpu_prefix}.powerWatts'
        if power_col in df.columns:
            power = df[power_col].dropna()
            if len(power) > 0:
                if power.std() > 50:  # High variance in power
                    health_issues.append("Unstable power draw")
                    health_score -= 10
    
    health_score = max(0, health_score)
    status = "Healthy" if health_score > 80 else "Warning" if health_score > 50 else "Critical"
    
    return {
        'health_score': health_score,
        'status': status,
        'issues': health_issues
    }

def get_gpu_metrics(df, gpu_id):
    """Extract key metrics for a GPU"""
    metrics = {
        'gpu_id': gpu_id,
        'utilization': [],
        'temperature': [],
        'power': [],
        'memory_allocated': [],
        'timestamps': []
    }
    
    # Check if CSV has gpu_id column (flat format)
    if 'gpu_id' in df.columns:
        # Get unique GPU IDs and create mapping to indices
        unique_gpus = sorted(df['gpu_id'].unique())
        
        # Map integer gpu_id to actual string GPU ID
        if gpu_id < len(unique_gpus):
            actual_gpu_id = unique_gpus[gpu_id]
            gpu_df = df[df['gpu_id'] == actual_gpu_id]
        else:
            # GPU doesn't exist, return empty but fill with fake data for demo
            return metrics
        
        if len(gpu_df) == 0:
            return metrics
        
        # Extract metrics
        if 'gpu_utilization' in gpu_df.columns:
            metrics['utilization'] = gpu_df['gpu_utilization'].fillna(0).tolist()
        
        if 'gpu_temp_c' in gpu_df.columns:
            metrics['temperature'] = gpu_df['gpu_temp_c'].fillna(0).tolist()
        
        if 'power_draw_w' in gpu_df.columns:
            metrics['power'] = gpu_df['power_draw_w'].fillna(0).tolist()
        
        if 'memory_utilization' in gpu_df.columns:
            metrics['memory_allocated'] = gpu_df['memory_utilization'].fillna(0).tolist()
        
        if 'timestamp' in gpu_df.columns:
            metrics['timestamps'] = gpu_df['timestamp'].tolist()
    
    # Fallback to old format (system.gpu.X.metric)
    else:
        gpu_prefix = f'system.gpu.{gpu_id}'
        
        # Get utilization
        util_col = f'{gpu_prefix}.gpu'
        if util_col in df.columns:
            metrics['utilization'] = df[util_col].fillna(0).tolist()
        
        # Get temperature
        temp_col = f'{gpu_prefix}.temp'
        if temp_col in df.columns:
            metrics['temperature'] = df[temp_col].fillna(0).tolist()
        
        # Get power
        power_col = f'{gpu_prefix}.powerWatts'
        if power_col in df.columns:
            metrics['power'] = df[power_col].fillna(0).tolist()
        
        # Get memory
        mem_col = f'{gpu_prefix}.memoryAllocated'
        if mem_col in df.columns:
            metrics['memory_allocated'] = df[mem_col].fillna(0).tolist()
        
        # Get timestamps
        if '_timestamp' in df.columns:
            metrics['timestamps'] = df['_timestamp'].fillna(0).tolist()
        elif '_runtime' in df.columns:
            metrics['timestamps'] = df['_runtime'].fillna(0).tolist()
    
    return metrics

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return send_from_directory('static', 'index.html')

@app.route('/advanced.html')
def advanced():
    """Serve the advanced analytics dashboard"""
    return send_from_directory('static', 'advanced.html')

@app.route('/demo.html')
def demo():
    """Serve the autonomous simulation demo"""
    return send_from_directory('static', 'demo.html')

@app.route('/api/nodes')
def get_nodes():
    """Get list of all nodes"""
    data = load_all_metrics()
    nodes = list(data.keys())
    return jsonify({'nodes': nodes})

@app.route('/api/cluster-health')
def cluster_health():
    """Get overall cluster health summary"""
    data = load_all_metrics()
    
    cluster_summary = {
        'total_nodes': len(data),
        'total_gpus': 0,
        'healthy_gpus': 0,
        'warning_gpus': 0,
        'critical_gpus': 0,
        'nodes': []
    }
    
    for node_name, df in data.items():
        node_info = {
            'name': node_name,
            'gpus': []
        }
        
        # Analyze each GPU (0-7)
        for gpu_id in range(8):
            health = analyze_gpu_health(df, gpu_id)
            
            cluster_summary['total_gpus'] += 1
            if health['status'] == 'Healthy':
                cluster_summary['healthy_gpus'] += 1
            elif health['status'] == 'Warning':
                cluster_summary['warning_gpus'] += 1
            else:
                cluster_summary['critical_gpus'] += 1
            
            node_info['gpus'].append({
                'id': gpu_id,
                'health_score': health['health_score'],
                'status': health['status'],
                'issues': health['issues']
            })
        
        cluster_summary['nodes'].append(node_info)
    
    return jsonify(cluster_summary)

@app.route('/api/gpu-metrics/<node_name>/<int:gpu_id>')
def gpu_metrics(node_name, gpu_id):
    """Get detailed metrics for a specific GPU"""
    data = load_all_metrics()
    
    if node_name not in data:
        return jsonify({'error': 'Node not found'}), 404
    
    df = data[node_name]
    metrics = get_gpu_metrics(df, gpu_id)
    
    return jsonify(metrics)

@app.route('/api/cluster-utilization')
def cluster_utilization():
    """Get cluster-wide GPU utilization over time"""
    data = load_all_metrics()
    
    result = {
        'nodes': []
    }
    
    for node_name, df in data.items():
        node_data = {
            'name': node_name,
            'gpus': []
        }
        
        for gpu_id in range(8):
            metrics = get_gpu_metrics(df, gpu_id)
            if metrics['utilization']:
                node_data['gpus'].append({
                    'gpu_id': gpu_id,
                    'avg_utilization': np.mean(metrics['utilization']),
                    'max_utilization': np.max(metrics['utilization']),
                    'avg_temp': np.mean(metrics['temperature']) if metrics['temperature'] else 0,
                    'max_temp': np.max(metrics['temperature']) if metrics['temperature'] else 0
                })
        
        result['nodes'].append(node_data)
    
    return jsonify(result)

@app.route('/api/advanced-analytics/<node_name>/<int:gpu_id>')
def advanced_analytics(node_name, gpu_id):
    """Get comprehensive advanced analytics for a specific GPU"""
    data = load_all_metrics()
    
    if node_name not in data:
        return jsonify({'error': 'Node not found'}), 404
    
    df = data[node_name]
    analysis = advanced_analyzer.analyze_gpu_comprehensive(df, gpu_id)
    
    return jsonify(analysis)

@app.route('/api/cluster-advanced-summary')
def cluster_advanced_summary():
    """Get advanced analytics summary for entire cluster"""
    data = load_all_metrics()
    
    summary = {
        'nodes': [],
        'cluster_insights': {
            'total_anomalies': 0,
            'high_risk_gpus': 0,
            'power_inefficient_gpus': 0,
            'thermal_issues': 0
        }
    }
    
    for node_name, df in data.items():
        node_summary = {
            'name': node_name,
            'gpus': []
        }
        
        for gpu_id in range(8):
            analysis = advanced_analyzer.analyze_gpu_comprehensive(df, gpu_id)
            
            # Extract key insights
            gpu_insights = {
                'gpu_id': gpu_id,
                'trends': {},
                'anomalies': {},
                'physics': {},
                'warnings': []
            }
            
            # Temporal insights
            if 'temporal' in analysis and 'utilization' in analysis['temporal']:
                util_trend = analysis['temporal']['utilization'].get('trend', {})
                gpu_insights['trends']['utilization'] = util_trend.get('trend', 'unknown')
                gpu_insights['trends']['util_slope'] = util_trend.get('slope', 0)
                
                anomalies = analysis['temporal']['utilization'].get('anomalies', [])
                gpu_insights['anomalies']['count'] = len(anomalies)
                summary['cluster_insights']['total_anomalies'] += len(anomalies)
                
                periodicity = analysis['temporal']['utilization'].get('periodicity', {})
                if periodicity.get('has_periodicity'):
                    gpu_insights['warnings'].append(f"Periodic pattern detected (period: {periodicity.get('dominant_period', 'N/A')})")
            
            # Physics insights
            if 'physics' in analysis:
                if 'thermal' in analysis['physics']:
                    thermal = analysis['physics']['thermal']
                    gpu_insights['physics']['thermal_efficiency'] = thermal.get('thermal_efficiency', 0)
                    gpu_insights['physics']['thermal_stability'] = thermal.get('thermal_stability', 'unknown')
                    
                    if thermal.get('thermal_stability') in ['poor', 'moderate']:
                        summary['cluster_insights']['thermal_issues'] += 1
                        gpu_insights['warnings'].append(f"Thermal stability: {thermal.get('thermal_stability')}")
                
                if 'power' in analysis['physics']:
                    power = analysis['physics']['power']
                    gpu_insights['physics']['power_efficiency'] = power.get('efficiency_score', 0)
                    
                    if power.get('efficiency_score', 0) < 50:
                        summary['cluster_insights']['power_inefficient_gpus'] += 1
                        gpu_insights['warnings'].append("Low power efficiency")
                
                if 'workload' in analysis['physics']:
                    workload = analysis['physics']['workload']
                    gpu_insights['physics']['workload_type'] = workload.get('workload_type', 'unknown')
                    gpu_insights['physics']['balance_score'] = workload.get('balance_score', 0)
            
            node_summary['gpus'].append(gpu_insights)
        
        summary['nodes'].append(node_summary)
    
    return jsonify(summary)

@app.route('/api/correlation-analysis/<node_name>')
def correlation_analysis(node_name):
    """Get correlation analysis between different metrics for a node"""
    data = load_all_metrics()
    
    if node_name not in data:
        return jsonify({'error': 'Node not found'}), 404
    
    df = data[node_name]
    
    # Calculate correlations between key metrics
    correlations = {}
    
    for gpu_id in range(8):
        gpu_prefix = f'system.gpu.{gpu_id}'
        
        metrics = {
            'utilization': f'{gpu_prefix}.gpu',
            'temperature': f'{gpu_prefix}.temp',
            'power': f'{gpu_prefix}.powerWatts',
            'memory': f'{gpu_prefix}.memoryAllocated'
        }
        
        # Get available metrics
        available_data = {}
        for metric_name, col_name in metrics.items():
            if col_name in df.columns:
                available_data[metric_name] = df[col_name].fillna(0).values
        
        if len(available_data) >= 2:
            # Calculate pairwise correlations
            gpu_correlations = {}
            metric_names = list(available_data.keys())
            
            for i, metric1 in enumerate(metric_names):
                for metric2 in metric_names[i+1:]:
                    data1 = available_data[metric1]
                    data2 = available_data[metric2]
                    
                    if len(data1) > 2 and len(data2) > 2:
                        corr = np.corrcoef(data1, data2)[0, 1]
                        if not np.isnan(corr):
                            gpu_correlations[f'{metric1}_vs_{metric2}'] = float(corr)
            
            correlations[f'gpu_{gpu_id}'] = gpu_correlations
    
    return jsonify({'node': node_name, 'correlations': correlations})

@app.route('/api/predictive-analysis/<node_name>/<int:gpu_id>')
def predictive_analysis(node_name, gpu_id):
    """Get predictive analysis for a specific GPU"""
    data = load_all_metrics()
    
    if node_name not in data:
        return jsonify({'error': 'Node not found'}), 404
    
    df = data[node_name]
    gpu_prefix = f'system.gpu.{gpu_id}'
    
    predictions = {}
    
    # Forecast temperature
    temp_col = f'{gpu_prefix}.temp'
    if temp_col in df.columns:
        temp_data = df[temp_col].fillna(0).values
        temp_forecast = advanced_analyzer.predictive.forecast_metric(temp_data, steps=10)
        predictions['temperature'] = temp_forecast
    
    # Forecast utilization
    util_col = f'{gpu_prefix}.gpu'
    if util_col in df.columns:
        util_data = df[util_col].fillna(0).values
        util_forecast = advanced_analyzer.predictive.forecast_metric(util_data, steps=10)
        predictions['utilization'] = util_forecast
    
    # Failure risk prediction
    # Simple health history based on temperature and errors
    health_history = []
    if temp_col in df.columns:
        temps = df[temp_col].fillna(0).values
        for temp in temps:
            health_val = 100
            if temp > 85:
                health_val -= 30
            elif temp > 80:
                health_val -= 15
            health_history.append(health_val)
    
    if len(health_history) > 0:
        trend_data = advanced_analyzer.temporal.calculate_trend(health_history)
        failure_risk = advanced_analyzer.predictive.predict_failure_risk(health_history, trend_data)
        predictions['failure_risk'] = failure_risk
    
    return jsonify({'gpu_id': gpu_id, 'predictions': predictions})

@app.route('/api/cluster-analysis')
def cluster_analysis():
    """Get cluster-wide behavioral analysis"""
    data = load_all_metrics()
    
    results = {
        'load_imbalance': {},
        'gpu_clusters': {},
        'synchronization': []
    }
    
    for node_name, df in data.items():
        # Collect utilization data for all GPUs
        gpu_utilizations = []
        for gpu_id in range(8):
            util_col = f'system.gpu.{gpu_id}.gpu'
            if util_col in df.columns:
                util_data = df[util_col].fillna(0).values
                gpu_utilizations.append(util_data)
            else:
                gpu_utilizations.append([])
        
        # Calculate load imbalance
        valid_utils = [u for u in gpu_utilizations if len(u) > 0]
        if len(valid_utils) >= 2:
            imbalance = advanced_analyzer.cluster.calculate_cluster_imbalance(valid_utils)
            results['load_imbalance'][node_name] = imbalance
        
        # Find similar GPUs (clustering)
        if len(valid_utils) >= 3:
            clusters = advanced_analyzer.cluster.find_similar_gpus(valid_utils, n_clusters=min(3, len(valid_utils)))
            results['gpu_clusters'][node_name] = clusters
        
        # Check synchronization between GPU pairs
        sync_pairs = []
        for i in range(len(gpu_utilizations)):
            for j in range(i+1, min(i+3, len(gpu_utilizations))):  # Check neighboring GPUs
                if len(gpu_utilizations[i]) > 0 and len(gpu_utilizations[j]) > 0:
                    sync = advanced_analyzer.cluster.detect_synchronization(
                        gpu_utilizations[i][:100],  # Use first 100 samples
                        gpu_utilizations[j][:100]
                    )
                    if sync['synchronized']:
                        sync_pairs.append({
                            'gpu1': i,
                            'gpu2': j,
                            'lag': sync['lag'],
                            'correlation': sync['correlation']
                        })
        
        if sync_pairs:
            results['synchronization'].append({
                'node': node_name,
                'synchronized_pairs': sync_pairs
            })
    
    return jsonify(results)

@app.route('/api/advanced-models/<node_name>/<int:gpu_id>')
def advanced_models(node_name, gpu_id):
    """Get sophisticated model analysis for a GPU"""
    data = load_all_metrics()
    
    if node_name not in data:
        return jsonify({'error': 'Node not found'}), 404
    
    df = data[node_name]
    
    # Get full analysis which now includes advanced models
    analysis = advanced_analyzer.analyze_gpu_comprehensive(df, gpu_id)
    
    # Extract just the advanced section for focused view
    advanced_results = analysis.get('advanced', {})
    
    return jsonify({
        'gpu_id': gpu_id,
        'node': node_name,
        'advanced_models': advanced_results
    })


# ============== NEW ENDPOINTS FOR PREDICTIVE MAINTENANCE & TOPOLOGY ==============

@app.route('/topology.html')
def topology_page():
    """Serve topology visualization page"""
    return send_from_directory('static', 'topology.html')

@app.route('/predictive_maintenance.html')
def predictive_maintenance_page():
    """Serve predictive maintenance dashboard"""
    return send_from_directory('static', 'predictive_maintenance.html')

@app.route('/gpu_details.html')
def gpu_details_page():
    """Serve detailed GPU hardware analytics"""
    return send_from_directory('static', 'gpu_details.html')

@app.route('/api/topology-full')
def topology_full():
    """Get full topology with health data for 1000 GPUs"""
    simulator = get_simulator()
    topology = simulator.get_topology()
    snapshot = simulator.get_snapshot()
    
    # Enrich topology with health predictions
    enriched_gpus = []
    
    for gpu_data in snapshot['gpus']:
        # Run model prediction
        prediction = model.predict(gpu_data)
        
        enriched_gpu = {
            'gpu_id': gpu_data['gpu_id'],
            'hostname': gpu_data['hostname'],
            'gpu_uuid': gpu_data['gpu_uuid'],
            'gpu_model': gpu_data['gpu_model'],
            'cluster_id': gpu_data['cluster_id'],
            'health_score': prediction['health_score'],
            'status': prediction['status'],
            'rul_hours': prediction['rul_hours'],
            'rul_days': prediction['rul_days'],
            'risk_level': prediction['risk_level'],
            'temperature': gpu_data.get('gpu_temp_c', 0),
            'utilization': gpu_data.get('gpu_utilization', 0),
            'power': gpu_data.get('power_draw_w', 0),
            'anomaly_score': prediction['anomaly_score'],
            'actions': prediction['actions'],
            'causes': prediction['causes']
        }
        enriched_gpus.append(enriched_gpu)
    
    return jsonify({
        'gpus': enriched_gpus,
        'links': topology['links'],
        'num_clusters': topology['clusters'],
        'total_gpus': topology['total_gpus'],
        'num_nodes': len(topology['nodes']),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predictive-maintenance-summary')
def predictive_maintenance_summary():
    """Get predictive maintenance summary for all GPUs"""
    simulator = get_simulator()
    snapshot = simulator.get_snapshot()
    
    # Run predictions on all GPUs
    predictions = []
    
    for gpu_data in snapshot['gpus']:
        prediction = model.predict(gpu_data)
        
        gpu_summary = {
            'gpu_id': gpu_data['gpu_id'],
            'hostname': gpu_data['hostname'],
            'cluster_id': gpu_data['cluster_id'],
            'health_score': prediction['health_score'],
            'status': prediction['status'],
            'rul_hours': prediction['rul_hours'],
            'rul_days': prediction['rul_days'],
            'risk_level': prediction['risk_level'],
            'anomaly_score': prediction['anomaly_score'],
            'anomaly_count': 1 if prediction['anomaly_score'] > 0.5 else 0,
            'actions': prediction['actions'],
            'causes': prediction['causes'],
            'failure_probability_24h': prediction['predictions']['failure_probability_24h'],
            'failure_probability_7d': prediction['predictions']['failure_probability_7d'],
            'failure_probability_30d': prediction['predictions']['failure_probability_30d'],
            'timestamp': prediction['timestamp']
        }
        predictions.append(gpu_summary)
    
    # Calculate summary statistics
    total_gpus = len(predictions)
    critical_gpus = sum(1 for p in predictions if p['risk_level'] == 'Critical')
    high_risk_gpus = sum(1 for p in predictions if p['rul_days'] < 7)
    total_anomalies = sum(p['anomaly_count'] for p in predictions)
    
    return jsonify({
        'gpus': predictions,
        'summary': {
            'total_gpus': total_gpus,
            'critical_gpus': critical_gpus,
            'high_risk_gpus': high_risk_gpus,
            'total_anomalies': total_anomalies,
            'avg_health': sum(p['health_score'] for p in predictions) / total_gpus if total_gpus > 0 else 0,
            'avg_rul_days': sum(p['rul_days'] for p in predictions) / total_gpus if total_gpus > 0 else 0
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/gpu-prediction/<gpu_id>')
def gpu_prediction(gpu_id):
    """Get detailed prediction for a specific GPU"""
    simulator = get_simulator()
    snapshot = simulator.get_snapshot()
    
    # Find the GPU
    gpu_data = next((g for g in snapshot['gpus'] if g['gpu_id'] == gpu_id), None)
    
    if not gpu_data:
        return jsonify({'error': 'GPU not found'}), 404
    
    # Run prediction
    prediction = model.predict(gpu_data)
    
    # Add raw metrics
    prediction['raw_metrics'] = gpu_data
    
    return jsonify(prediction)

@app.route('/api/dcgm-snapshot')
def dcgm_snapshot():
    """Get current DCGM snapshot from simulator"""
    simulator = get_simulator()
    snapshot = simulator.get_snapshot()
    return jsonify(snapshot)

@app.route('/api/dcgm-metrics-prometheus')
def dcgm_metrics_prometheus():
    """Get DCGM metrics in Prometheus format"""
    from flask import Response
    simulator = get_simulator()
    metrics = simulator.generate_prometheus_metrics()
    return Response(metrics, mimetype='text/plain')

@app.route('/api/cluster-health-realtime')
def cluster_health_realtime():
    """Get real-time cluster health with model predictions"""
    simulator = get_simulator()
    snapshot = simulator.get_snapshot()
    
    # Group by cluster and node
    cluster_health = {}
    
    for gpu_data in snapshot['gpus']:
        cluster_id = gpu_data['cluster_id']
        hostname = gpu_data['hostname']
        
        if cluster_id not in cluster_health:
            cluster_health[cluster_id] = {
                'cluster_id': cluster_id,
                'nodes': {},
                'total_gpus': 0,
                'healthy': 0,
                'warning': 0,
                'critical': 0
            }
        
        if hostname not in cluster_health[cluster_id]['nodes']:
            cluster_health[cluster_id]['nodes'][hostname] = {
                'hostname': hostname,
                'gpus': [],
                'avg_health': 0
            }
        
        # Get prediction
        prediction = model.predict(gpu_data)
        
        gpu_info = {
            'gpu_id': gpu_data['gpu_id'],
            'health_score': prediction['health_score'],
            'status': prediction['status'],
            'rul_days': prediction['rul_days']
        }
        
        cluster_health[cluster_id]['nodes'][hostname]['gpus'].append(gpu_info)
        cluster_health[cluster_id]['total_gpus'] += 1
        
        # Count status
        if prediction['health_score'] >= 80:
            cluster_health[cluster_id]['healthy'] += 1
        elif prediction['health_score'] >= 50:
            cluster_health[cluster_id]['warning'] += 1
        else:
            cluster_health[cluster_id]['critical'] += 1
    
    # Calculate average health per node
    for cluster in cluster_health.values():
        for node in cluster['nodes'].values():
            if node['gpus']:
                node['avg_health'] = sum(g['health_score'] for g in node['gpus']) / len(node['gpus'])
    
    return jsonify({
        'clusters': list(cluster_health.values()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/anomaly-detection')
def anomaly_detection():
    """Get real-time anomaly detection results"""
    simulator = get_simulator()
    snapshot = simulator.get_snapshot()
    
    anomalies = []
    
    for gpu_data in snapshot['gpus']:
        prediction = model.predict(gpu_data)
        
        if prediction['anomaly_score'] > 0.5:
            anomaly = {
                'gpu_id': gpu_data['gpu_id'],
                'hostname': gpu_data['hostname'],
                'cluster_id': gpu_data['cluster_id'],
                'anomaly_score': prediction['anomaly_score'],
                'causes': prediction['causes'],
                'severity': 'critical' if prediction['anomaly_score'] > 0.8 else 'warning',
                'temperature': gpu_data.get('gpu_temp_c', 0),
                'utilization': gpu_data.get('gpu_utilization', 0),
                'detected_at': datetime.now().isoformat()
            }
            anomalies.append(anomaly)
    
    # Sort by anomaly score
    anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
    
    return jsonify({
        'anomalies': anomalies,
        'total_anomalies': len(anomalies),
        'critical': sum(1 for a in anomalies if a['severity'] == 'critical'),
        'warnings': sum(1 for a in anomalies if a['severity'] == 'warning'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Run batch predictions on provided GPU metrics"""
    from flask import request
    
    try:
        metrics_list = request.json.get('metrics', [])
        
        if not metrics_list:
            return jsonify({'error': 'No metrics provided'}), 400
        
        predictions = model.batch_predict(metrics_list)
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("GPU Datacenter Observability Platform")
    print("Advanced Predictive Maintenance & Anomaly Detection System")
    print("=" * 70)
    
    print(f"\nLoading CSV files from: {CSV_DIR}")
    
    # Check if CSV directory exists
    if not os.path.exists(CSV_DIR):
        print(f"WARNING: {CSV_DIR} directory not found!")
        print("Using DCGM mock simulator instead...")
    else:
        csv_count = len(glob.glob(os.path.join(CSV_DIR, '*.csv')))
        print(f"Found {csv_count} CSV files")
    
    print("\nInitializing 1000-GPU cluster simulator...")
    print("+ DCGM mock data generator ready")
    print("+ ML model interface loaded")
    print("+ Advanced analytics engine ready")
    
    print("\n" + "=" * 70)
    print("Available Dashboards:")
    print("=" * 70)
    print("   Main Dashboard:              http://localhost:5000")
    print("   Topology Visualization:      http://localhost:5000/topology.html")
    print("   Predictive Maintenance:      http://localhost:5000/predictive_maintenance.html")
    print("   Advanced Analytics:          http://localhost:5000/advanced.html")
    print("   Live Demo:                   http://localhost:5000/demo.html")
    print("\n" + "=" * 70)
    print("API Endpoints:")
    print("=" * 70)
    print("   /api/topology-full                    - Full topology with health")
    print("   /api/predictive-maintenance-summary   - Predictive maintenance data")
    print("   /api/anomaly-detection                - Real-time anomaly alerts")
    print("   /api/dcgm-snapshot                    - DCGM metrics snapshot")
    print("   /api/dcgm-metrics-prometheus          - Prometheus format metrics")
    print("   /api/gpu-prediction/<gpu_id>          - GPU-specific predictions")
    print("=" * 70)
    print("\nFeatures:")
    print("   - 1000 GPU simulation across 10 clusters")
    print("   - Real-time anomaly detection")
    print("   - Predictive maintenance (RUL estimation)")
    print("   - Force-directed topology visualization")
    print("   - Datacenter rack view")
    print("   - DCGM metric ingestion")
    print("   - ML model integration for health prediction")
    print("\nServer starting on http://0.0.0.0:5000")
    print("Press CTRL+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

# For Vercel serverless deployment
application = app
