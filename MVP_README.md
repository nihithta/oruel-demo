# GPU Cluster Health Monitoring MVP

Simple web-based dashboard to monitor GPU cluster health from CSV metrics.

## Features

- 📊 **Real-time Dashboard** - Visual overview of cluster health
- 🖥️ **GPU Monitoring** - Track utilization, temperature, power, and memory
- ⚠️ **Health Scoring** - Automatic detection of unhealthy GPUs
- 📈 **Visualizations** - Interactive charts for cluster analysis
- 🚨 **Issue Detection** - Identifies temperature problems, memory errors, low utilization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python app.py
```

### 3. Open Dashboard

Open your browser to: **http://localhost:5000**

## What It Shows

### Cluster Overview
- Total nodes and GPUs
- Healthy/Warning/Critical GPU counts
- Overall cluster health percentage

### Per-Node Analysis
- Health score for each GPU (0-100)
- Visual status indicators (Green/Yellow/Red)
- Detailed issues when clicking on a GPU

### Detected Issues
- 🔥 **High Temperature** - GPUs over 80°C
- ⚡ **Power Instability** - Unstable power draw
- 💾 **Memory Errors** - Corrected or uncorrected errors
- 📉 **Low Utilization** - Underutilized GPUs (<10%)

### Visualizations
- Average GPU utilization by node
- Maximum temperature by node
- Color-coded warnings

## Health Scoring

Each GPU gets a score (0-100) based on:
- Temperature: -30 points if >85°C, -15 if >80°C
- Memory errors: -10 per corrected, -40 per uncorrected
- Low utilization: -20 points if <10%
- Power instability: -10 points for high variance

**Status Levels:**
- 🟢 **Healthy**: Score > 80
- 🟡 **Warning**: Score 50-80
- 🔴 **Critical**: Score < 50

## Data Source

The app reads CSV files from the `system_metrics/` folder. Each CSV should contain GPU metrics like:
- `system.gpu.X.temp` - Temperature
- `system.gpu.X.gpu` - Utilization %
- `system.gpu.X.powerWatts` - Power consumption
- `system.gpu.X.memoryAllocated` - Memory usage
- `system.gpu.X.correctedMemoryErrors` - Memory errors
- `system.gpu.X.uncorrectedMemoryErrors` - Critical errors

Where X is the GPU ID (0-7).

## No External Dependencies

- ✅ No Docker required
- ✅ No Prometheus/Grafana
- ✅ No database setup
- ✅ Just Python + Flask + CSV files

## Project Structure

```
.
├── app.py                  # Flask backend API
├── static/
│   └── index.html          # Web dashboard
├── system_metrics/         # CSV files with metrics
├── requirements.txt        # Python dependencies
└── MVP_README.md          # This file
```

## API Endpoints

- `GET /` - Dashboard UI
- `GET /api/nodes` - List all nodes
- `GET /api/cluster-health` - Cluster health summary
- `GET /api/cluster-utilization` - Utilization statistics
- `GET /api/gpu-metrics/<node>/<gpu_id>` - Detailed GPU metrics

## Customization

Edit `app.py` to modify:
- Health scoring thresholds
- Temperature limits
- Utilization thresholds
- Analysis logic

## Troubleshooting

**Server won't start?**
- Make sure port 5000 is not in use
- Check that `system_metrics/` folder exists with CSV files

**No data showing?**
- Verify CSV files are in `system_metrics/` folder
- Check browser console for errors
- Ensure CSV columns match expected format

**Charts not rendering?**
- Make sure you have internet connection (Chart.js loads from CDN)
- Check browser console for JavaScript errors
