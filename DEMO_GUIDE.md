# Live Demo Simulation Guide

## Overview

An autonomous, cinematic demonstration of the GPU cluster monitoring system with real-time waveforms, automatic issue detection, and a sleek black & white UI.

## Features

### Visual Design
- **Chic Black & White UI**: Professional, high-contrast design
- **Premium Typography**: Inter font family for UI, Roboto Mono for metrics
- **Smooth Animations**: Sliding alerts, pulsing warnings, smooth chart updates
- **Real-time Waveforms**: Streaming line charts updated 10x per second

### Autonomous Behavior
- **One-Click Start**: Press "START SIMULATION" and watch it run
- **Auto-Issue Detection**: Automatically flags problems as they occur
- **Live Alerts**: Real-time alert feed with timestamps
- **Color-Coded Severity**: Critical (red), Warning (yellow), Info (blue)

### Real-Time Visualization
- **Streaming Waveforms**: 3 live charts (Utilization, Temperature, Power)
- **GPU Grid**: 8 GPU boxes showing live status with color indicators
- **Cluster Metrics**: 4 key aggregate metrics updated in real-time
- **Alert Feed**: Scrolling list of issues as they're detected

### Data Simulation
- **CSV Replay**: Uses your actual cluster data from system_metrics/
- **Intelligent Flagging**: Detects high temps, low utilization, anomalies
- **Smooth Playback**: 100ms update interval for fluid motion
- **Multi-Node Support**: Switch between nodes during simulation

## How to Use

### 1. Start the Server
```bash
python app.py
```

### 2. Open the Demo
Navigate to: **http://localhost:5000/demo.html**

Or click "Launch Live Demo" from the main dashboard.

### 3. Start Simulation
1. Click the **START SIMULATION** button
2. Data loads (shows loading animation)
3. Simulation begins automatically
4. Watch as metrics stream and alerts appear

### 4. During Simulation
- **Switch Nodes**: Click node tabs at top to view different nodes
- **Watch Alerts**: Right panel shows real-time issues
- **Monitor Waveforms**: 3 streaming charts show live data
- **GPU Status**: Grid shows per-GPU health with color coding

### 5. Stop Simulation
Click **STOP** button to pause the demo

## UI Elements

### Header
```
GPU CLUSTER MONITOR
├─ NODES: 4          # Total nodes in cluster
├─ GPUS: 32          # Total GPUs (8 per node)
├─ ALERTS: 15        # Cumulative alert count
└─ UPTIME: 00:05:23  # Simulation runtime
```

### Metrics Dashboard
- **Cluster Util**: Average utilization across active GPUs
- **Avg Temp**: Mean temperature across GPUs
- **Total Power**: Sum of power consumption
- **Health Score**: Composite health (0-100)

### GPU Grid
8 boxes showing per-GPU status:
- **White border**: Normal operation
- **Yellow border + pulse**: Warning (temp 80-85°C or low util)
- **Red border + pulse**: Critical (temp >85°C)

### Waveforms
Real-time streaming charts:
1. **Utilization**: White line, 0-100%
2. **Temperature**: Yellow line, temperatures in °C
3. **Power**: Blue line, watts consumed

### Alert Types
- **CRITICAL** (Red): Immediate attention required
  - GPU temperature >85°C
  - Hardware failures
  
- **WARNING** (Yellow): Monitor closely
  - Temperature 80-85°C
  - Low utilization <10%
  
- **INFO** (Blue): Informational
  - Anomaly patterns detected
  - Behavioral changes

## Alert Examples

```
15:23:45
GPU 3 Critical Temperature
Temperature reached 87.2°C on node-2

15:23:12
GPU 7 Low Utilization
Utilization dropped to 4.3% on node-1

15:22:58
GPU 2 Anomaly Detected
Unusual pattern detected in utilization on node-3
```

## Technical Details

### Data Flow
1. Loads CSV files for all nodes on startup
2. Plays back data at 10Hz (100ms intervals)
3. Analyzes metrics for threshold violations
4. Generates alerts with 1-5% probability per update
5. Updates all UI elements synchronously

### Performance
- **Update Frequency**: 10 Hz (100ms)
- **Chart Window**: 50 data points (5 seconds)
- **Alert Limit**: 20 most recent alerts shown
- **Memory**: ~10MB for data caching
- **CPU**: <5% on modern systems

### Alert Logic
```python
Critical Temperature:
  - Threshold: >85°C
  - Alert probability: 2% per update when exceeded
  
Warning Temperature:
  - Threshold: 80-85°C
  - Reduces health score by 5 points
  
Low Utilization:
  - Threshold: <10%
  - Alert probability: 1% per update
  
Anomaly:
  - Random detection: 0.5% per update
  - Simulates ML anomaly detection
```

### Chart Configuration
- Type: Line chart (Chart.js)
- Animation: Disabled for performance
- Update mode: 'none' (manual refresh)
- Data window: Sliding 50-point window
- Tension: 0.4 (smooth curves)

## Customization

### Change Update Speed
Edit `demo.html`, line ~250:
```javascript
const UPDATE_INTERVAL = 100; // milliseconds (default: 100ms = 10Hz)
```

### Adjust Window Size
Edit `demo.html`, line ~251:
```javascript
const WINDOW_SIZE = 50; // data points (default: 50 = 5 seconds)
```

### Modify Alert Thresholds
Edit `demo.html`, lines ~380-400:
```javascript
if (temp > 85) {  // Change critical threshold
    // ...
    if (Math.random() < 0.02) {  // Change alert probability
        addAlert('critical', ...);
    }
}
```

### Change Colors
Edit `demo.html` CSS section:
```css
/* Critical alert color */
.alert-item.critical {
    border-left-color: #ff4444;  /* Change to your color */
}

/* Utilization chart line */
borderColor: '#ffffff',  /* Change chart colors */
```

## Use Cases

### 1. Product Demo
- Show to clients/stakeholders
- Demonstrate real-time monitoring
- Highlight automatic issue detection
- Professional, polished presentation

### 2. Trade Shows
- Run on loop at booth
- Eye-catching black & white design
- Autonomous operation (no interaction needed)
- Shows product in action

### 3. Sales Presentations
- Start simulation during pitch
- Point out features as they appear
- Live alerts demonstrate value
- Professional aesthetic

### 4. Internal Demos
- Onboard new team members
- Show system capabilities
- Test visualization ideas
- Validate UI/UX decisions

## Tips for Best Demo

### Before Starting
1. Ensure server is running (`python app.py`)
2. Have browser in fullscreen (F11)
3. Test that data loads properly
4. Pre-position window for presentation

### During Demo
- Let simulation run for 30-60 seconds
- Point out alerts as they appear
- Switch between nodes to show scalability
- Explain waveform patterns
- Highlight automatic detection

### Talking Points
- "One-click autonomous monitoring"
- "Real-time waveform visualization"
- "Automatic issue detection and alerting"
- "Multi-node cluster management"
- "Professional, actionable insights"

## Troubleshooting

### Simulation Won't Start
**Issue**: Button disabled, nothing happens
**Fix**: 
1. Check browser console (F12) for errors
2. Verify server is running: http://localhost:5000
3. Check CSV files exist in `system_metrics/`

### No Waveforms Showing
**Issue**: Charts are flat or not updating
**Fix**:
1. Verify data loaded (check browser console)
2. Ensure Chart.js CDN loaded
3. Try refreshing page

### Alerts Not Appearing
**Issue**: No alerts despite issues
**Fix**:
- Alerts are probabilistic (0.5-2% chance)
- Wait 20-30 seconds for alerts to trigger
- Check thresholds match your data ranges

### Performance Issues
**Issue**: Laggy, choppy animation
**Fix**:
1. Reduce `UPDATE_INTERVAL` to 200ms
2. Reduce `WINDOW_SIZE` to 30
3. Close other browser tabs
4. Use Chrome/Edge for best performance

## Browser Compatibility

**Recommended**:
- Chrome 90+
- Edge 90+
- Firefox 88+
- Safari 14+

**Required Features**:
- ES6 JavaScript
- CSS Grid
- Flexbox
- Fetch API
- Canvas (Chart.js)

## Advanced Features (Future)

Potential enhancements:
- [ ] Fourier transform visualizations
- [ ] Spectral analysis overlay
- [ ] Predictive failure indicators
- [ ] Export demo video/screenshots
- [ ] Configurable playback speed
- [ ] Custom alert rules
- [ ] Multi-cluster view
- [ ] Dark/light theme toggle

---

**Perfect for:** Product demos, sales presentations, trade shows, internal showcases

**Designed for:** Maximum visual impact with minimal interaction
