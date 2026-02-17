# 🔬 Advanced Analytics Guide

## Overview

This system provides **deep, multi-dimensional analysis** of GPU cluster metrics beyond simple health monitoring. It combines temporal analysis, physics-based models, statistical methods, and machine learning to provide comprehensive insights.

---

## 🎯 Analytics Capabilities

### 1. Temporal Analysis

**Time-series pattern detection and trend analysis**

#### Features:
- **Trend Detection**: Linear regression with R² fit quality
  - Calculates slope, acceleration (2nd derivative)
  - Classifies as: increasing, decreasing, or stable
  
- **Anomaly Detection**: Isolation Forest algorithm
  - Automatically identifies unusual patterns
  - Configurable contamination threshold
  - Works without labeled data
  
- **Periodicity Detection**: FFT-based frequency analysis
  - Detects cyclic patterns in utilization/temperature
  - Identifies dominant frequencies
  - Measures periodicity strength
  
- **Volatility Analysis**: Rolling standard deviation
  - Current, average, and maximum volatility
  - Helps identify unstable GPUs

#### Example Insights:
```
GPU 3 Utilization:
├─ Trend: Decreasing (-0.23% per sample)
├─ R²: 0.87 (strong fit)
├─ Acceleration: -0.05 (declining faster)
├─ Anomalies: 12 detected
├─ Periodicity: Yes (period ~15 samples)
└─ Volatility: 8.4% (moderate)
```

---

### 2. Physics-Based Analysis

**Model GPU behavior using physical principles**

#### Thermal Dynamics

Analyzes heat generation, dissipation, and transfer:

- **Thermal Efficiency**: Temperature per watt (°C/W)
  - Lower is better (better cooling)
  - Typical range: 0.1-0.3 °C/W
  
- **Heat Dissipation Rate**: °C per unit time
  - How quickly GPU cools down
  - Indicates cooling system effectiveness
  
- **Thermal Correlation**: Power-temperature relationship
  - Should be high (0.7-0.9)
  - Low correlation indicates cooling issues
  
- **Thermal Stability**: Classification based on temp std dev
  - Excellent: < 2°C std
  - Good: 2-5°C std
  - Moderate: 5-10°C std
  - Poor: > 10°C std

#### Power Efficiency

Analyzes energy consumption patterns:

- **Power Efficiency**: Utilization per watt (%/W)
  - Higher is better
  - Measures compute per energy unit
  
- **Idle Power**: Baseline consumption
  - Should be low (< 50W for modern GPUs)
  
- **Peak Power**: Maximum observed
  - Should stay below enforced limit
  
- **Efficiency Score** (0-100): Correlation between power and utilization
  - 100: Perfect linear scaling
  - < 50: Inefficient (power doesn't track work)

#### Workload Characteristics

Classifies computation patterns:

- **Workload Types**:
  - `compute_intensive`: High util + high memory
  - `memory_intensive`: High memory, low util
  - `compute_bound`: High util, low memory
  - `idle`: Both low
  - `balanced`: Moderate both
  
- **Compute Intensity**: Average GPU utilization
- **Memory Intensity**: Average memory usage
- **Balance Score** (0-100): How well compute/memory are matched
  - High score = efficient use of both

#### Example Insights:
```
GPU 5 Physics:
Thermal:
├─ Efficiency: 0.15 °C/W (excellent)
├─ Dissipation: 0.8 °C/s
├─ Power-Temp Corr: 0.89 (strong)
└─ Stability: Excellent (σ = 1.2°C)

Power:
├─ Efficiency Score: 87/100
├─ Idle: 45W, Peak: 320W
└─ Range: 275W

Workload:
├─ Type: Compute Intensive
├─ Compute: 92%, Memory: 87%
└─ Balance: 95/100
```

---

### 3. Statistical Analysis

**Rigorous statistical characterization**

#### Distribution Statistics

Complete statistical profile for each metric:

- **Central Tendency**: Mean, median
- **Spread**: Standard deviation, variance, IQR
- **Shape**: Skewness, kurtosis
- **Range**: Min, max, quartiles

#### Outlier Detection

Two methods available:

1. **IQR Method**: Q1 - 1.5×IQR, Q3 + 1.5×IQR
2. **Z-Score Method**: |z| > 3

Returns:
- Outlier indices
- Count and percentage
- Used for data quality checks

#### Correlation Analysis

Pairwise correlations between metrics:

- Utilization vs Temperature
- Utilization vs Power
- Utilization vs Memory
- Temperature vs Power
- Temperature vs Memory
- Power vs Memory

**Interpretation**:
- |r| > 0.7: Strong correlation
- |r| 0.4-0.7: Moderate
- |r| < 0.4: Weak

#### Example Insights:
```
GPU 2 Statistics:
Utilization:
├─ Mean: 68.3%, Median: 72.1%
├─ Std Dev: 12.4%
├─ Skewness: -0.3 (slight left skew)
└─ Outliers: 8 (2.1%)

Correlations:
├─ Util vs Temp: 0.85 (strong)
├─ Util vs Power: 0.91 (very strong)
├─ Temp vs Power: 0.88 (strong)
└─ Util vs Memory: 0.42 (moderate)
```

---

### 4. Predictive Analytics

**Forward-looking failure prediction and forecasting**

#### Failure Risk Prediction

Multi-factor risk assessment:

**Risk Factors Analyzed**:
1. **Health Trend**: Declining scores over time
   - Weight: +30 if slope < -0.5
   
2. **Acceleration**: Rapid deterioration
   - Weight: +20 if acceleration < -0.1
   
3. **Recent Health**: Last 5 measurements
   - Weight: +40 if avg < 50
   - Weight: +20 if avg < 70

**Risk Levels**:
- **Critical**: Score > 70 (immediate attention)
- **High**: Score 40-70 (monitor closely)
- **Moderate**: Score 20-40 (watch)
- **Low**: Score < 20 (healthy)

#### Metric Forecasting

Linear regression-based prediction:

- **Forecast Horizon**: Next 10 time steps
- **Confidence Score**: Based on residual variance
  - 100 - (σ_residual × 10)
  
- **Metrics Forecasted**:
  - Temperature (predict thermal runaway)
  - Utilization (predict workload changes)
  - Power (predict consumption)

#### Example Insights:
```
GPU 7 Predictions:
Failure Risk:
├─ Level: MODERATE
├─ Score: 35/100
└─ Factors:
    ├─ Declining health trend
    └─ Below-average health

Temperature Forecast:
├─ Next 10: 82°, 83°, 83°, 84°, 85°, 86°, 87°, 88°, 89°, 90°
├─ Confidence: 78%
└─ WARNING: Approaching critical threshold

Utilization Forecast:
├─ Trend: Stable
├─ Confidence: 92%
└─ Expected: 65-70% range
```

---

## 🎨 Dashboard Features

### Overview Tab
- **Cluster-wide metrics**:
  - Total anomalies across all GPUs
  - High-risk GPU count
  - Power inefficient GPUs
  - Thermal issues count

- **Per-node summaries**:
  - Warning/critical GPU counts
  - Top issues per node

### Temporal Analysis Tab
- Select node and GPU
- View all temporal metrics
- Trend indicators with visual cues
- Anomaly counts and locations
- Periodicity detection results

### Physics Models Tab
- Thermal dynamics dashboard
- Power efficiency analysis
- Workload classification
- Visual badges for stability/efficiency

### Correlations Tab
- Correlation matrices per GPU
- Color-coded strength indicators
- Identifies unusual relationships
- Helps diagnose issues

### Predictive Analytics Tab
- Failure risk assessment
- Forecast charts with confidence
- Risk factor breakdown
- Time-series predictions

---

## 🔧 Technical Details

### Algorithms Used

1. **Isolation Forest**: Unsupervised anomaly detection
   - Scikit-learn implementation
   - Contamination: 10% (configurable)

2. **FFT (Fast Fourier Transform)**: Periodicity detection
   - SciPy signal processing
   - Detrending before analysis

3. **Linear Regression**: Trend and forecasting
   - Scikit-learn LinearRegression
   - R² for fit quality

4. **Statistical Tests**:
   - Pearson correlation
   - IQR/Z-score outlier detection
   - Distribution moments

### Data Requirements

**Minimum for analysis**:
- 10 samples for anomaly detection
- 3 samples for trend analysis
- 20 samples for periodicity detection
- 5 samples for correlation analysis

**Recommended**:
- 100+ samples for robust statistics
- Consistent sampling intervals
- Minimal missing data

### Performance Considerations

- Analysis runs on-demand (not cached)
- Typical response: 0.5-2 seconds per GPU
- Memory usage: ~10MB per node
- CPU: Single-threaded (parallelizable)

---

## 📊 Interpretation Guide

### When to Take Action

#### 🔴 Critical (Immediate)
- Uncorrected memory errors > 0
- Temperature > 85°C
- Health score < 50
- Failure risk: Critical
- Thermal stability: Poor

#### 🟡 Warning (Monitor)
- Temperature 80-85°C
- Health score 50-80
- Failure risk: High/Moderate
- Power efficiency < 50
- High anomaly count (> 20)

#### 🟢 Healthy (Normal)
- Temperature < 80°C
- Health score > 80
- Failure risk: Low
- Good thermal stability
- Few anomalies (< 10)

### Common Patterns

**Pattern 1: Thermal Runaway**
```
Symptoms:
- Temperature trend: increasing
- Acceleration: positive
- Thermal volatility: high
- Forecast: exceeding limits

Action: Check cooling, reduce load
```

**Pattern 2: Underutilization**
```
Symptoms:
- Low average utilization (< 30%)
- High idle power consumption
- Poor power efficiency score

Action: Increase workload or investigate scheduling
```

**Pattern 3: Memory Bottleneck**
```
Symptoms:
- Workload type: memory_intensive
- Low compute intensity
- High memory intensity
- Power not scaling with compute

Action: Optimize memory access patterns
```

**Pattern 4: Periodic Throttling**
```
Symptoms:
- Strong periodicity detected
- Utilization oscillations
- Correlation spikes

Action: Check for thermal throttling or scheduling issues
```

---

## 🚀 Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

Includes:
- `scipy`: Scientific computing
- `scikit-learn`: Machine learning algorithms

### Run Server
```bash
python app.py
```

### Access Dashboards
- **Basic**: http://localhost:5000
- **Advanced**: http://localhost:5000/advanced.html

### API Endpoints

```python
# Advanced analytics for specific GPU
GET /api/advanced-analytics/<node>/<gpu_id>

# Cluster-wide summary
GET /api/cluster-advanced-summary

# Correlation analysis per node
GET /api/correlation-analysis/<node>

# Predictive analysis
GET /api/predictive-analysis/<node>/<gpu_id>
```

### Example API Call
```bash
curl http://localhost:5000/api/advanced-analytics/node1/0
```

Returns full temporal, physics, and statistical analysis for GPU 0 on node1.

---

## 🎓 Advanced Use Cases

### 1. Capacity Planning
- Analyze utilization trends
- Forecast future workloads
- Identify underutilized resources

### 2. Failure Prevention
- Track health degradation
- Predict failures before they occur
- Schedule preventive maintenance

### 3. Energy Optimization
- Identify inefficient GPUs
- Optimize power-performance tradeoffs
- Reduce idle power consumption

### 4. Thermal Management
- Monitor cooling effectiveness
- Detect thermal throttling
- Optimize datacenter airflow

### 5. Workload Optimization
- Match workload types to GPUs
- Balance compute/memory usage
- Improve scheduling efficiency

---

## 🔬 Research Applications

This analytics framework can be used for:

- **Academic Research**: Study GPU behavior patterns
- **Datacenter Operations**: Optimize large-scale clusters
- **Hardware Design**: Validate cooling/power systems
- **AI Training**: Monitor training stability
- **Performance Tuning**: Identify bottlenecks

---

## 📚 References

### Algorithms
- Isolation Forest: Liu et al. (2008)
- FFT: Cooley-Tukey algorithm
- Linear Regression: Ordinary Least Squares

### Libraries
- SciPy: Scientific computing
- Scikit-learn: Machine learning
- NumPy: Numerical computing
- Pandas: Data manipulation

---

## 🎯 Next Steps

Want to extend the analytics? Consider adding:

1. **Deep Learning Models**: LSTM for better forecasting
2. **Clustering**: Group similar GPU behaviors
3. **Anomaly Classification**: Categorize types of anomalies
4. **Root Cause Analysis**: Automated diagnostics
5. **Adaptive Thresholds**: Learn normal ranges per GPU
6. **Multi-Node Correlations**: Cross-node pattern detection
7. **Real-Time Alerts**: Webhook notifications
8. **Historical Comparison**: Compare to baselines

All analytics code is in `advanced_analytics.py` - fully customizable!

---

## 🐛 Troubleshooting

**Issue**: NaN values in results
- **Cause**: Insufficient data
- **Fix**: Ensure > 10 samples per metric

**Issue**: Low forecast confidence
- **Cause**: High variance in data
- **Fix**: Increase data collection frequency

**Issue**: No periodicity detected
- **Cause**: Not enough samples or no pattern
- **Fix**: Collect > 20 samples or pattern may not exist

**Issue**: Correlations all ~1.0
- **Cause**: Static/constant metrics
- **Fix**: Verify GPUs are actually working

---

Built with ❤️ for deep GPU cluster insights!
