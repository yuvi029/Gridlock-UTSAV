# 🚦 Gridlock – Predicting Urban Traffic Demand

> A Spatio-Temporal Momentum Architecture for Intelligent Traffic Demand Forecasting

![Hackathon](https://img.shields.io/badge/Hackathon-Gridlock%202.0-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Overview

Urban traffic congestion remains one of the most significant barriers to economic productivity and efficient transportation systems. Modern smart cities require AI-driven forecasting systems capable of understanding complex mobility patterns and predicting transportation demand in real time.

**Gridlock** is an advanced traffic demand prediction framework developed during **Gridlock Hackathon 2.0**. The system combines geospatial intelligence, temporal momentum modeling, ensemble learning, and automated optimization to accurately forecast urban traffic demand across multiple city regions.

### Key Highlights

- 🌍 Geospatial Intelligence
- ⏱ Temporal Momentum Modeling
- 🚦 Traffic Flow Dynamics
- 🌦 Environmental Impact Analysis
- 🤖 Ensemble Machine Learning
- 📊 Bayesian Hyperparameter Optimization

---

## 🎯 Problem Statement

Traffic congestion is a major challenge in modern urban environments. Efficient transportation systems require accurate forecasting of traffic demand to optimize infrastructure, reduce delays, and improve mobility.

The objective is to build an intelligent system capable of predicting transportation demand using:

- Passenger travel patterns
- Booking behavior
- Trip cancellations
- Road infrastructure data
- Weather conditions
- Temporal traffic dynamics

---

## 📊 Dataset Overview

| Property | Training Data | Test Data |
|-----------|-------------|-----------|
| Shape | 77,299 × 11 | 41,778 × 10 |
| Days | 48 & 49 | Day 49 |
| Unique Geohashes | 1,249 | 1,190 |

### Missing Value Handling

| Feature | Missing Values | Imputation Strategy |
|----------|--------------|---------------------|
| RoadType | 924 | Geohash Mode |
| Temperature | 3,844 | Time-Geohash Mean |
| Weather | 1,228 | Temporal Mode |

---

## 🔍 Exploratory Data Analysis

### Traffic Demand Distribution

- Demand values are bounded between **0 and 1**
- Highly right-skewed distribution
- Most locations operate at low demand
- Few locations experience extreme surges

### Important Finding

A logarithmic transformation (`log1p`) significantly reduced performance by compressing meaningful variation.

Therefore:

```python
predictions = np.clip(predictions, 0.0, 1.0)
```

Models were trained on the raw target scale.

---

### ⏰ Diurnal Traffic Patterns

Traffic follows cyclical behavior:

- Morning Rush Hours
- Evening Rush Hours
- Night-Time Decline

To preserve temporal continuity:

```python
hour_sin
hour_cos
minute_sin
minute_cos
```

were engineered as cyclical time features.

---

### 🌦 Infrastructure & Environmental Impact

Key findings:

- Highways carry significantly more demand than residential roads.
- Weather conditions strongly affect traffic capacity.
- Road type and lane count influence throughput.

---

### 📈 Historical Demand Correlation

Strong correlation observed between:

```text
Day 48 Demand
       ↕
Day 49 Demand
```

This demonstrates that historical traffic patterns are strong predictors of future traffic demand.

---

# ⚙ Feature Engineering Pipeline

## 🌍 Geospatial Decoding

### Geohash Translation

Geohashes were decoded into:

```python
latitude
longitude
```

allowing the model to learn spatial relationships directly.

### Spatial Centrality

Created:

```python
dist_to_center
```

to measure distance from the city center.

---

## ⏱ Cyclical Temporal Features

Generated:

```python
hour_sin
hour_cos
minute_sin
minute_cos
```

Additional flags:

```python
is_rush_hour
is_night
```

---

## 🚦 Infrastructure Capacity Modeling

Interaction Features:

```python
road_type * number_of_lanes
```

This estimates road throughput capacity.

---

## 🌦 Environmental Strain Modeling

Interaction Features:

```python
temperature * weather
```

Used to model environmental effects on traffic flow.

---

## 🤖 Unsupervised Spatial Learning

### K-Means Clustering

Grouped traffic locations into:

```python
10 traffic zones
```

Benefits:

- Reduced spatial noise
- Captured regional traffic behavior
- Improved model generalization

---

### Principal Component Analysis (PCA)

Applied PCA to:

- Extract dominant traffic movement directions
- Reduce feature redundancy
- Improve model efficiency

---

## 📈 Rolling Window Momentum

### Rolling Mean

```python
rolling_mean_3
```

Captures short-term traffic buildup.

### Rolling Standard Deviation

```python
rolling_std_3
```

Measures traffic volatility.

---

# 🧠 Model Architecture

The final solution uses an ensemble of three gradient boosting models.

---

## 🚀 LightGBM

### Strengths

- Leaf-wise tree growth
- Learns complex nonlinear relationships
- Captures localized traffic surges

### Role

High-frequency traffic fluctuation detector.

---

## 🐱 CatBoost

### Strengths

- Native categorical handling
- Symmetric tree architecture
- Reduced leakage risk

### Role

Strong performance on:

- RoadType
- Weather
- Categorical features

---

## ⚡ XGBoost

### Strengths

- Conservative depth-wise growth
- Noise resistant
- Stable learning process

### Role

Acts as the structural backbone of the ensemble.

---

# 🔒 Validation Strategy

## Why Not K-Fold?

Random K-Fold causes temporal leakage.

Example:

```text
Train on Day 49
Predict Day 48
```

This leads to unrealistic validation performance.

---

## Time Series Cross Validation

Implemented:

```python
TimeSeriesSplit(n_splits=3)
```

Benefits:

- Preserves chronology
- Prevents leakage
- Mimics real-world forecasting

---

# 🎯 Hyperparameter Optimization

## Optuna Bayesian Search

Each model underwent:

```text
40 Optimization Trials
```

Optimized Parameters:

- Learning Rate
- Tree Depth
- L2 Regularization
- Feature Subsampling

Objective:

```text
Minimize RMSE
```

---

# 🏆 Dynamic Ensemble Weighting

Weights assigned automatically:

```python
weight = 1 / rmse
```

Advantages:

- Merit-based contribution
- Better generalization
- Fully automated weighting

---

# 🚀 Final Inference Pipeline

### Multi-Seed Training

Seeds Used:

```python
[42, 123, 456, 789, 2024]
```

Total Models:

```python
3 Algorithms × 5 Seeds = 15 Models
```

Benefits:

- Reduced variance
- Increased stability
- More robust predictions

---

## Production Safety Checks

```python
assert no_null_values
assert no_negative_predictions
```

Final outputs bounded:

```python
np.clip(predictions, 0.0, 1.0)
```

---

# 🛠 Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- CatBoost
- XGBoost
- Optuna
- PCA
- K-Means Clustering

---

# 📈 Key Innovations

✅ Geohash Spatial Decoding

✅ Cyclic Time Embeddings

✅ Rolling Momentum Features

✅ Environmental Strain Modeling

✅ K-Means Traffic Zone Learning

✅ PCA-Based Spatial Compression

✅ Bayesian Hyperparameter Optimization

✅ Dynamic Ensemble Weighting

✅ Multi-Seed Variance Reduction

---

# 👥 Team – The Learning Machines

| Name |
|--------|
| Yuvraj Singh |
| Utsav Shresth |
| Shristi Shreya |
| Shubhangi |

---

# 🌟 Vision

> Transforming urban mobility through AI-powered traffic intelligence and spatio-temporal forecasting.

---

### Built for Gridlock Hackathon 2.0 🚦