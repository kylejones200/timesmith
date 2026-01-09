# TimeSmith Example Notebooks

This directory contains comprehensive Jupyter notebook examples demonstrating the TimeSmith library.

## Notebook Index

### 01_getting_started.ipynb
**Getting Started with TimeSmith**
- Introduction to TimeSmith architecture
- Scientific types (`SeriesLike`, `PanelLike`, `TableLike`)
- Basic forecasting example
- Basic transformation example

### 02_data_loading_preprocessing.ipynb
**Data Loading and Preprocessing**
- Loading and validating time series data
- Handling missing dates and values
- Resampling time series
- Outlier detection and removal
- Splitting data into train/test sets

### 03_feature_engineering.ipynb
**Feature Engineering**
- Lag features
- Rolling window statistics
- Time-based features (hour, day of week, month, etc.)
- Differencing
- Seasonal features
- Feature unions (combining multiple featurizers)

### 04_forecasting.ipynb
**Forecasting Methods**
- Simple Moving Average
- Exponential Smoothing
- ARIMA (if available)
- Monte Carlo forecasting with uncertainty
- Comparing multiple forecasting methods

### 05_anomaly_detection.ipynb
**Anomaly Detection**
- Creating data with anomalies
- Z-Score based detection
- Hampel filter
- Visualization of detected anomalies

### 06_changepoint_detection.ipynb
**Change Point Detection**
- Creating data with change points
- PELT (Pruned Exact Linear Time) detector
- CUSUM (Cumulative Sum) detector
- Visualizing detected change points

### 07_network_analysis.ipynb
**Network Analysis for Time Series**
- Horizontal Visibility Graph (HVG)
- Natural Visibility Graph (NVG)
- Recurrence Networks
- Transfer Entropy for causal inference
- Windowed network analysis
- Multiscale graph analysis

### 08_evaluation_backtesting.ipynb
**Evaluation and Backtesting**
- Creating forecast tasks
- Running backtests
- Summarizing backtest results
- Expanding window splits
- Sliding window splits
- Model comparison

### 09_pipelines_composition.ipynb
**Pipelines and Composition**
- Simple pipelines (chaining transformers)
- Forecaster pipelines (preprocessing + forecasting)
- Feature unions (combining featurizers)
- Adapters (converting between scitypes)

### 10_advanced_topics.ipynb
**Advanced Topics**
- Numba JIT performance optimization
- Time series decomposition
- Trend detection
- Seasonality detection
- Autocorrelation analysis (ACF/PACF)
- Monte Carlo simulation

## Running the Notebooks

### Prerequisites

```bash
# Install TimeSmith with all optional dependencies
pip install timesmith[all]

# Or install specific extras
pip install timesmith[network,forecasters,dev]
```

### Launch Jupyter

```bash
# Install Jupyter if needed
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Then navigate to `examples/notebooks/` and open any notebook.

## Notebook Structure

Each notebook follows a consistent structure:
1. **Introduction**: Overview of the topic
2. **Setup**: Imports and data preparation
3. **Examples**: Step-by-step code examples
4. **Visualizations**: Plots and charts where applicable
5. **Summary**: Key takeaways

## Tips

- Start with `01_getting_started.ipynb` if you're new to TimeSmith
- Notebooks are designed to be run sequentially, but can also be used independently
- Some features require optional dependencies (e.g., `ruptures` for change point detection)
- Check the error messages - they'll guide you to install missing dependencies

## Contributing

If you create additional example notebooks, please:
1. Follow the naming convention: `NN_topic_name.ipynb`
2. Include markdown cells explaining concepts
3. Add visualizations where helpful
4. Handle optional dependencies gracefully with try/except blocks
5. Update this README with a brief description

## Questions?

For issues or questions, please open an issue on the [TimeSmith GitHub repository](https://github.com/kylejones200/timesmith).

