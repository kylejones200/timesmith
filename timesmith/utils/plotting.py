"""Plotting utilities using plotsmith for time series visualization."""

import logging
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import plotsmith
try:
    import plotsmith as ps
    HAS_PLOTSMITH = True
except ImportError:
    HAS_PLOTSMITH = False
    ps = None
    logger.warning(
        "plotsmith not installed. Plotting functions will not be available. "
        "Install with: pip install plotsmith"
    )

# Export HAS_PLOTSMITH and all plotting functions
__all__ = [
    'HAS_PLOTSMITH',
    'plot_timeseries',
    'plot_forecast',
    'plot_residuals',
    'plot_multiple_series',
    'plot_autocorrelation',
    'plot_monte_carlo_paths',
]


def plot_timeseries(
    data: Union[pd.Series, pd.DataFrame],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs
):
    """Plot time series data using plotsmith.

    Args:
        data: Time series data (Series or DataFrame).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size (width, height).
        **kwargs: Additional arguments passed to plotsmith.

    Returns:
        Figure and axes objects.
    """
    if not HAS_PLOTSMITH:
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    import matplotlib.pyplot as plt
    
    # Try to use plotsmith's plot_timeseries if available
    try:
        if ps is not None and hasattr(ps, 'plot_timeseries'):
            if isinstance(data, pd.Series):
                data_for_plot = data.to_frame()
            else:
                data_for_plot = data
            fig, ax = ps.plot_timeseries(
                data_for_plot,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize,
                **kwargs
            )
            return fig, ax
    except (AttributeError, TypeError, Exception) as e:
        logger.debug(f"Plotsmith plot_timeseries not available, using matplotlib: {e}")
    
    # Fallback to matplotlib
    fig, ax = plt.subplots(figsize=figsize or (12, 6))
    if isinstance(data, pd.Series):
        ax.plot(data.index, data.values, **kwargs)
    else:
        for col in data.columns:
            ax.plot(data.index, data[col].values, label=col, **kwargs)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if isinstance(data, pd.DataFrame) and len(data.columns) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_forecast(
    historical: pd.Series,
    forecast: pd.Series,
    intervals: Optional[pd.DataFrame] = None,
    title: Optional[str] = "Forecast",
    **kwargs
):
    """Plot forecast with historical data and optional confidence intervals.

    Args:
        historical: Historical time series data.
        forecast: Forecasted values.
        intervals: Optional DataFrame with 'lower' and 'upper' columns for confidence intervals.
        title: Plot title.
        **kwargs: Additional arguments passed to plotsmith.

    Returns:
        Figure and axes objects.
    """
    if not HAS_PLOTSMITH:
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    import matplotlib.pyplot as plt
    
    # Try to use plotsmith if available
    try:
        if ps is not None and hasattr(ps, 'plot_timeseries'):
            combined = pd.concat([historical, forecast])
            fig, ax = ps.plot_timeseries(
                combined,
                title=title,
                **kwargs
            )
        else:
            raise AttributeError("Plotsmith plot_timeseries not available")
    except (AttributeError, TypeError, Exception) as e:
        logger.debug(f"Plotsmith not available, using matplotlib: {e}")
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical.index, historical.values, label='Historical', linewidth=2)
        ax.plot(forecast.index, forecast.values, label='Forecast', linewidth=2, linestyle='--')
        if title:
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Add forecast start line
    ax.axvline(historical.index[-1], color='red', linestyle=':', alpha=0.7, label='Forecast Start')

    # Add confidence intervals if provided
    if intervals is not None:
        ax.fill_between(
            forecast.index,
            intervals['lower'].values,
            intervals['upper'].values,
            alpha=0.3,
            label='Confidence Interval'
        )

    ax.legend()
    return fig, ax


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    plot_type: str = 'scatter',
    title: Optional[str] = "Residuals",
    **kwargs
):
    """Plot residuals using plotsmith.

    Args:
        actual: Actual values.
        predicted: Predicted values.
        plot_type: Type of plot ('scatter', 'line', 'histogram').
        title: Plot title.
        **kwargs: Additional arguments passed to plotsmith.

    Returns:
        Figure and axes objects.
    """
    if not HAS_PLOTSMITH:
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    residuals = actual - predicted
    import matplotlib.pyplot as plt
    
    # Try to use plotsmith's plot_residuals if available
    try:
        if ps is not None and hasattr(ps, 'plot_residuals'):
            fig, ax = ps.plot_residuals(
                actual,
                predicted,
                plot_type=plot_type,
                title=title,
                **kwargs
            )
            return fig, ax
    except (AttributeError, TypeError, Exception) as e:
        logger.debug(f"Plotsmith plot_residuals not available, using matplotlib: {e}")
    
    # Fallback to matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_type == 'scatter':
        ax.scatter(actual, residuals, **kwargs)
    elif plot_type == 'line':
        ax.plot(actual, residuals, **kwargs)
    elif plot_type == 'histogram':
        ax.hist(residuals, bins=30, **kwargs)
    ax.set_title(title)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Residuals')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_multiple_series(
    series_dict: dict,
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs
):
    """Plot multiple time series on the same plot.

    Args:
        series_dict: Dictionary mapping labels to Series/DataFrame.
        title: Plot title.
        figsize: Figure size (width, height).
        **kwargs: Additional arguments passed to plotsmith.

    Returns:
        Figure and axes objects.
    """
    if not HAS_PLOTSMITH:
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    import matplotlib.pyplot as plt
    
    # Try to use plotsmith if available
    try:
        if ps is not None and hasattr(ps, 'plot_timeseries'):
            # Combine all series into a DataFrame
            combined = pd.DataFrame(series_dict)
            fig, ax = ps.plot_timeseries(
                combined,
                title=title,
                figsize=figsize,
                **kwargs
            )
            return fig, ax
    except (AttributeError, TypeError, Exception) as e:
        logger.debug(f"Plotsmith plot_timeseries not available, using matplotlib: {e}")
    
    # Fallback to matplotlib
    fig, ax = plt.subplots(figsize=figsize or (12, 6))
    for label, series in series_dict.items():
        ax.plot(series.index, series.values, label=label, **kwargs)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_autocorrelation(
    acf_values: np.ndarray,
    pacf_values: Optional[np.ndarray] = None,
    max_lag: Optional[int] = None,
    title: Optional[str] = "Autocorrelation",
    **kwargs
):
    """Plot autocorrelation and partial autocorrelation functions.

    Args:
        acf_values: ACF values.
        pacf_values: Optional PACF values.
        max_lag: Maximum lag to display.
        title: Plot title.
        **kwargs: Additional arguments passed to plotsmith.

    Returns:
        Figure and axes objects.
    """
    if not HAS_PLOTSMITH:
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    lags = np.arange(len(acf_values))
    if max_lag is not None:
        lags = lags[:max_lag]
        acf_values = acf_values[:max_lag]
        if pacf_values is not None:
            pacf_values = pacf_values[:max_lag]

    # Use matplotlib for ACF/PACF plots (bar charts)
    import matplotlib.pyplot as plt
    
    if pacf_values is not None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].bar(lags, acf_values)
        axes[0].set_title(f'{title} - ACF')
        axes[0].set_ylabel('ACF')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].bar(lags, pacf_values)
        axes[1].set_title(f'{title} - PACF')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('PACF')
        axes[1].grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(lags, acf_values)
        ax.set_title(title)
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.grid(True, alpha=0.3)
        axes = [ax]

    plt.tight_layout()
    return fig, axes


def plot_monte_carlo_paths(
    paths: np.ndarray,
    title: Optional[str] = "Monte Carlo Simulation",
    show_mean: bool = True,
    show_percentiles: bool = True,
    **kwargs
):
    """Plot Monte Carlo simulation paths using plotsmith.

    Args:
        paths: Array of shape (n_steps, n_simulations) with simulation paths.
        title: Plot title.
        show_mean: Whether to show mean path.
        show_percentiles: Whether to show percentile bands.
        **kwargs: Additional arguments passed to plotsmith.

    Returns:
        Figure and axes objects.
    """
    if not HAS_PLOTSMITH:
        raise ImportError(
            "plotsmith is required for plotting. Install with: pip install plotsmith"
        )

    import matplotlib.pyplot as plt
    
    # Ensure paths is 2D: (n_steps, n_simulations)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 1)
    
    n_steps, n_simulations = paths.shape
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual paths
    alpha = 0.3 if n_simulations > 10 else 0.7
    for i in range(min(n_simulations, 100)):  # Limit to 100 paths for performance
        ax.plot(paths[:, i], color='gray', alpha=alpha, linewidth=0.5)

    if show_mean:
        mean_path = paths.mean(axis=1)
        ax.plot(mean_path, color='black', linewidth=2, label='Mean Path')

    if show_percentiles:
        lower = np.percentile(paths, 2.5, axis=1)
        upper = np.percentile(paths, 97.5, axis=1)
        ax.fill_between(
            np.arange(n_steps),
            lower,
            upper,
            alpha=0.2,
            label='95% Confidence Interval'
        )

    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Simulated Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax

