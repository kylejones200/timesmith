"""Monte Carlo simulation utilities for time series."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from timesmith.typing import SeriesLike

logger = logging.getLogger(__name__)

# Optional scipy for norm.ppf
try:
    from scipy.stats import norm

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    norm = None
    logger.warning(
        "scipy not installed. black_scholes_monte_carlo will use numpy fallback. "
        "Install with: pip install scipy or pip install timesmith[scipy]"
    )

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def monte_carlo_simulation(
    initial_value: float,
    mu: float,
    sigma: float,
    days: int,
    simulations: int = 1000,
) -> np.ndarray:
    """Perform a Monte Carlo simulation for time series forecasting.

    Uses geometric Brownian motion model.

    Args:
        initial_value: Initial value of the series.
        mu: Drift parameter (mean return).
        sigma: Volatility parameter (standard deviation).
        days: Number of days to simulate.
        simulations: Number of simulation paths.

    Returns:
        Array of shape (days, simulations) with simulated paths.
    """
    dt = 1  # Daily step size
    prices = np.zeros((days, simulations))
    prices[0] = initial_value

    for t in range(1, days):
        random_shock = np.random.normal(
            loc=mu * dt, scale=sigma * np.sqrt(dt), size=simulations
        )
        prices[t] = prices[t - 1] * np.exp(random_shock)

    return prices


def black_scholes_monte_carlo(
    historical_data: SeriesLike,
    forecast_days: int,
    n_simulations: int = 1000,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Black-Scholes Monte Carlo simulation for asset price forecasting.

    Estimates drift and volatility from historical log returns and simulates
    future price paths using geometric Brownian motion. This is the standard
    approach used in option pricing and financial forecasting.

    The model assumes:
    - Log returns are normally distributed
    - Market is efficient (random walk)
    - Geometric Brownian motion: ΔS = S * (μΔt + σε√Δt)

    Args:
        historical_data: Historical price series (pandas Series with datetime index).
        forecast_days: Number of days to forecast.
        n_simulations: Number of simulation paths to generate.
        random_state: Random seed for reproducibility.

    Returns:
        Array of shape (forecast_days, n_simulations) with simulated price paths.
        Each column represents one simulation path.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
        >>> prices = pd.Series(100 + np.random.randn(100).cumsum(), index=dates)
        >>> paths = black_scholes_monte_carlo(prices, forecast_days=30, n_simulations=1000)
        >>> print(paths.shape)  # (30, 1000)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Convert to Series if needed
    if isinstance(historical_data, pd.DataFrame):
        if historical_data.shape[1] == 1:
            historical_data = historical_data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have exactly one column")

    # Ensure we have a pandas Series with numeric values
    if not isinstance(historical_data, pd.Series):
        historical_data = pd.Series(historical_data)

    # Calculate log returns: log(1 + pct_change) = log(S_t / S_{t-1})
    log_returns = np.log(1 + historical_data.pct_change()).dropna()

    if len(log_returns) < 2:
        raise ValueError("Need at least 2 data points to calculate returns")

    # Estimate parameters from historical data
    u = log_returns.mean()  # Mean log return
    var = log_returns.var()  # Variance of log returns
    drift = u - (0.5 * var)  # Adjusted drift for geometric Brownian motion
    stdev = log_returns.std()  # Standard deviation (volatility)

    # Get initial price (last observed price)
    S0 = historical_data.iloc[-1]

    # Generate random shocks for all simulations and all time steps
    # Using inverse normal CDF (ppf) for better distribution properties if scipy available
    if HAS_SCIPY and norm is not None:
        random_shocks = norm.ppf(np.random.rand(forecast_days, n_simulations))
    else:
        # Fallback: use numpy's inverse normal approximation
        # numpy doesn't have ppf, so we use Box-Muller transform approximation
        uniform = np.random.rand(forecast_days, n_simulations)
        # Approximate inverse normal using erf inverse approximation
        random_shocks = np.sqrt(2) * np.sign(uniform - 0.5) * np.sqrt(
            -np.log(1 - np.abs(2 * uniform - 1))
        )

    # Calculate daily returns: exp(drift + stdev * random_shock)
    daily_returns = np.exp(drift + stdev * random_shocks)

    # Initialize price matrix
    price_paths = np.zeros((forecast_days, n_simulations))
    price_paths[0] = S0

    # Apply Monte Carlo simulation: S_t = S_{t-1} * daily_return_t
    for t in range(1, forecast_days):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]

    return price_paths


def plot_monte_carlo(
    prices: np.ndarray,
    title: str = "Monte Carlo Simulation",
    save_path: Optional[str] = None,
) -> None:
    """Plot the Monte Carlo simulation results using plotsmith.

    Args:
        prices: Array of simulated paths from monte_carlo_simulation.
        title: Plot title.
        save_path: Optional path to save the plot.

    Raises:
        ImportError: If plotsmith is not installed.
    """
    try:
        import matplotlib.pyplot as plt

        from timesmith.utils.plotting import plot_monte_carlo_paths

        fig, ax = plot_monte_carlo_paths(
            prices, title=title, show_mean=True, show_percentiles=True
        )

        if save_path:
            fig.savefig(save_path)
        else:
            plt.show()

        plt.close(fig)
    except ImportError:
        raise ImportError(
            "plotsmith is required for plot_monte_carlo. "
            "Install with: pip install plotsmith"
        )
