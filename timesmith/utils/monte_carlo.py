"""Monte Carlo simulation utilities for time series."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
        from timesmith.utils.plotting import plot_monte_carlo_paths
        import matplotlib.pyplot as plt
        
        fig, ax = plot_monte_carlo_paths(
            prices,
            title=title,
            show_mean=True,
            show_percentiles=True
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

