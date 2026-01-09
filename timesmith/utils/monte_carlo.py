"""Monte Carlo simulation utilities for time series."""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    """Plot the Monte Carlo simulation results.

    Args:
        prices: Array of simulated paths from monte_carlo_simulation.
        title: Plot title.
        save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(prices, color="gray", alpha=0.3)
    plt.plot(prices.mean(axis=1), color="black", linewidth=2, label="Mean Path")
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Simulated Value")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

