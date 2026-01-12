"""Logging configuration for TimeSmith.

This module provides centralized logging configuration for the TimeSmith library.
Users can configure logging levels and formats through environment variables
or by calling configure_logging() directly.
"""

import logging
import os
import sys
from typing import Optional


def configure_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    stream: Optional[object] = None,
) -> None:
    """Configure logging for TimeSmith.

    This function sets up logging for the TimeSmith package. It can be called
    explicitly or will use environment variables if available.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, reads from TIMESMITH_LOG_LEVEL environment variable,
               or defaults to WARNING.
        format_string: Log format string. If None, uses a default format.
        stream: Stream to write logs to. If None, uses stderr.

    Environment Variables:
        TIMESMITH_LOG_LEVEL: Set default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        TIMESMITH_LOG_FORMAT: Set log format (simple, detailed, json)
    """
    # Get level from parameter, environment, or default
    if level is None:
        level = os.getenv("TIMESMITH_LOG_LEVEL", "WARNING").upper()
    else:
        level = level.upper()

    # Validate level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level not in valid_levels:
        raise ValueError(
            f"Invalid log level: {level}. Must be one of {valid_levels}"
        )

    log_level = getattr(logging, level)

    # Get format from parameter, environment, or default
    if format_string is None:
        format_pref = os.getenv("TIMESMITH_LOG_FORMAT", "simple").lower()
        if format_pref == "json":
            # JSON format for structured logging (if needed in future)
            format_string = (
                '{"time": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            )
        elif format_pref == "detailed":
            format_string = (
                "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(message)s"
            )
        else:  # simple (default)
            format_string = "%(levelname)s: %(name)s - %(message)s"
    else:
        format_string = format_string

    # Configure root logger for timesmith package
    logger = logging.getLogger("timesmith")
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create console handler
    if stream is None:
        stream = sys.stderr

    handler = logging.StreamHandler(stream)
    handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    This is a convenience function that ensures logging is configured
    before returning a logger.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        Logger instance for the specified name.
    """
    # Configure if not already done (one-time setup)
    logger = logging.getLogger(name)
    if not logger.handlers and name.startswith("timesmith"):
        # Only auto-configure for timesmith loggers
        configure_logging()
    return logger


# Auto-configure on import if environment variable is set
if os.getenv("TIMESMITH_LOG_LEVEL"):
    configure_logging()

