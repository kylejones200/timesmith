"""Command-line interface for TimeSmith."""

import argparse
import sys
from typing import Optional

from timesmith import __version__


def print_version() -> None:
    """Print TimeSmith version information."""
    print(f"TimeSmith {__version__}")
    print("A time series machine learning library with strict layer boundaries.")


def print_info() -> None:
    """Print information about TimeSmith."""
    print_version()
    print("\nFor more information, visit:")
    print("  - Documentation: https://timesmith.readthedocs.io/")
    print("  - GitHub: https://github.com/kylejones200/timesmith")
    print("\nQuick start:")
    print("  >>> import timesmith")
    print("  >>> from timesmith import ForecastTask, backtest_forecaster")


def main(args: Optional[list] = None) -> int:
    """Main entry point for TimeSmith CLI.

    Args:
        args: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = argparse.ArgumentParser(
        prog="timesmith",
        description="TimeSmith: A time series machine learning library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"TimeSmith {__version__}",
        help="Show version information and exit",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show information about TimeSmith",
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.info:
        print_info()
        return 0

    # Default: show version
    print_version()
    return 0


if __name__ == "__main__":
    sys.exit(main())
