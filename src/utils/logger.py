"""Logging utilities for Ethereal DN Trader."""

import logging
import sys
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console

# Global logger registry
_loggers: dict[str, logging.Logger] = {}
_console: Optional[Console] = None


def get_console() -> Console:
    """Get or create the global Rich console."""
    global _console
    if _console is None:
        _console = Console()
    return _console


def setup_logger(
    name: str = "ethereal",
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger with Rich formatting.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging to file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Prevent duplicate logs from parent loggers
    logger.propagate = False

    # Rich console handler
    console_handler = RichHandler(
        console=get_console(),
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "ethereal") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)
