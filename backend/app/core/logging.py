"""Application logging configuration for VisionGuard.

Establishes structured console logging without disclosing secrets,
raw pixel data, or private user file paths.
"""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure and return root/application logger.

    Args:
        log_level: Desired log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logger for the visionguard package.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger("visionguard")
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers when setup_logging is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Also set default formatting on the root logger if not already configured
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logger
