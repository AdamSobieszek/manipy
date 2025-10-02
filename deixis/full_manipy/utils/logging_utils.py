# utils/logging_utils.py
import logging
import sys

def setup_logger(name='TrainingLog', level=logging.INFO):
    """Sets up a basic logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Example usage:
# logger = setup_logger()
# logger.info("This is an info message.")
