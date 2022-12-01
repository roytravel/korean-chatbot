import sys
import logging

def get_logger(name : str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(actime)s] %(message)s"))
        logger.addHandler(handler)
    return logger