# src/utils.py

import logging

def setup_logging(verbose: bool):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
