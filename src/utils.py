# src/utils.py

import os
import io
import logging
import torch
import time
import cProfile
import pstats
from getpass import getpass
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        pr.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        logging.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        logging.debug(f"Profiling results for {func.__name__}:\n{s.getvalue()}")
        
        return result
    return wrapper

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    
    # File handler with rotation
    file_handler = RotatingFileHandler('autoquant.log', maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(log_formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

def get_device(device: str):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def get_api_key(key_name: str):
    load_dotenv()
    api_key = os.getenv(key_name)
    if not api_key:
        api_key = getpass(f"Enter your {key_name}: ")
        with open(".env", "a") as f:
            f.write(f"\n{key_name}={api_key}")
    return api_key