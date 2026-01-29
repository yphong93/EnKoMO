"""
Logging utilities for experiments
"""
import logging
import sys
import os
from datetime import datetime

def setup_logging(save_dir, log_filename='experiment.log'):
    """
    Setup logging to both console and file
    
    Args:
        save_dir: Directory to save log file
        log_filename: Name of the log file
    
    Returns:
        logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('EnKoMa_Experiment')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - detailed logging
    log_file_path = os.path.join(save_dir, log_filename)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - simpler format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Log start message
    logger.info("="*80)
    logger.info(f"Experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file_path}")
    logger.info("="*80)
    
    return logger

class LoggerWriter:
    """
    Wrapper to redirect stdout/stderr to logger
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ''
    
    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())
    
    def flush(self):
        pass

