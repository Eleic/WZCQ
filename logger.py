"""
Logging system for the WZCQ project.
This module provides a unified way to log messages across the project.
"""

import os
import logging
import datetime
from pathlib import Path
from config_manager import config

class Logger:
    """Manages logging for the WZCQ project."""
    
    LEVELS = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    def __init__(self, name="wzcq", level="info"):
        """Initialize the logger."""
        self.name = name
        self.level = self.LEVELS.get(level.lower(), logging.INFO)
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Set up the logger with file and console handlers."""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Clear existing handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Create logs directory if it doesn't exist
        logs_dir = config.get("paths", "logs_dir")
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(logs_dir, f"{self.name}_{today}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def debug(self, message):
        """Log a debug message."""
        self.logger.debug(message)
    
    def info(self, message):
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error message."""
        self.logger.error(message)
    
    def critical(self, message):
        """Log a critical message."""
        self.logger.critical(message)
    
    def exception(self, message):
        """Log an exception message with traceback."""
        self.logger.exception(message)

# Create a singleton instance
logger = Logger()