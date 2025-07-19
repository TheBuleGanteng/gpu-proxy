"""
Enhanced logging setup for GPU proxy project.
Provides colored console output and detailed file logging.

Usage:
    from utils.logger import logger
    
    logger.info("GPU proxy started")
    logger.warning("Connection timeout detected")
    logger.error("Model failed to load")
"""

import logging
import os
from datetime import datetime
from typing import Optional, Union, Dict, Any
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset to default
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to the level name for console output
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.colored_levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message with the parent class
        formatted = super().format(record)
        return formatted


def setup_logging(
    log_file_path: Optional[str] = None,
    console_level: str = "DEBUG",
    file_level: str = "DEBUG",
    max_file_size_mb: int = 10,
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration with colored console output and file logging.
    
    Args:
        log_file_path: Path to the log file (if None, uses relative path from project root)
        console_level: Minimum level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_level: Minimum level for file output
        max_file_size_mb: Maximum size of log file before rotation (in MB)
        backup_count: Number of backup files to keep
    """
    
    # Use relative path if no specific path provided
    if log_file_path is None:
        # Get the project root (assuming logger.py is in src/utils/)
        current_file = Path(__file__)  # src/utils/logger.py
        project_root = current_file.parent.parent.parent  # Go up 3 levels to project root
        log_file_path_resolved = project_root / "logs" / "non-cron.log"
    else:
        log_file_path_resolved = Path(log_file_path)
    
    # Ensure log directory exists
    log_dir = log_file_path_resolved.parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Log directory created/verified: {log_dir}")
    except PermissionError as e:
        print(f"Permission denied creating log directory {log_dir}: {e}")
        # Fallback to /tmp
        log_file_path_resolved = Path("/tmp/gpu_proxy.log")
        log_file_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        print(f"Falling back to: {log_file_path_resolved}")
    except Exception as e:
        print(f"Failed to create log directory {log_dir}: {e}")
        # Fallback to /tmp
        log_file_path_resolved = Path("/tmp/gpu_proxy.log")
        try:
            log_file_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # /tmp should always exist
        print(f"Falling back to: {log_file_path_resolved}")
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper()))
    
    console_format = ColoredFormatter(
        fmt='%(asctime)s | %(colored_levelname)-8s | %(name)-20s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler with rotation and error handling
    file_handler = None
    try:
        from logging.handlers import RotatingFileHandler
        
        # Test if we can write to the file first
        log_file_path_resolved.touch()  # Create file if it doesn't exist
        
        file_handler = RotatingFileHandler(
            str(log_file_path_resolved),
            maxBytes=max_file_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, file_level.upper()))
        
        file_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        print(f"File handler created for: {log_file_path_resolved}")
        
    except Exception as e:
        print(f"Failed to create file handler for {log_file_path_resolved}: {e}")
        print("File logging will be disabled, only console logging available")
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    
    # Log the setup completion
    setup_logger = logging.getLogger(__name__)
    setup_logger.info(f"Logging initialized - Console: {console_level}, File: {file_level}")
    setup_logger.info(f"Log file: {log_file_path_resolved}")
    
    if file_handler:
        # Force a test write to ensure file logging works
        setup_logger.info("Test file write - if you see this in the file, logging is working!")
        file_handler.flush()  # Force immediate write
    else:
        setup_logger.warning("File logging not available - check permissions and path")
    
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the calling module.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        from utils.logger import logger
        logger.info("This is an info message")
    """
    if name is None:
        name = __name__
    
    # If logging hasn't been set up yet, do it now
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)


# Performance logging utilities for GPU operations
class GPUPerformanceLogger:
    """Utility class for logging GPU performance metrics and training progress."""
    
    def __init__(self, logger_name: str = "gpu_performance") -> None:
        self.logger = get_logger(logger_name)
    
    def log_training_batch(self, batch_num: int, total_batches: int, loss: float, 
                          accuracy: Optional[float] = None, duration: Optional[float] = None) -> None:
        """Log training batch results."""
        msg = f"Batch {batch_num}/{total_batches} - Loss: {loss:.4f}"
        
        if accuracy is not None:
            msg += f", Accuracy: {accuracy:.4f}"
        
        if duration is not None:
            msg += f", Duration: {duration:.3f}s"
        
        self.logger.info(msg)
    
    def log_gpu_memory(self, allocated: float, cached: float, total: float) -> None:
        """Log GPU memory usage."""
        self.logger.debug(f"GPU Memory - Allocated: {allocated:.2f}GB, "
                         f"Cached: {cached:.2f}GB, Total: {total:.2f}GB")
    
    def log_model_transfer(self, model_size: float, duration: float) -> None:
        """Log model transfer to/from GPU."""
        self.logger.info(f"Model transfer - Size: {model_size:.2f}MB, Duration: {duration:.3f}s")
    
    def log_serverless_call(self, call_type: str, duration: float, 
                           data_size: Optional[float] = None, success: bool = True) -> None:
        """Log serverless GPU calls."""
        status = "SUCCESS" if success else "FAILED"
        msg = f"Serverless {call_type} - {status}, Duration: {duration:.3f}s"
        
        if data_size is not None:
            msg += f", Data size: {data_size:.2f}MB"
        
        if success:
            self.logger.info(msg)
        else:
            self.logger.error(msg)


# Context manager for timing GPU operations
class TimedGPUOperation:
    """Context manager for timing and logging GPU operations."""
    
    def __init__(self, operation_name: str, logger_name: Optional[str] = None) -> None:
        self.operation_name = operation_name
        self.logger = get_logger(logger_name or __name__)
        self.start_time: Optional[datetime] = None
    
    def __enter__(self) -> 'TimedGPUOperation':
        self.start_time = datetime.now()
        self.logger.debug(f"running {self.operation_name} ... starting GPU operation")
        return self
    
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        if self.start_time is not None:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            if exc_type is None:
                self.logger.debug(f"running {self.operation_name} ... completed in {duration:.3f}s")
            else:
                self.logger.error(f"running {self.operation_name} ... failed after {duration:.3f}s: {exc_val}")
        else:
            self.logger.error(f"running {self.operation_name} ... timer was not properly initialized")


# Automatic logger setup - initialize logging when module is imported
# This ensures logging is set up once when the module is first imported
if not logging.getLogger().handlers:
    setup_logging()

# Create a default logger for this module
logger = get_logger(__name__)


# Example usage and testing
if __name__ == "__main__":
    # Test different log levels
    test_logger = get_logger(__name__)
    
    test_logger.debug("running test_logging ... testing debug message")
    test_logger.info("running test_logging ... testing info message")
    test_logger.warning("running test_logging ... testing warning message")
    test_logger.error("running test_logging ... testing error message")
    test_logger.critical("running test_logging ... testing critical message")
    
    # Test GPU performance logger
    gpu_logger = GPUPerformanceLogger()
    gpu_logger.log_training_batch(1, 100, 0.5234, 0.8567, 0.045)
    gpu_logger.log_serverless_call("forward_pass", 0.123, 15.6, True)
    
    # Test timed operation
    with TimedGPUOperation("model_loading"):
        import time
        time.sleep(0.1)  # Simulate work
    
    print("\nLogging setup complete! Check your console output and log file.")