import logging
import os
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler

# Environment configuration
ENV = os.getenv("ENVIRONMENT", "development")
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

# Log levels by environment
LOG_LEVELS = {
    "development": logging.DEBUG,
    "production": logging.INFO,
    "testing": logging.WARNING
}
LOG_LEVEL = LOG_LEVELS.get(ENV, logging.INFO)

# Formatters
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
SIMPLE_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def cleanup_old_logs(days_to_keep: int = 30) -> None:
    """
    Delete log files older than specified days.
    
    Args:
        days_to_keep: Number of days to retain logs (default: 30)
    """
    if not os.path.exists(LOGS_DIR):
        return
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    for filename in os.listdir(LOGS_DIR):
        if filename.startswith("log_") and filename.endswith(".log"):
            filepath = os.path.join(LOGS_DIR, filename)
            try:
                file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_modified < cutoff_date:
                    os.remove(filepath)
            except OSError as e:
                print(f"Error deleting {filename}: {e}")

# Clean up old logs on module import
cleanup_old_logs(days_to_keep=30)

# File handler with rotation
file_handler = TimedRotatingFileHandler(
    LOG_FILE,
    when='midnight',
    interval=1,
    backupCount=30,
    encoding='utf-8'
)
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(
    logging.Formatter(DETAILED_FORMAT if ENV == "development" else SIMPLE_FORMAT)
)

# Handlers list
handlers = [file_handler]

# Add console handler in development
if ENV == "development":
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
    handlers.append(console_handler)

# Configure root logger
logging.basicConfig(
    level=LOG_LEVEL,
    handlers=handlers
)

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    return logger