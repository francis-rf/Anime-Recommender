📄
setup.py

- Complete Breakdown
  WHAT is this file?
  This is a Python package configuration file that makes your project installable as a Python package. It uses setuptools to define how your project should be packaged and distributed.

WHY do we need it?
Makes the project installable - Allows you to run pip install -e . to install your project in development mode
Manages dependencies - Automatically installs all required packages from
requirements.txt
Enables imports - After installation, you can import modules from anywhere (e.g., from src.module import function)
Professional structure - Standard practice for Python projects
Deployment ready - Makes it easier to deploy and distribute your package
WHEN is it used?
During development: When you run pip install -e . (editable/development mode)
During deployment: When installing the package in production
In Docker builds: The Dockerfile likely runs this to install the package
In CI/CD pipelines: Automated testing/deployment systems use it
WHERE does it fit in the project?
Location: Root directory of the project
Scope: Affects the entire project structure
Related files:
Works with
requirements.txt
(line 3-4)
Creates ANIME_RECOMMENDER.egg-info/ directory (you have this!)
HOW does it work?
Line-by-Line Explanation:
python

# Line 1: Import necessary functions

from setuptools import setup, find_packages
setup - Main function to configure the package
find_packages() - Automatically discovers all Python packages (folders with **init**.py)
python

# Lines 3-4: Read dependencies from requirements.txt

with open("requirements.txt") as f:
requirements = f.read().splitlines()
Opens
requirements.txt
Reads all lines and removes newline characters
Stores dependencies in a list
python

# Lines 6-12: Configure the package

setup(
name="ANIME-RECOMMENDER", # Package name
version="0.1", # Version number
author="Francis", # Your name
packages=find_packages(), # Auto-find all packages
install_requires = requirements, # Dependencies to install
)
HOW to use it?
Installation Commands:
bash

# Development mode (editable install)

pip install -e .

# Regular installation

pip install .

# With specific requirements

pip install -r requirements.txt
Current Configuration Analysis:
Parameter Value Purpose
name "ANIME-RECOMMENDER" Package identifier
version "0.1" Current version (early development)
author "Francis" Package creator
packages find_packages() Auto-discovers: app/, src/, pipeline/, utils/
install_requires From requirements.txt Auto-installs dependencies
What happens when you run pip install -e .?
✅ Reads
setup.py
✅ Finds all packages (src, app, pipeline, utils)
✅ Reads
requirements.txt
and installs all dependencies
✅ Creates ANIME_RECOMMENDER.egg-info/ (metadata directory)
✅ Makes your code importable from anywhere
Example: After installation, you can do:

python
from src.some_module import some_function
from utils.helper import helper_function
Summary:
This
setup.py
is a simple but effective configuration that:

✅ Packages your anime recommender as an installable Python package
✅ Automatically manages dependencies
✅ Enables clean imports across your project
✅ Follows Python best practices

---

# 📁 **`utils/` Directory - Complete Breakdown**

## **Overview**

The `utils/` directory contains **utility modules** that provide cross-cutting functionality used throughout the entire project. These are helper functions and classes that don't belong to any specific feature but are needed everywhere.

**Files in this directory:**

1. `__init__.py` - Package initializer
2. `custom_exception.py` - Custom exception handling
3. `logger.py` - Logging configuration

---

## 📄 **File 1: `__init__.py`**

### **WHAT** is this file?

An **empty Python package initializer** that marks the `utils/` directory as a Python package.

### **WHY** do we need it?

- ✅ Makes `utils/` importable as a Python module
- ✅ Allows you to do: `from utils.logger import get_logger`
- ✅ Required by Python to recognize the directory as a package
- ✅ Can be used to expose commonly used functions (currently empty)

### **WHEN** is it used?

- **At import time** - Python checks for this file when you import from `utils`
- **During package discovery** - `find_packages()` in `setup.py` looks for this

### **WHERE** does it fit?

- **Location**: `utils/__init__.py`
- **Scope**: Makes the entire `utils/` directory a package

### **HOW** does it work?

```python
# Currently empty, but could contain:
from .logger import get_logger
from .custom_exception import CustomException

__all__ = ['get_logger', 'CustomException']
```

### **Current State**: Empty (minimal approach)

---

## 📄 **File 2: `custom_exception.py`**

### **WHAT** is this file?

A **custom exception class** that provides detailed error information including file name, line number, and error context.

### **WHY** do we need it?

1. **Better Debugging** - Shows exactly where errors occur
2. **Production-Ready** - Provides detailed error context for logs
3. **Consistent Error Handling** - Standardized error format across the project
4. **Traceable Errors** - Includes file name and line number automatically

### **WHEN** is it used?

- **During error handling** in try-except blocks
- **When raising custom errors** with additional context
- **In production** for better error tracking and debugging

### **WHERE** is it used?

- Throughout the project (src, pipeline, app modules)
- Anywhere you need to raise informative exceptions
- In production error logging

### **HOW** does it work?

#### **Line-by-Line Breakdown:**

```python
# Line 1: Import sys for exception info
import sys
```

- `sys.exc_info()` provides traceback information

```python
# Lines 3-6: Custom Exception Class
class CustomException(Exception):
    def __init__(self, message: str, error_detail: Exception = None):
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)
```

- **Inherits from `Exception`** - Standard Python exception
- **`message`** - Your custom error message
- **`error_detail`** - The original exception (optional)
- **Calls parent constructor** with formatted message

```python
# Lines 8-13: Static method to format error message
@staticmethod
def get_detailed_error_message(message, error_detail):
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown File"
    line_number = exc_tb.tb_lineno if exc_tb else "Unknown Line"
    return f"{message} | Error: {error_detail} | File: {file_name} | Line: {line_number}"
```

- **`sys.exc_info()`** - Gets current exception traceback
- **`exc_tb.tb_frame.f_code.co_filename`** - Extracts file name
- **`exc_tb.tb_lineno`** - Extracts line number
- **Returns formatted string** with all error details

```python
# Lines 15-16: String representation
def __str__(self):
    return self.error_message
```

- **Makes the exception printable** with full details

### **Usage Example:**

```python
from utils.custom_exception import CustomException

try:
    # Some operation
    result = risky_operation()
except Exception as e:
    raise CustomException("Failed to process anime data", e)

# Output:
# Failed to process anime data | Error: ValueError('Invalid ID') | File: /path/to/file.py | Line: 42
```

### **Benefits:**

- ✅ **Automatic file/line tracking** - No manual logging needed
- ✅ **Preserves original error** - Doesn't lose the root cause
- ✅ **Consistent format** - All errors look the same
- ✅ **Production-ready** - Easy to parse in log aggregators

---

## 📄 **File 3: `logger.py`**

### **WHAT** is this file?

A **centralized logging configuration** that creates daily log files and provides a logger factory function.

### **WHY** do we need it?

1. **Centralized Logging** - All logs go to one place
2. **Daily Log Files** - Organized by date for easy debugging
3. **Consistent Format** - All logs have timestamp, level, and message
4. **Production Monitoring** - Track application behavior in production
5. **Debugging** - Trace issues without print statements

### **WHEN** is it used?

- **Throughout the application** - Every module that needs logging
- **At startup** - Logging is configured when the module is imported
- **In production** - For monitoring and debugging
- **During development** - For tracking application flow

### **WHERE** does it fit?

- **Used by**: All modules (src, pipeline, app)
- **Logs stored in**: `logs/` directory
- **Log files**: `logs/log_2025-12-09.log` (date-based)

### **HOW** does it work?

#### **Line-by-Line Breakdown:**

```python
# Lines 1-3: Imports
import logging
import os
from datetime import datetime
```

- **`logging`** - Python's built-in logging module
- **`os`** - For directory operations
- **`datetime`** - For date-based log file names

```python
# Lines 5-6: Create logs directory
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
```

- **Creates `logs/` folder** if it doesn't exist
- **`exist_ok=True`** - No error if directory already exists

```python
# Line 8: Generate log file name
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
```

- **Creates date-based filename** - e.g., `log_2025-12-09.log`
- **One file per day** - Easy to find logs by date
- **Format**: `logs/log_YYYY-MM-DD.log`

```python
# Lines 10-14: Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
```

- **`filename`** - Where to write logs
- **`format`** - Log message structure:
  - `%(asctime)s` - Timestamp (e.g., `2025-12-09 23:00:00`)
  - `%(levelname)s` - Log level (INFO, WARNING, ERROR)
  - `%(message)s` - Your log message
- **`level=logging.INFO`** - Minimum log level to record

```python
# Lines 16-19: Logger factory function
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
```

- **Creates a named logger** - Usually the module name
- **Sets log level** - INFO and above (INFO, WARNING, ERROR, CRITICAL)
- **Returns configured logger** - Ready to use

### **Usage Example:**

```python
from utils.logger import get_logger

logger = get_logger(__name__)  # __name__ = module name

logger.info("Starting anime recommendation process")
logger.warning("Low confidence score: 0.3")
logger.error("Failed to load anime data")
```

### **Log Output Format:**

```
2025-12-09 23:00:00,123 - INFO - Starting anime recommendation process
2025-12-09 23:00:05,456 - WARNING - Low confidence score: 0.3
2025-12-09 23:00:10,789 - ERROR - Failed to load anime data
```

### **Log File Organization:**

```
logs/
├── log_2025-12-07.log
├── log_2025-12-08.log
└── log_2025-12-09.log  ← Today's logs
```

### **Benefits:**

- ✅ **Automatic daily rotation** - New file each day
- ✅ **Timestamped entries** - Know exactly when things happened
- ✅ **Centralized** - All logs in one place
- ✅ **Production-ready** - Can be integrated with log aggregators
- ✅ **Easy debugging** - Search logs by date

---

## 🎯 **How These Utils Work Together**

### **Typical Usage Pattern:**

```python
# In any module (e.g., src/data_processor.py)
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

def process_anime_data(data):
    try:
        logger.info("Starting data processing")
        # Process data
        result = transform(data)
        logger.info("Data processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise CustomException("Failed to process anime data", e)
```

### **What Happens:**

1. ✅ **Logger logs** the operation to `logs/log_2025-12-09.log`
2. ✅ **If error occurs**, CustomException provides detailed context
3. ✅ **Error includes** file name, line number, and original error
4. ✅ **Both logged and raised** for proper error handling

---

## 📊 **Summary: `utils/` Directory**

| File                  | Purpose        | Key Features                                |
| --------------------- | -------------- | ------------------------------------------- |
| `__init__.py`         | Package marker | Makes `utils/` importable                   |
| `custom_exception.py` | Error handling | Auto-tracks file/line, preserves context    |
| `logger.py`           | Logging system | Daily files, consistent format, centralized |

### **Design Principles:**

- ✅ **DRY (Don't Repeat Yourself)** - Centralized utilities
- ✅ **Separation of Concerns** - Each file has one job
- ✅ **Production-Ready** - Proper logging and error handling
- ✅ **Developer-Friendly** - Easy to use, consistent patterns

### **Best Practices Implemented:**

1. **Centralized logging** instead of scattered print statements
2. **Custom exceptions** with context instead of generic errors
3. **Daily log rotation** for easy debugging
4. **Consistent error format** across the entire project

---

**Ready for the next file/directory?** 🚀
