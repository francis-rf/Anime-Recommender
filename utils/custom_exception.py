import sys
import traceback
import inspect
from typing import Optional

class CustomException(Exception):
    """
    Custom exception with detailed error context.
    
    Automatically captures file name, line number, and full traceback.
    Preserves the original exception for debugging.
    """
    
    def __init__(self, message: str, error_detail: Optional[Exception] = None):
        """
        Initialize custom exception.
        
        Args:
            message: Custom error message
            error_detail: Original exception (if any)
        """
        self.original_error = error_detail
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(message: str, error_detail: Optional[Exception]) -> str:
        """
        Generate detailed error message with context.
        
        Args:
            message: Custom error message
            error_detail: Original exception
            
        Returns:
            Formatted error message with file, line, and traceback
        """
        # Try to get exception info first
        exc_type, exc_value, exc_tb = sys.exc_info()
        
        if exc_tb:
            # Extract from exception traceback
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            tb_str = ''.join(traceback.format_tb(exc_tb))
        else:
            # Fallback: Get from current stack frame
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                file_name = caller_frame.f_code.co_filename
                line_number = caller_frame.f_lineno
            else:
                file_name = "Unknown File"
                line_number = "Unknown Line"
            tb_str = ""
        
        # Build error message
        error_msg = f"{message} | Error: {error_detail} | File: {file_name} | Line: {line_number}"
        
        # Add full traceback if available (useful for debugging)
        if tb_str:
            error_msg += f"\n\nTraceback:\n{tb_str}"
        
        return error_msg

    def __str__(self) -> str:
        """String representation of the exception."""
        return self.error_message
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"CustomException('{self.error_message}')"