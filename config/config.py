"""
Configuration module for AI Anime Recommender.

Loads environment variables and provides configuration settings
for the application with validation and defaults.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)


class Config:
    """
    Application configuration class.
    
    Loads and validates environment variables for the AI Anime Recommender.
    Provides default values and validation for critical settings.
    """
    
    # API Configuration
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    MODEL_NAME: str = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")  # Default model
    
    # Application Settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    
    # Database Configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "chroma_db")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> None:
        """
        Validate critical configuration values.
        
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set it in your .env file or environment variables."
            )
        
        if not cls.MODEL_NAME:
            raise ValueError(
                "OPENAI_MODEL is not set. Please set it in your .env file or use default."
            )
        
        # Validate environment
        valid_environments = ["development", "production", "testing"]
        if cls.ENVIRONMENT not in valid_environments:
            raise ValueError(
                f"Invalid ENVIRONMENT: {cls.ENVIRONMENT}. Must be one of {valid_environments}"
            )
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """
        Get a summary of current configuration (safe for logging).
        
        Returns:
            Dictionary with non-sensitive configuration values
        """
        return {
            "model_name": cls.MODEL_NAME,
            "environment": cls.ENVIRONMENT,
            "debug": cls.DEBUG,
            "chroma_db_path": cls.CHROMA_DB_PATH,
            "log_level": cls.LOG_LEVEL,
            "api_key_set": bool(cls.GROQ_API_KEY),
        }


# Validate configuration on module import
try:
    Config.validate()
except ValueError as e:
    print(f"⚠️  Configuration Error: {e}")
    # Don't raise in development to allow for easier debugging
    if Config.ENVIRONMENT == "production":
        raise


# Export commonly used values for convenience
GROQ_API_KEY = Config.GROQ_API_KEY
MODEL_NAME = Config.MODEL_NAME