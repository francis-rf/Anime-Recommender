"""
Data loader module for AI Anime Recommender.

Handles loading, processing, and validation of anime dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


class AnimeDataLoader:
    """
    Loads and processes anime data from CSV files.
    
    Combines anime metadata (name, genres, synopsis) into a single
    text field suitable for vector embeddings.
    """
    
    def __init__(self, original_csv: str, processed_csv: str):
        """
        Initialize the AnimeDataLoader.
        
        Args:
            original_csv: Path to the original anime dataset CSV
            processed_csv: Path where processed data will be saved
            
        Raises:
            CustomException: If original CSV file doesn't exist
        """
        self.original_csv = Path(original_csv)
        self.processed_csv = Path(processed_csv)
        
        # Validate input file exists
        if not self.original_csv.exists():
            raise CustomException(
                f"Original CSV file not found: {original_csv}",
                FileNotFoundError(f"File does not exist: {original_csv}")
            )
        
        logger.info(f"Initialized AnimeDataLoader with source: {self.original_csv}")
    
    def load_and_process(self) -> str:
        """
        Load and process anime data from CSV.
        
        Reads the original CSV, validates required columns, combines
        relevant fields into a single text column, and saves to processed CSV.
        
        Returns:
            Path to the processed CSV file as string
            
        Raises:
            CustomException: If required columns are missing or processing fails
        """
        try:
            logger.info(f"Loading data from: {self.original_csv}")
            
            # Load CSV with error handling
            df = pd.read_csv(
                self.original_csv,
                encoding='utf-8',
                on_bad_lines='warn'
            )
            
            initial_rows = len(df)
            logger.info(f"Loaded {initial_rows} rows from CSV")
            
            # Drop rows with missing values
            df = df.dropna()
            rows_after_dropna = len(df)
            
            if rows_after_dropna < initial_rows:
                logger.warning(
                    f"Dropped {initial_rows - rows_after_dropna} rows with missing values"
                )
            
            # Validate required columns (fixed typo: sypnopsis -> synopsis)
            required_cols = {"Name", "Genres", "sypnopsis"}  # Keep original column name
            missing = required_cols - set(df.columns)
            
            if missing:
                error_msg = f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
                logger.error(error_msg)
                raise CustomException(
                    error_msg,
                    ValueError(f"Missing columns: {missing}")
                )
            
            logger.info("All required columns present")
            
            # Combine information into single text field
            logger.info("Combining anime information into single field")
            df['combined_info'] = (
                "Title: " + df['Name'].astype(str) + 
                " | Overview: " + df['sypnopsis'].astype(str) + 
                " | Genres: " + df['Genres'].astype(str)
            )
            
            # Create output directory if it doesn't exist
            self.processed_csv.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed data
            logger.info(f"Saving processed data to: {self.processed_csv}")
            df[['combined_info']].to_csv(
                self.processed_csv,
                index=False,
                encoding='utf-8'
            )
            
            logger.info(
                f"Successfully processed {len(df)} anime entries and saved to {self.processed_csv}"
            )
            
            return str(self.processed_csv)
            
        except pd.errors.EmptyDataError as e:
            logger.error(f"CSV file is empty: {self.original_csv}")
            raise CustomException("CSV file is empty", e)
            
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {self.original_csv}")
            raise CustomException("Failed to parse CSV file", e)
            
        except Exception as e:
            logger.error(f"Unexpected error during data processing: {e}")
            raise CustomException("Data processing failed", e)
    
    def get_data_info(self) -> Optional[dict]:
        """
        Get information about the processed data.
        
        Returns:
            Dictionary with data statistics, or None if file doesn't exist
        """
        if not self.processed_csv.exists():
            logger.warning(f"Processed CSV not found: {self.processed_csv}")
            return None
        
        try:
            df = pd.read_csv(self.processed_csv)
            info = {
                "total_entries": len(df),
                "file_path": str(self.processed_csv),
                "file_size_mb": self.processed_csv.stat().st_size / (1024 * 1024),
            }
            logger.info(f"Data info: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get data info: {e}")
            raise CustomException("Failed to retrieve data information", e)