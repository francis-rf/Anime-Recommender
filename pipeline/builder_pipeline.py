"""
Builder pipeline for AI Anime Recommender.

Orchestrates the complete data processing and vector store building pipeline:
1. Loads and processes raw anime CSV data
2. Builds and persists ChromaDB vector store
"""

import os
from pathlib import Path
from typing import Optional

from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


class DataPipelineConfig:
    """Configuration for the data processing pipeline."""
    
    def __init__(
        self,
        original_csv: Optional[str] = None,
        processed_csv: Optional[str] = None,
        persist_dir: str = 'chroma_db',
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        """
        Initialize pipeline configuration.
        
        Args:
            original_csv: Path to original anime CSV (default: data/anime_with_synopsis.csv)
            processed_csv: Path for processed CSV (default: data/anime_processed.csv)
            persist_dir: Vector store directory (default: chroma_db)
            chunk_size: Text chunk size (default: 1000)
            chunk_overlap: Chunk overlap (default: 100)
        """
        # Get base directory (project root)
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / 'data'
        
        # Set default paths
        self.original_csv = Path(original_csv) if original_csv else data_dir / 'anime_with_synopsis.csv'
        self.processed_csv = Path(processed_csv) if processed_csv else data_dir / 'anime_processed.csv'
        self.persist_dir = base_dir / persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"Pipeline config - Original: {self.original_csv}, Processed: {self.processed_csv}")


def run_builder_pipeline(config: Optional[DataPipelineConfig] = None) -> bool:
    """
    Run the complete data processing and vector store building pipeline.
    
    Args:
        config: Pipeline configuration (uses defaults if None)
        
    Returns:
        True if pipeline completed successfully, False otherwise
        
    Raises:
        CustomException: If pipeline fails at any stage
    """
    if config is None:
        config = DataPipelineConfig()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 Starting Anime Recommender Builder Pipeline")
        logger.info("=" * 60)
        
        # Stage 1: Data Loading and Processing
        logger.info("\n📊 Stage 1/2: Loading and Processing Data")
        logger.info("-" * 60)
        
        if not config.original_csv.exists():
            raise CustomException(
                f"Original CSV not found: {config.original_csv}",
                FileNotFoundError(f"File does not exist: {config.original_csv}")
            )
        
        loader = AnimeDataLoader(
            original_csv=str(config.original_csv),
            processed_csv=str(config.processed_csv)
        )
        
        processed_csv = loader.load_and_process()
        logger.info(f"✅ Data processed successfully: {processed_csv}")
        
        # Get data info
        data_info = loader.get_data_info()
        if data_info:
            logger.info(f"📈 Processed {data_info['total_entries']} anime entries")
        
        # Stage 2: Vector Store Building
        logger.info("\n🔮 Stage 2/2: Building Vector Store")
        logger.info("-" * 60)
        
        vector_builder = VectorStoreBuilder(
            csv_path=processed_csv,
            persist_dir=str(config.persist_dir),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        vector_builder.build_and_save_vectorstore()
        logger.info(f"✅ Vector store built successfully: {config.persist_dir}")
        
        # Get vector store info
        vs_info = vector_builder.get_vectorstore_info()
        if vs_info:
            logger.info(f"📊 Vector store contains {vs_info['total_chunks']} chunks")
        
        # Pipeline Complete
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Pipeline Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"✅ Processed CSV: {config.processed_csv}")
        logger.info(f"✅ Vector Store: {config.persist_dir}")
        
        return True
        
    except CustomException as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in pipeline: {e}")
        raise CustomException("Builder pipeline failed", e)


def main():
    """Main entry point for the builder pipeline."""
    try:
        # Run with default configuration
        success = run_builder_pipeline()
        
        if success:
            logger.info("\n✨ You can now run the recommendation pipeline!")
            return 0
        else:
            logger.error("\n❌ Pipeline failed")
            return 1
            
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
