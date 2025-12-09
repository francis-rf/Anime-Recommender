import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from dotenv import load_dotenv  
from utils.logger import get_logger
from utils.custom_exception import CustomException
load_dotenv(override=True)

logger = get_logger(__name__)

def main():
    try:
        logger.info("Starting Pipeline...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        original_csv = os.path.join(data_dir, 'anime_with_synopsis.csv')
        processed_csv_path = os.path.join(data_dir, 'anime_processed.csv')
        
        loader = AnimeDataLoader(original_csv=original_csv, processed_csv=processed_csv_path)
        processed_csv = loader.load_and_process()
        logger.info("Data loaded and processed successfull....")   
        vector_builder = VectorStoreBuilder(csv_path=processed_csv)
        vector_builder.build_and_save_vectorstore()
        logger.info("Vectorstore built successfully....")
        logger.info("Pipeline completed successfully....")
    except Exception as e:
        logger.error(f"Error in Pipeline: {str(e)}")
        raise CustomException(f"Error in Pipeline: {str(e)}")

if __name__ == "__main__":
    main()

