import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY,MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException


logger = get_logger(__name__)

class AnimeRecommendationPipeline:
    def __init__(self,persist_dir ='chroma_db'):
        try:
            logger.info('Initializing Pipeline')
            vector_builder = VectorStoreBuilder(csv_path="",persist_dir=persist_dir)
            retriever = vector_builder.load_vectorstore().as_retriever()
            self.recommender = AnimeRecommender(model_name=MODEL_NAME,api_key=GROQ_API_KEY,retriever=retriever)
            logger.info('Pipeline initialized successfully')
        except Exception as e:
            logger.error(f'Error initializing pipeline: {str(e)}')
            raise CustomException(f'Error initializing pipeline: {str(e)}')
    def recommend(self,query:str) ->str:
        try:
            logger.info(f"Received Query: {query}")
            recommendations = self.recommender.get_recommendations(query)
            logger.info(f"Recommendations Received succesfully")
            return recommendations
        except Exception as e:
            logger.error(f'Error getting recommendations: {str(e)}')
            raise CustomException(f'Error getting recommendations: {str(e)}')


