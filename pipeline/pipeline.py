"""
Recommendation pipeline for AI Anime Recommender.

Provides a high-level interface for generating anime recommendations
using the complete RAG pipeline (retrieval + generation).
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from src.prompt_template import get_prompt_by_style
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


class AnimeRecommendationPipeline:
    """
    Complete anime recommendation pipeline using RAG.
    
    Combines vector similarity search with LLM generation to provide
    intelligent, context-aware anime recommendations.
    """
    
    def __init__(
        self,
        persist_dir: str = 'chroma_db',
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        top_k: int = 5,
        prompt_style: str = "default",
        num_recommendations: int = 3
    ):
        """
        Initialize the recommendation pipeline.
        
        Args:
            persist_dir: Vector store directory (default: chroma_db)
            model_name: Groq model name (default: from config)
            api_key: Groq API key (default: from config)
            temperature: LLM temperature (default: 0.0)
            top_k: Number of documents to retrieve (default: 5)
            prompt_style: Prompt style ("default", "detailed", "casual", "genre")
            num_recommendations: Number of anime to recommend (default: 3)
            
        Raises:
            CustomException: If initialization fails
        """
        try:
            logger.info("=" * 60)
            logger.info("🎬 Initializing Anime Recommendation Pipeline")
            logger.info("=" * 60)
            
            # Use config values if not provided
            self.model_name = model_name or MODEL_NAME
            self.api_key = api_key or GROQ_API_KEY
            self.temperature = temperature
            self.top_k = top_k
            self.prompt_style = prompt_style
            self.num_recommendations = num_recommendations
            
            # Validate API key
            if not self.api_key:
                raise CustomException(
                    "GROQ_API_KEY not found. Please set it in your .env file",
                    ValueError("API key is required")
                )
            
            # Get base directory and construct persist_dir path
            base_dir = Path(__file__).parent.parent
            self.persist_dir = base_dir / persist_dir
            
            # Load vector store
            logger.info(f"📚 Loading vector store from: {self.persist_dir}")
            
            # Use the actual processed CSV file
            processed_csv = base_dir / "data" / "processed_anime.csv"
            
            vector_builder = VectorStoreBuilder(
                csv_path=str(processed_csv),
                persist_dir=str(self.persist_dir)
            )
            
            # Load existing vector store
            vectorstore = vector_builder.load_vectorstore()
            logger.info("✅ Vector store loaded successfully")
            
            # Get vector store info
            vs_info = vector_builder.get_vectorstore_info()
            if vs_info:
                logger.info(f"📊 Vector store contains {vs_info['total_chunks']} chunks")
            
            # Create retriever
            self.retriever = vectorstore.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            logger.info(f"🔍 Retriever configured (top_k={self.top_k})")
            
            # Initialize recommender
            logger.info(f"🤖 Initializing recommender with model: {self.model_name}")
            self.recommender = AnimeRecommender(
                retriever=self.retriever,
                api_key=self.api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                top_k=self.top_k
            )
            
            logger.info("=" * 60)
            logger.info("✅ Pipeline Initialized Successfully!")
            logger.info("=" * 60)
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Temperature: {self.temperature}")
            logger.info(f"Top-K: {self.top_k}")
            logger.info(f"Prompt Style: {self.prompt_style}")
            logger.info(f"Recommendations per query: {self.num_recommendations}")
            logger.info("=" * 60)
            
        except CustomException:
            raise
            
        except Exception as e:
            logger.error(f"❌ Error initializing pipeline: {e}")
            raise CustomException("Failed to initialize recommendation pipeline", e)
    
    def recommend(self, query: str) -> str:
        """
        Get anime recommendations for a query.
        
        Args:
            query: User's question or preference
            
        Returns:
            AI-generated anime recommendations as string
            
        Raises:
            CustomException: If recommendation generation fails
        """
        if not query or not query.strip():
            raise CustomException(
                "Query cannot be empty",
                ValueError("query must be a non-empty string")
            )
        
        try:
            logger.info(f"\n🎯 Processing query: '{query}'")
            logger.info("-" * 60)
            
            recommendations = self.recommender.get_recommendations(query)
            
            logger.info("✅ Recommendations generated successfully")
            logger.info("-" * 60)
            
            return recommendations
            
        except CustomException:
            raise
            
        except Exception as e:
            logger.error(f"❌ Error getting recommendations: {e}")
            raise CustomException(f"Failed to get recommendations for: '{query}'", e)
    
    def recommend_with_context(self, query: str) -> Dict[str, Any]:
        """
        Get recommendations with retrieved context documents.
        
        Useful for debugging and transparency.
        
        Args:
            query: User's question or preference
            
        Returns:
            Dictionary with recommendation and context documents
            
        Raises:
            CustomException: If recommendation generation fails
        """
        try:
            logger.info(f"\n🎯 Processing query with context: '{query}'")
            logger.info("-" * 60)
            
            result = self.recommender.get_recommendations_with_context(query)
            
            logger.info(f"✅ Generated recommendations using {result['num_context_docs']} context documents")
            logger.info("-" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting recommendations with context: {e}")
            raise CustomException(
                f"Failed to get recommendations with context for: '{query}'",
                e
            )
    
    def batch_recommend(self, queries: List[str]) -> List[str]:
        """
        Get recommendations for multiple queries.
        
        Args:
            queries: List of user queries
            
        Returns:
            List of recommendations (one per query)
            
        Raises:
            CustomException: If batch processing fails
        """
        try:
            logger.info(f"\n📦 Processing batch of {len(queries)} queries")
            logger.info("=" * 60)
            
            results = []
            for i, query in enumerate(queries, 1):
                logger.info(f"\n[{i}/{len(queries)}] Processing: '{query}'")
                recommendation = self.recommend(query)
                results.append(recommendation)
            
            logger.info("\n" + "=" * 60)
            logger.info(f"✅ Batch processing complete: {len(results)} recommendations generated")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in batch processing: {e}")
            raise CustomException("Batch recommendation failed", e)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary with pipeline settings
        """
        info = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "prompt_style": self.prompt_style,
            "num_recommendations": self.num_recommendations,
            "persist_dir": str(self.persist_dir),
            "api_key_set": bool(self.api_key),
        }
        
        logger.info(f"Pipeline info: {info}")
        return info


def main():
    """Demo of the recommendation pipeline."""
    try:
        # Initialize pipeline
        pipeline = AnimeRecommendationPipeline(
            temperature=0.0,
            top_k=5,
            num_recommendations=3
        )
        
        # Example query
        query = "I want action anime with ninjas"
        print(f"\n🎯 Query: {query}\n")
        
        # Get recommendations
        recommendations = pipeline.recommend(query)
        print(recommendations)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
