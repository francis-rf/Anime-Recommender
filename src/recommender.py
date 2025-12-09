"""
Anime recommender module for AI Anime Recommender.

Handles LLM-based anime recommendations using RAG (Retrieval Augmented Generation)
with Groq API and ChromaDB vector store.
"""

from typing import Optional, Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.prompt_template import get_anime_prompt
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


class AnimeRecommender:
    """
    AI-powered anime recommendation system using RAG.
    
    Combines vector similarity search with LLM generation to provide
    contextual, personalized anime recommendations based on user queries.
    """
    
    def __init__(
        self,
        retriever,
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_k: int = 5
    ):
        """
        Initialize the AnimeRecommender.
        
        Args:
            retriever: LangChain retriever for vector similarity search
            api_key: Groq API key
            model_name: Name of the Groq model to use
            temperature: LLM temperature (0.0 = deterministic, 1.0 = creative, default: 0.0)
            max_tokens: Maximum tokens in response (None = model default)
            top_k: Number of similar documents to retrieve (default: 5)
            
        Raises:
            CustomException: If initialization fails
        """
        if not api_key:
            raise CustomException(
                "API key is required",
                ValueError("api_key cannot be empty")
            )
        
        if not model_name:
            raise CustomException(
                "Model name is required",
                ValueError("model_name cannot be empty")
            )
        
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.retriever = retriever
        
        # Initialize LLM
        logger.info(f"Initializing ChatGroq with model: {model_name}")
        try:
            llm_kwargs = {
                "api_key": api_key,
                "model_name": model_name,
                "temperature": temperature
            }
            
            if max_tokens:
                llm_kwargs["max_tokens"] = max_tokens
            
            self.llm = ChatGroq(**llm_kwargs)
            logger.info("ChatGroq initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq: {e}")
            raise CustomException("Failed to initialize LLM", e)
        
        # Get prompt template
        try:
            self.prompt = get_anime_prompt()
            logger.info("Prompt template loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            raise CustomException("Failed to load prompt template", e)
        
        # Create RAG chain using LCEL (LangChain Expression Language)
        logger.info("Building RAG chain...")
        try:
            self.chain = (
                RunnableParallel(
                    context=self.retriever,
                    input=RunnablePassthrough()
                )
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            logger.info("RAG chain built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build RAG chain: {e}")
            raise CustomException("Failed to build recommendation chain", e)
        
        logger.info(
            f"AnimeRecommender initialized - Model: {model_name}, "
            f"Temperature: {temperature}, Top-K: {top_k}"
        )
    
    def get_recommendations(self, query: str) -> str:
        """
        Get anime recommendations based on user query.
        
        Uses RAG to retrieve relevant anime from vector store and
        generates personalized recommendations using LLM.
        
        Args:
            query: User's question or preference (e.g., "action anime with ninjas")
            
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
            logger.info(f"Generating recommendations for query: '{query}'")
            
            # Invoke the RAG chain
            result = self.chain.invoke(query)
            
            logger.info("Recommendations generated successfully")
            logger.debug(f"Result length: {len(result)} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise CustomException(
                f"Failed to generate recommendations for query: '{query}'",
                e
            )
    
    def get_recommendations_with_context(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Get recommendations with retrieved context documents.
        
        Useful for debugging and understanding what documents
        were used to generate the recommendations.
        
        Args:
            query: User's question or preference
            
        Returns:
            Dictionary with 'recommendation' and 'context_documents'
            
        Raises:
            CustomException: If recommendation generation fails
        """
        if not query or not query.strip():
            raise CustomException(
                "Query cannot be empty",
                ValueError("query must be a non-empty string")
            )
        
        try:
            logger.info(f"Generating recommendations with context for: '{query}'")
            
            # Retrieve context documents
            context_docs = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(context_docs)} context documents")
            
            # Generate recommendation
            recommendation = self.chain.invoke(query)
            
            result = {
                "recommendation": recommendation,
                "context_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in context_docs
                ],
                "num_context_docs": len(context_docs)
            }
            
            logger.info("Recommendations with context generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating recommendations with context: {e}")
            raise CustomException(
                f"Failed to generate recommendations with context for: '{query}'",
                e
            )
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current recommender configuration.
        
        Returns:
            Dictionary with configuration settings
        """
        config = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "api_key_set": bool(self.api_key),
        }
        logger.info(f"Recommender config: {config}")
        return config