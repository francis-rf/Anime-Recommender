"""
Vector store module for AI Anime Recommender.

Handles creation and management of ChromaDB vector store
for semantic search of anime data using embeddings.
"""

from pathlib import Path
from typing import Optional, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


class VectorStoreBuilder:
    """
    Builds and manages ChromaDB vector store for anime recommendations.
    
    Uses HuggingFace embeddings to create semantic search capabilities
    over anime dataset. Supports both building new stores and loading
    existing ones.
    """
    
    def __init__(
        self,
        csv_path: str,
        persist_dir: str = 'chroma_db',
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        device: str = 'cpu'
    ):
        """
        Initialize the VectorStoreBuilder.
        
        Args:
            csv_path: Path to the processed CSV file with anime data
            persist_dir: Directory to save/load the vector store (default: 'chroma_db')
            chunk_size: Maximum size of text chunks (default: 1000)
            chunk_overlap: Overlap between chunks for context (default: 100)
            embedding_model: HuggingFace model for embeddings (default: all-MiniLM-L6-v2)
            device: Device to run embeddings on ('cpu' or 'cuda', default: 'cpu')
            
        Raises:
            CustomException: If CSV file doesn't exist
        """
        self.csv_path = Path(csv_path) if csv_path else None
        self.persist_dir = Path(persist_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.device = device
        
        # Only validate CSV exists if we have a real CSV path
        # This allows loading existing vector stores without needing the original CSV
        if self.csv_path and self.csv_path.exists():
            logger.info(f"CSV file found: {self.csv_path}")
        elif self.csv_path and not str(self.csv_path).endswith("dummy.csv"):
            # Only raise error if it's a real path (not dummy) and doesn't exist
            raise CustomException(
                f"CSV file not found: {csv_path}",
                FileNotFoundError(f"File does not exist: {csv_path}")
            )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Initialize embeddings
        logger.info(f"Initializing embeddings with model: {self.embedding_model}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': self.device}
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise CustomException("Failed to initialize embedding model", e)
        
        logger.info(
            f"VectorStoreBuilder initialized - CSV: {self.csv_path}, "
            f"Persist Dir: {self.persist_dir}, Chunk Size: {self.chunk_size}"
        )
    
    def build_and_save_vectorstore(self) -> None:
        """
        Build vector store from CSV and persist to disk.
        
        Loads CSV data, splits into chunks, creates embeddings,
        and saves to ChromaDB for later retrieval.
        
        Raises:
            CustomException: If building or saving fails
        """
        try:
            logger.info(f"Loading data from CSV: {self.csv_path}")
            
            # Load CSV data
            loader = CSVLoader(
                file_path=str(self.csv_path),
                encoding='utf-8',
                metadata_columns=[]
            )
            data = loader.load()
            logger.info(f"Loaded {len(data)} documents from CSV")
            
            if not data:
                raise CustomException(
                    "No data loaded from CSV",
                    ValueError("CSV file is empty or has no valid data")
                )
            
            # Split documents into chunks
            logger.info(f"Splitting documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
            chunks = self.text_splitter.split_documents(data)
            logger.info(f"Created {len(chunks)} chunks from {len(data)} documents")
            
            # Create persist directory if it doesn't exist
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Build vector store
            logger.info(f"Building vector store with {len(chunks)} chunks...")
            logger.info("This may take a few minutes depending on dataset size...")
            
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(self.persist_dir)
            )
            
            # Persist to disk
            logger.info(f"Persisting vector store to: {self.persist_dir}")
            vectorstore.persist()
            
            logger.info(
                f"✅ Vector store built successfully! "
                f"Stored {len(chunks)} chunks in {self.persist_dir}"
            )
            
        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")
            raise CustomException("Vector store creation failed", e)
    
    def load_vectorstore(self) -> Chroma:
        """
        Load existing vector store from disk.
        
        Returns:
            Chroma vector store instance ready for querying
            
        Raises:
            CustomException: If vector store doesn't exist or loading fails
        """
        try:
            # Check if vector store exists
            if not self.persist_dir.exists():
                raise CustomException(
                    f"Vector store not found at: {self.persist_dir}. "
                    "Please build it first using build_and_save_vectorstore()",
                    FileNotFoundError(f"Directory does not exist: {self.persist_dir}")
                )
            
            logger.info(f"Loading vector store from: {self.persist_dir}")
            
            vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
            
            logger.info("✅ Vector store loaded successfully")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise CustomException("Vector store loading failed", e)
    
    def get_vectorstore_info(self) -> Optional[dict]:
        """
        Get information about the vector store.
        
        Returns:
            Dictionary with vector store statistics, or None if not exists
        """
        if not self.persist_dir.exists():
            logger.warning(f"Vector store not found at: {self.persist_dir}")
            return None
        
        try:
            vectorstore = self.load_vectorstore()
            
            # Get collection info
            collection = vectorstore._collection
            count = collection.count()
            
            info = {
                "persist_directory": str(self.persist_dir),
                "total_chunks": count,
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "device": self.device,
            }
            
            logger.info(f"Vector store info: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get vector store info: {e}")
            raise CustomException("Failed to retrieve vector store information", e)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query text
            k: Number of results to return (default: 5)
            
        Returns:
            List of most similar documents
            
        Raises:
            CustomException: If search fails
        """
        try:
            logger.info(f"Performing similarity search for: '{query}' (k={k})")
            
            vectorstore = self.load_vectorstore()
            results = vectorstore.similarity_search(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise CustomException("Similarity search failed", e)    