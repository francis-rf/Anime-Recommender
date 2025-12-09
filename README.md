# 🎌 AI Anime Recommender

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Discover your next favorite anime with AI-powered recommendations using RAG (Retrieval Augmented Generation)**

An intelligent anime recommendation system that combines semantic search with large language models to provide personalized, context-aware anime suggestions based on your preferences.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🎯 Core Features

- **🤖 AI-Powered Recommendations** - Uses Groq's LLM for intelligent, context-aware suggestions
- **🔍 Semantic Search** - ChromaDB vector store with HuggingFace embeddings for similarity matching
- **📊 RAG Pipeline** - Retrieval Augmented Generation for accurate, grounded recommendations
- **🎨 Beautiful UI** - Interactive Streamlit web interface with gradient design
- **💡 Example Queries** - Pre-built query templates for quick exploration
- **📜 Query History** - Track your previous searches (last 5 queries)
- **⚡ Fast & Efficient** - Optimized vector search and caching

### 🛠️ Technical Features

- **Production-Ready Code** - Comprehensive logging, error handling, and type hints
- **Modular Architecture** - Clean separation of concerns across 6 layers
- **Configurable Pipelines** - Flexible data processing and recommendation workflows
- **Multiple Prompt Styles** - Default, detailed, casual, and genre-specific prompts
- **Batch Processing** - Process multiple queries simultaneously
- **Context Retrieval** - Optional display of source anime used for recommendations

---

## 🏗️ Architecture

The project follows a **6-layer architecture** for clean separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 6: User Interface (Streamlit App)                │
├─────────────────────────────────────────────────────────┤
│  Layer 5: Pipeline Orchestration                        │
│  ├─ builder_pipeline.py (Data + Vector Store)          │
│  └─ pipeline.py (Recommendation Pipeline)              │
├─────────────────────────────────────────────────────────┤
│  Layer 4: RAG & Recommendation Engine                   │
│  └─ recommender.py (AnimeRecommender)                  │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Vector Store & Embeddings                     │
│  └─ vector_store.py (VectorStoreBuilder)               │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Core Data Processing                          │
│  ├─ data_loader.py (AnimeDataLoader)                   │
│  └─ prompt_template.py (Prompt Engineering)            │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Configuration & Utilities                     │
│  ├─ config.py (Config)                                 │
│  ├─ logger.py (Logging)                                │
│  └─ custom_exception.py (Error Handling)               │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

### Core Technologies

- **Python 3.8+** - Programming language
- **Streamlit** - Web UI framework
- **LangChain** - LLM orchestration framework
- **Groq** - Fast LLM inference (llama-3.3-70b-versatile)
- **ChromaDB** - Vector database for embeddings
- **HuggingFace** - Sentence transformers for embeddings

### Key Libraries

```
langchain>=0.1.0
langchain-community>=0.0.20
langchain-groq>=0.0.1
langchain-huggingface>=0.0.1
chromadb>=0.4.22
sentence-transformers>=2.2.2
streamlit>=1.28.0
pandas>=2.0.0
python-dotenv>=1.0.0
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Groq API key ([Get one here](https://console.groq.com))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ai-anime-recommender.git
cd ai-anime-recommender
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Project as Package

```bash
pip install -e .
```

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_MODEL=llama-3.3-70b-versatile
ENVIRONMENT=development
DEBUG=True
```

### Step 6: Build the Vector Store

```bash
python pipeline/builder_pipeline.py
```

This will:

1. Load and process the anime dataset (`data/anime_with_synopsis.csv`)
2. Create embeddings using HuggingFace
3. Build and persist the ChromaDB vector store

---

## 🚀 Usage

### Running the Web App

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Recommendation Pipeline (Python)

```python
from pipeline.pipeline import AnimeRecommendationPipeline

# Initialize the pipeline
pipeline = AnimeRecommendationPipeline(
    temperature=0.0,      # 0 = deterministic, 1 = creative
    top_k=5,              # Number of similar anime to retrieve
    num_recommendations=3 # Number of anime to recommend
)

# Get recommendations
query = "action anime with ninjas"
recommendations = pipeline.recommend(query)
print(recommendations)

# Get recommendations with context
result = pipeline.recommend_with_context(query)
print(result['recommendation'])
print(f"Used {result['num_context_docs']} context documents")

# Batch processing
queries = [
    "romance anime with school setting",
    "sci-fi anime with time travel",
    "dark fantasy with complex plot"
]
results = pipeline.batch_recommend(queries)
```

### Building the Vector Store Programmatically

```python
from pipeline.builder_pipeline import run_builder_pipeline, DataPipelineConfig

# Custom configuration
config = DataPipelineConfig(
    original_csv="data/anime_with_synopsis.csv",
    processed_csv="data/processed_anime.csv",
    persist_dir="chroma_db",
    chunk_size=1000,
    chunk_overlap=100
)

# Run the pipeline
success = run_builder_pipeline(config)
```

---

## 📁 Project Structure

```
ai-anime-recommender/
├── app/
│   ├── __init__.py
│   └── app.py                      # Streamlit web application
├── config/
│   ├── __init__.py
│   └── config.py                   # Configuration management
├── data/
│   ├── anime_with_synopsis.csv     # Original dataset
│   └── processed_anime.csv         # Processed dataset
├── pipeline/
│   ├── __init__.py
│   ├── builder_pipeline.py         # Data processing pipeline
│   └── pipeline.py                 # Recommendation pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # CSV data loading & processing
│   ├── prompt_template.py          # LLM prompt templates
│   ├── recommender.py              # RAG recommendation engine
│   └── vector_store.py             # ChromaDB vector store
├── utils/
│   ├── __init__.py
│   ├── custom_exception.py         # Custom exception handling
│   └── logger.py                   # Logging configuration
├── logs/                           # Application logs
├── chroma_db/                      # Persisted vector store
├── .env                            # Environment variables
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── setup.py                        # Package configuration
├── project_architecture.html       # Interactive architecture diagram
└── README.md                       # This file
```

---

## ⚙️ Configuration

### Environment Variables

| Variable         | Description             | Default                   |
| ---------------- | ----------------------- | ------------------------- |
| `GROQ_API_KEY`   | Groq API key (required) | -                         |
| `OPENAI_MODEL`   | Groq model name         | `llama-3.3-70b-versatile` |
| `ENVIRONMENT`    | Environment mode        | `development`             |
| `DEBUG`          | Enable debug mode       | `True`                    |
| `LOG_LEVEL`      | Logging level           | `INFO`                    |
| `CHROMA_DB_PATH` | Vector store path       | `chroma_db`               |

### Pipeline Configuration

**VectorStoreBuilder:**

```python
VectorStoreBuilder(
    csv_path="data/processed_anime.csv",
    persist_dir="chroma_db",
    chunk_size=1000,           # Text chunk size
    chunk_overlap=100,         # Overlap between chunks
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"               # or "cuda" for GPU
)
```

**AnimeRecommender:**

```python
AnimeRecommender(
    retriever=retriever,
    api_key="your_api_key",
    model_name="llama-3.3-70b-versatile",
    temperature=0.0,           # 0-1, higher = more creative
    max_tokens=2000,           # Max response length
    top_k=5                    # Documents to retrieve
)
```

---

## 🔍 How It Works

### 1️⃣ Data Processing

```python
# AnimeDataLoader processes the CSV
loader = AnimeDataLoader(
    original_csv="data/anime_with_synopsis.csv",
    processed_csv="data/processed_anime.csv"
)
processed_csv = loader.load_and_process()
```

**What it does:**

- Loads anime CSV with title, genres, and synopsis
- Drops rows with missing values
- Combines fields into `combined_info` column
- Saves processed data

### 2️⃣ Vector Store Creation

```python
# VectorStoreBuilder creates embeddings
builder = VectorStoreBuilder(csv_path=processed_csv)
builder.build_and_save_vectorstore()
```

**What it does:**

- Chunks text into manageable pieces (1000 chars, 100 overlap)
- Generates embeddings using HuggingFace model
- Stores vectors in ChromaDB
- Persists to disk for reuse

### 3️⃣ Semantic Search

```python
# Retriever finds similar anime
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.get_relevant_documents("action anime with ninjas")
```

**What it does:**

- Converts query to embedding
- Performs similarity search in vector space
- Returns top-k most similar anime

### 4️⃣ LLM Generation

```python
# AnimeRecommender generates recommendations
recommender = AnimeRecommender(retriever=retriever, api_key=api_key)
recommendations = recommender.get_recommendations("action anime with ninjas")
```

**What it does:**

- Retrieves relevant anime from vector store
- Constructs prompt with context
- Calls Groq LLM for generation
- Returns formatted recommendations

---

## 📚 API Reference

### Core Classes

#### `AnimeDataLoader`

```python
class AnimeDataLoader:
    def __init__(self, original_csv: str, processed_csv: str)
    def load_and_process(self) -> str
    def get_data_info(self) -> dict
```

#### `VectorStoreBuilder`

```python
class VectorStoreBuilder:
    def __init__(self, csv_path: str, persist_dir: str = "chroma_db", ...)
    def build_and_save_vectorstore(self) -> None
    def load_vectorstore(self) -> Chroma
    def get_vectorstore_info(self) -> dict
    def similarity_search(self, query: str, k: int = 5) -> list
```

#### `AnimeRecommender`

```python
class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str = "llama-3.3-70b-versatile", ...)
    def get_recommendations(self, query: str) -> str
    def get_recommendations_with_context(self, query: str) -> dict
    def get_config(self) -> dict
```

#### `AnimeRecommendationPipeline`

```python
class AnimeRecommendationPipeline:
    def __init__(self, persist_dir: str = "chroma_db", ...)
    def recommend(self, query: str) -> str
    def recommend_with_context(self, query: str) -> dict
    def batch_recommend(self, queries: list) -> list
    def get_pipeline_info(self) -> dict
```

### Prompt Templates

```python
from src.prompt_template import get_prompt_by_style

# Available styles
prompt = get_prompt_by_style("default", num_recommendations=3)
prompt = get_prompt_by_style("detailed", num_recommendations=5)
prompt = get_prompt_by_style("casual", num_recommendations=3)
prompt = get_prompt_by_style("genre", genre="action", num_recommendations=3)
```

---

## 🛠️ Development



### Logging

Logs are stored in the `logs/` directory with automatic rotation:

- **Console logging** - Real-time output during development
- **File logging** - Persistent logs with daily rotation
- **Auto cleanup** - Keeps last 7 days of logs

### Code Quality

The project follows these best practices:

- ✅ **Type hints** - All functions have type annotations
- ✅ **Docstrings** - Comprehensive documentation
- ✅ **Error handling** - Custom exceptions with detailed tracebacks
- ✅ **Logging** - Extensive logging at all levels
- ✅ **Validation** - Input validation throughout
- ✅ **Modular design** - Clean separation of concerns

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ai-anime-recommender.git
cd ai-anime-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt  # if available
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings to all classes and methods
- Write meaningful commit messages
- Add tests for new features

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 AI Anime Recommender

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **Groq** - For providing fast LLM inference
- **LangChain** - For the RAG framework
- **HuggingFace** - For embedding models
- **ChromaDB** - For vector storage
- **Streamlit** - For the web UI framework
- **Anime Dataset** - Original dataset contributors

---

## 📧 Contact

**Project Maintainer:** Your Name

- GitHub: [francis-rf](https://github.com/yourusername)
- Email: rfranci789@gmail.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/ai-anime-recommender&type=Date)](https://star-history.com/#yourusername/ai-anime-recommender&Date)

---

## 📊 Project Stats

![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/ai-anime-recommender)
![GitHub contributors](https://img.shields.io/github/contributors/yourusername/ai-anime-recommender)
![GitHub stars](https://img.shields.io/github/stars/yourusername/ai-anime-recommender?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/ai-anime-recommender?style=social)

---

<div align="center">

**Made with ❤️ using Python, LangChain, and Groq**

[⬆ Back to Top](#-ai-anime-recommender)

</div>
