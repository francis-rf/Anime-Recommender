"""
Streamlit web application for AI Anime Recommender.

Provides an interactive web interface for getting anime recommendations
using the RAG-based recommendation pipeline.
"""

import streamlit as st
from pipeline.pipeline import AnimeRecommendationPipeline
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="🎌 AI Anime Recommender",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .example-button {
        margin: 0.25rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_pipeline():
    """
    Initialize the recommendation pipeline with caching.
    
    Returns:
        AnimeRecommendationPipeline instance
    """
    try:
        logger.info("Initializing Streamlit app pipeline")
        pipeline = AnimeRecommendationPipeline(
            temperature=0.0,  # Deterministic
            top_k=5,          # Retrieve 5 documents
            num_recommendations=3  # Recommend 3 anime
        )
        logger.info("Pipeline initialized successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        st.error(f"❌ Failed to initialize: {e}")
        st.stop()


# Initialize pipeline
pipeline = init_pipeline()

# Main content
st.markdown('<h1 class="main-header">🎌 AI Anime Recommender</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Discover your next favorite anime with AI-powered recommendations</p>',
    unsafe_allow_html=True
)

# Initialize session state
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'history' not in st.session_state:
    st.session_state.history = []

# Query input
query = st.text_input(
    "🔍 What kind of anime are you looking for?",
    value=st.session_state.query,
    placeholder="e.g., action anime with strong female lead",
    help="Describe your preferences, favorite genres, themes, or moods"
)

# Example queries section
st.markdown("### 💡 Try these examples:")
example_queries = [
    "Action anime with ninjas",
    "Romance anime with school setting",
    "Sci-fi anime with time travel",
    "Comedy anime with supernatural elements",
    "Dark fantasy with complex plot",
    "Slice of life with heartwarming moments"
]

# Display examples in 3 columns
cols = st.columns(3)
for i, example in enumerate(example_queries):
    with cols[i % 3]:
        if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
            st.session_state.query = example
            st.rerun()

st.markdown("---")

# Action buttons
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.query = ""
        st.rerun()

with col2:
    if st.button("📜 History", use_container_width=True) and st.session_state.history:
        with st.expander("Query History", expanded=True):
            for i, (q, _) in enumerate(reversed(st.session_state.history[-5:]), 1):
                st.markdown(f"{i}. {q}")

# Process query
if query:
    try:
        with st.spinner("🔮 Generating recommendations..."):
            logger.info(f"Processing query: {query}")
            
            # Get recommendations
            recommendations = pipeline.recommend(query)
            
            # Add to history
            st.session_state.history.append((query, recommendations))
            
            # Display recommendations
            st.markdown("---")
            st.markdown("## 🎬 Your Recommendations")
            st.markdown(recommendations)
            
            # Feedback section
            st.markdown("---")
            st.markdown("### Was this helpful?")
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("👍 Yes, helpful!", use_container_width=True):
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎 Not helpful", use_container_width=True):
                    st.info("We'll improve our recommendations!")
            
            logger.info("Recommendations displayed successfully")
            
    except CustomException as e:
        logger.error(f"Custom exception: {e}")
        st.error(f"❌ Error: {e}")
        st.info("💡 Try rephrasing your query or try again later")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error(f"❌ Unexpected error: {e}")
        st.info("💡 Please try again or contact support")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Made with ❤️ using Streamlit, LangChain, and Groq | 
        Powered by AI 🤖
    </div>
    """,
    unsafe_allow_html=True
)
