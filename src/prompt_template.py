"""
Prompt template module for AI Anime Recommender.

Provides optimized prompt templates for generating high-quality
anime recommendations using LLMs.
"""

from typing import Optional
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from utils.logger import get_logger

logger = get_logger(__name__)


def get_anime_prompt(num_recommendations: int = 3) -> PromptTemplate:
    """
    Get the default anime recommendation prompt template.
    
    This prompt instructs the LLM to provide structured, detailed
    anime recommendations based on user preferences and retrieved context.
    
    Args:
        num_recommendations: Number of anime to recommend (default: 3)
        
    Returns:
        PromptTemplate configured for anime recommendations
    """
    template = f"""You are an expert anime recommender with deep knowledge of anime across all genres, eras, and styles.
Your job is to help users discover the perfect anime based on their preferences.

**IMPORTANT - Data Format:**
The context below contains anime information in this format:
"Title: [ANIME NAME] | Overview: [SYNOPSIS] | Genres: [GENRE LIST]"

You MUST extract the anime title from after "Title:" and before the first "|" symbol.

**Instructions:**
1. Analyze the user's question carefully to understand their preferences (genre, themes, mood, etc.)
2. Use the provided context to find the most relevant anime matches
3. Recommend exactly {num_recommendations} anime titles that best match the user's request
4. For each recommendation, provide:
   - **Title**: The exact anime name (extracted from "Title: X |" in the context)
   - **Synopsis**: A compelling 2-3 sentence plot summary (from the Overview section)
   - **Why It Matches**: Clear explanation of why this anime fits the user's preferences
   - **Genres**: List the main genres (from the Genres section)
5. Present recommendations in a clear, numbered list format
6. If the context doesn't contain relevant anime, be honest and suggest broadening the search

**Important Guidelines:**
- ALWAYS extract and display the actual anime title from the context
- Only recommend anime that appear in the provided context
- Be enthusiastic but honest in your recommendations
- Consider both popular and hidden gems
- If you're unsure, say so - never fabricate information
- Tailor your language to the user's tone (casual, formal, etc.)

**Context (Retrieved Anime Data):**
{{context}}

**User's Question:**
{{input}}

**Your Recommendations:**
"""
    
    logger.debug(f"Created anime prompt template with {num_recommendations} recommendations")
    return PromptTemplate(template=template, input_variables=["context", "input"])


def get_detailed_anime_prompt(num_recommendations: int = 3) -> PromptTemplate:
    """
    Get a detailed anime recommendation prompt with additional metadata.
    
    This prompt asks for more comprehensive information including
    episode count, release year, and target audience.
    
    Args:
        num_recommendations: Number of anime to recommend (default: 3)
        
    Returns:
        PromptTemplate configured for detailed recommendations
    """
    template = f"""You are a professional anime consultant with extensive knowledge of the anime industry.
Help users find their next favorite anime with detailed, informative recommendations.

**Your Task:**
Analyze the user's preferences and recommend exactly {num_recommendations} anime from the provided context.

**For Each Recommendation, Include:**
1. **Title**: Full anime name
2. **Synopsis**: Engaging 3-4 sentence plot summary
3. **Genres**: Primary genres (e.g., Action, Romance, Sci-Fi)
4. **Why It Matches**: Detailed explanation of relevance to user's request
5. **Target Audience**: Who would enjoy this (e.g., "Fans of psychological thrillers")
6. **Standout Features**: What makes this anime special or unique

**Quality Standards:**
- Only recommend anime from the provided context
- Prioritize quality matches over quantity
- Be specific about why each anime fits the user's criteria
- Maintain an engaging, knowledgeable tone
- If no good matches exist, explain why and suggest alternatives

**Available Anime Context:**
{{context}}

**User's Request:**
{{input}}

**Your Expert Recommendations:**
"""
    
    logger.debug(f"Created detailed anime prompt template with {num_recommendations} recommendations")
    return PromptTemplate(template=template, input_variables=["context", "input"])


def get_casual_anime_prompt(num_recommendations: int = 3) -> PromptTemplate:
    """
    Get a casual, friendly anime recommendation prompt.
    
    This prompt uses a more conversational tone, perfect for
    casual users or social media contexts.
    
    Args:
        num_recommendations: Number of anime to recommend (default: 3)
        
    Returns:
        PromptTemplate configured for casual recommendations
    """
    template = f"""Hey there, anime fan! 🎌 I'm here to help you find some awesome anime to watch!

Based on what you're looking for, I'll suggest {num_recommendations} anime that I think you'll love.

**For each anime, I'll tell you:**
- 📺 **Title** - What it's called
- 📖 **What it's about** - Quick plot summary (no spoilers!)
- ✨ **Why you'll like it** - How it matches what you're looking for
- 🎭 **Vibe** - The genres and overall feel

I'll only recommend anime from my database, so you know they're legit. If I can't find a perfect match, I'll let you know!

**Here's what I know about:**
{{context}}

**What you're looking for:**
{{input}}

**My recommendations for you:**
"""
    
    logger.debug(f"Created casual anime prompt template with {num_recommendations} recommendations")
    return PromptTemplate(template=template, input_variables=["context", "input"])


def get_genre_specific_prompt(genre: str, num_recommendations: int = 3) -> PromptTemplate:
    """
    Get a genre-specific anime recommendation prompt.
    
    Optimized for specific genres with tailored evaluation criteria.
    
    Args:
        genre: Target genre (e.g., "action", "romance", "sci-fi")
        num_recommendations: Number of anime to recommend (default: 3)
        
    Returns:
        PromptTemplate configured for genre-specific recommendations
    """
    genre_criteria = {
        "action": "intense fight scenes, compelling battles, and dynamic animation",
        "romance": "emotional depth, character chemistry, and relationship development",
        "comedy": "humor style, comedic timing, and entertainment value",
        "drama": "emotional impact, character development, and storytelling depth",
        "sci-fi": "world-building, technological concepts, and philosophical themes",
        "fantasy": "magic systems, world-building, and imaginative elements",
        "thriller": "suspense, plot twists, and psychological tension",
        "slice-of-life": "realistic portrayal, character interactions, and everyday moments"
    }
    
    criteria = genre_criteria.get(genre.lower(), "overall quality and entertainment value")
    
    template = f"""You are a {genre.title()} anime specialist with deep expertise in this genre.
Help users find the best {genre} anime based on their specific preferences.

**Your Mission:**
Recommend {num_recommendations} outstanding {genre} anime from the provided context.

**Evaluation Criteria for {genre.title()} Anime:**
Focus on: {criteria}

**For Each Recommendation:**
1. **Title**: Anime name
2. **Synopsis**: Brief plot overview (2-3 sentences)
3. **{genre.title()} Elements**: What makes this a great {genre} anime
4. **Why It Matches**: How it fits the user's specific request
5. **Standout Moments**: What makes this anime memorable in the {genre} genre

**Guidelines:**
- Only recommend from the provided context
- Prioritize quality {genre} anime that excel in the criteria above
- Be honest about strengths and potential drawbacks
- Consider both classic and modern {genre} anime

**Available Anime:**
{{context}}

**User's Request:**
{{input}}

**Your {genre.title()} Recommendations:**
"""
    
    logger.debug(f"Created {genre} genre-specific prompt with {num_recommendations} recommendations")
    return PromptTemplate(template=template, input_variables=["context", "input"])


def get_prompt_by_style(
    style: str = "default",
    num_recommendations: int = 3,
    genre: Optional[str] = None
) -> PromptTemplate:
    """
    Get a prompt template based on the desired style.
    
    Args:
        style: Prompt style ("default", "detailed", "casual", "genre")
        num_recommendations: Number of anime to recommend (default: 3)
        genre: Genre for genre-specific prompts (required if style="genre")
        
    Returns:
        PromptTemplate based on the specified style
        
    Raises:
        ValueError: If style is "genre" but no genre is provided
    """
    logger.info(f"Getting prompt template - Style: {style}, Recommendations: {num_recommendations}")
    
    if style == "detailed":
        return get_detailed_anime_prompt(num_recommendations)
    elif style == "casual":
        return get_casual_anime_prompt(num_recommendations)
    elif style == "genre":
        if not genre:
            raise ValueError("Genre must be specified for genre-specific prompts")
        return get_genre_specific_prompt(genre, num_recommendations)
    else:
        return get_anime_prompt(num_recommendations)
