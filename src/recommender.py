from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from src.prompt_template import get_anime_prompt

class AnimeRecommender:
    def __init__(self, retriever, api_key, model_name):
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=0
        )
        self.retriever = retriever
        self.prompt = get_anime_prompt()
        
        # Create chain using LCEL (LangChain Expression Language)
        # This avoids the need for langchain.chains module
        self.chain = (
            RunnableParallel(
                context=self.retriever,
                input=RunnablePassthrough()
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def get_recommendations(self, query: str):
        """
        Get anime recommendations based on user query
        
        Args:
            query: User's question or preference
            
        Returns:
            str: Anime recommendations
        """
        try:
            result = self.chain.invoke(query)
            return result
        except Exception as e:
            raise Exception(f"Error generating recommendations: {str(e)}")