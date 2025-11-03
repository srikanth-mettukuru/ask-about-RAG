# Embedding Generation
# Converts user query text to vector embeddings using OpenAI's 'text-embedding-3-small' model. 
        # This model is used because the curated knowledge base was converted to embeddings using the same model, ensuring compatibility for similarity searches.


import openai
import numpy as np
from config import (
    EMBEDDING_MODEL, EMBEDDING_DIMENSION
)

class EmbeddingGenerator:
    """
    Handles conversion of user query text to vector embeddings using OpenAI API.     
    """
    
    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use (default: text-embedding-3-small)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.dimension = EMBEDDING_DIMENSION       
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
                        
            # Extract embedding vector
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise