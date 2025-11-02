# Embedding Generation Module for RAG
# Converts text chunks to vector embeddings using OpenAI's text-embedding-3-small model
# Handles batch processing, rate limiting, and cost tracking

import openai
import numpy as np
import time
import json
from typing import List, Dict, Tuple
from pathlib import Path
from config import (
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, BATCH_SIZE, 
    MAX_RETRIES, RATE_LIMIT_DELAY
)

class EmbeddingGenerator:
    """
    Handles conversion of text chunks to vector embeddings using OpenAI API.
    
    Features:
    - Batch processing for efficiency (reduce API calls)
    - Rate limiting to respect API limits
    - Error handling and retry logic
    - Cost tracking (tokens processed)
    - Progress reporting for large datasets
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
        self.total_tokens = 0
        self.total_cost = 0.0  # Track estimated cost
        
        # Cost per 1K tokens for text-embedding-3-small (as of 2024)
        self.cost_per_1k_tokens = 0.00002
        
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
            
            # Track usage
            tokens_used = response.usage.total_tokens
            self.total_tokens += tokens_used
            self.total_cost += (tokens_used / 1000) * self.cost_per_1k_tokens
            
            # Extract embedding vector
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts in a single API call.
        More efficient than individual calls.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.array([])
        
        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float"
                )
                
                # Track usage
                tokens_used = response.usage.total_tokens
                self.total_tokens += tokens_used
                self.total_cost += (tokens_used / 1000) * self.cost_per_1k_tokens
                
                # Extract embeddings
                embeddings = []
                for data_point in response.data:
                    embeddings.append(data_point.embedding)
                
                return np.array(embeddings, dtype=np.float32)
                
            except Exception as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    print(f"Failed to generate embeddings after {MAX_RETRIES} retries: {e}")
                    raise
                
                print(f"Retry {retries}/{MAX_RETRIES} after error: {e}")
                time.sleep(RATE_LIMIT_DELAY * retries)  # Exponential backoff
    
    def process_chunks_to_embeddings(self, chunks: List[Dict]) -> Tuple[np.ndarray, List[int]]:
        """
        Convert a list of text chunks to embeddings with batch processing.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'chunk_id' keys
            
        Returns:
            tuple: (embeddings_array, chunk_ids_list)
        """
        print(f"Processing {len(chunks)} chunks to embeddings...")
        print(f"Using model: {self.model}")
        print(f"Batch size: {BATCH_SIZE}")
        
        all_embeddings = []
        chunk_ids = []
        
        # Process in batches for efficiency
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            batch_texts = [chunk['text'] for chunk in batch_chunks]
            batch_ids = [chunk['chunk_id'] for chunk in batch_chunks]
            
            print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(chunks) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            # Generate embeddings for this batch
            batch_embeddings = self.generate_embeddings_batch(batch_texts)
            
            all_embeddings.append(batch_embeddings)
            chunk_ids.extend(batch_ids)
            
            # Rate limiting between batches
            if i + BATCH_SIZE < len(chunks):
                time.sleep(RATE_LIMIT_DELAY)
        
        # Combine all batches
        final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        print(f"Embedding generation complete!")
        print(f"Total embeddings: {len(final_embeddings)}")
        print(f"Embedding dimension: {final_embeddings.shape[1] if len(final_embeddings) > 0 else 'N/A'}")
        print(f"Total tokens processed: {self.total_tokens:,}")
        print(f"Estimated cost: ${self.total_cost:.4f}")
        
        return final_embeddings, chunk_ids
    
    def get_usage_stats(self) -> Dict:
        """Get embedding generation statistics."""
        return {
            'total_tokens': self.total_tokens,
            'estimated_cost': self.total_cost,
            'model_used': self.model,
            'cost_per_1k_tokens': self.cost_per_1k_tokens
        }
