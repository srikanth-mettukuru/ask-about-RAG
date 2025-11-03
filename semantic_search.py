# Semantic Search Implementation for RAG
# Provides query processing, embedding generation, and similarity search using FAISS
# Returns ranked relevant chunks with metadata for knowledge retrieval

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from config import EMBEDDINGS_DIR, METADATA_DIR, SEARCH_TOP_K, EMBEDDING_MODEL
from embedding_generator import EmbeddingGenerator
from faiss_store import FAISSVectorStore

class SemanticSearcher:
    """
    Semantic search engine for RAG implementation.
    
    Features:
    - Query embedding generation
    - FAISS similarity search
    - Result ranking and filtering
    - Chunk content retrieval
    - Search analytics and debugging
    """
    def __init__(self, index_name: str = "rag_transcript_index"):
        """
        Initialize semantic searcher.
        
        Args:
            index_name: Name of the FAISS index to load
        """
        # Load environment variables (including OpenAI API key)
        load_dotenv()
        
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        
        self.index_name = index_name
        self.embedding_generator = EmbeddingGenerator(api_key=openai_api_key)
        self.vector_store = FAISSVectorStore()
        self.chunks_data = None  # Will store full chunk content
        self.is_ready = False

    def load_index_and_chunks(self) -> bool:
        """
        Load FAISS index and associated chunk data.
        Uses case-insensitive file search to handle filename variations.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index            
            if not self.vector_store.load_index(self.index_name):
                print("ERROR: Failed to load FAISS index")
                return False
            
            # Case-insensitive search for chunks file in METADATA_DIR
            base_name = self.index_name.replace('_index', '')
            chunks_filename_pattern = f"{base_name}_chunks.json"            
            
            # Search for matching file (case-insensitive)
            chunks_file = None
            if METADATA_DIR.exists():                
                file_count = 0
                for file_path in METADATA_DIR.iterdir():
                    file_count += 1                    
                    if file_path.is_file():                        
                        if file_path.name.lower() == chunks_filename_pattern.lower():
                            chunks_file = file_path                            
                            break                
            else:
                print(f"ERROR: METADATA_DIR does not exist: {METADATA_DIR}")
            
            if chunks_file is None:
                print(f"ERROR: Chunks metadata file not found. Looking for: {chunks_filename_pattern}")
                print(f"Available files in {METADATA_DIR}:")
                if METADATA_DIR.exists():
                    for file_path in METADATA_DIR.iterdir():
                        if file_path.is_file():
                            print(f"  - {file_path.name} (suffix: {file_path.suffix})")
                return False
            
            # Load chunk content from found metadata file            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)                
                if 'chunks' in metadata:
                    self.chunks_data = metadata['chunks']                    
                else:                    
                    return False
            
            self.is_ready = True
            print(f"SUCCESS: Search system ready!")
            print(f"  Index: {self.vector_store.index.ntotal} vectors")
            print(f"  Chunks: {len(self.chunks_data)} text chunks")
            print(f"  Loaded from: {chunks_file.name}")
            return True
            
        except Exception as e:
            print(f"ERROR: Exception in load_index_and_chunks: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_query(self, query: str) -> str:
        """
        Clean and prepare query for embedding generation.
        
        Args:
            query: Raw user query
            
        Returns:
            Cleaned query string
        """
        # Basic query cleaning (similar to document preprocessing)
        query = query.strip()
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Optional: Add query expansion or preprocessing here
        # For example: stemming, stopword removal, etc.
        
        return query
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for search query.
        
        Args:
            query: Preprocessed query string
              Returns:
            Query embedding vector
        """
        try:
            # Generate embedding using the same model used for documents
            embedding = self.embedding_generator.generate_embedding(query)
            return embedding
        
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            raise
    
    def search_similar_chunks(self, query_embedding: np.ndarray, 
                            top_k: int = SEARCH_TOP_K) -> Tuple[List[float], List[int]]:
        """
        Search for similar chunks using FAISS.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (similarity_scores, chunk_ids)
        """
        if not self.is_ready:
            raise ValueError("Search system not ready. Call load_index_and_chunks() first.")
        
        return self.vector_store.similarity_search(query_embedding, top_k)
    
    def retrieve_chunk_content(self, chunk_ids: List[int]) -> List[Dict]:
        """
        Retrieve full content for given chunk IDs.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            List of chunk dictionaries with full content
        """
        chunks = []
        for chunk_id in chunk_ids:
            if 0 <= chunk_id < len(self.chunks_data):
                chunks.append(self.chunks_data[chunk_id])
            else:
                print(f"Warning: Chunk ID {chunk_id} out of range")
        
        return chunks
    
    def format_search_results(self, similarities: List[float], 
                            chunk_ids: List[int], 
                            include_full_text: bool = False) -> List[Dict]:
        """
        Format search results with metadata and optional full text.
        
        Args:
            similarities: Similarity scores from FAISS
            chunk_ids: Corresponding chunk IDs
            include_full_text: Whether to include full chunk text
            
        Returns:
            Formatted results with metadata
        """
        chunks = self.retrieve_chunk_content(chunk_ids)
        results = []
        
        for similarity, chunk_id, chunk_data in zip(similarities, chunk_ids, chunks):
            result = {
                'chunk_id': chunk_id,
                'similarity_score': similarity,
                'word_count': chunk_data.get('word_count', 0),
                'token_count': chunk_data.get('token_count', 0),
                'preview': chunk_data.get('text', '')[:200] + '...' if len(chunk_data.get('text', '')) > 200 else chunk_data.get('text', '')
            }
            
            if include_full_text:
                result['full_text'] = chunk_data.get('text', '')
            
            results.append(result)
        
        return results
    
    def search(self, query: str, top_k: int = SEARCH_TOP_K, 
              include_full_text: bool = False) -> Dict:
        """
        Complete semantic search pipeline.
        
        Args:
            query: User query string
            top_k: Number of results to return
            include_full_text: Whether to include full chunk text in results
            
        Returns:
            Search results with metadata and statistics
        """
        if not self.is_ready:
            if not self.load_index_and_chunks():
                raise RuntimeError("Failed to initialize search system") 
       
        
        # Step 1: Preprocess query
        clean_query = self.preprocess_query(query)
        
        # Step 2: Generate query embedding
        query_embedding = self.generate_query_embedding(clean_query)
        
        # Step 3: Search similar chunks
        similarities, chunk_ids = self.search_similar_chunks(query_embedding, top_k)
        
        # Step 4: Format results
        results = self.format_search_results(similarities, chunk_ids, include_full_text)
        
        # Step 5: Create comprehensive response
        search_response = {
            'query': {
                'original': query,
                'processed': clean_query,
                'embedding_dimension': len(query_embedding)
            },
            'search_params': {
                'top_k': top_k,
                'similarity_metric': self.vector_store.similarity_metric,
                'total_chunks_available': len(self.chunks_data)
            },
            'results': results,
            'statistics': {
                'results_found': len(results),
                'highest_similarity': max(similarities) if similarities else 0.0,
                'lowest_similarity': min(similarities) if similarities else 0.0,
                'average_similarity': sum(similarities) / len(similarities) if similarities else 0.0
            }
        }
        
        return search_response
    
    def display_search_results(self, search_response: Dict):
        """
        Display formatted search results.
        
        Args:
            search_response: Complete search response from search() method
        """
        results = search_response['results']
        stats = search_response['statistics']
        
        if not results:
            print("No results found.")
            return
        
        print(f"Found {stats['results_found']} results:")
        print(f"Similarity range: {stats['lowest_similarity']:.3f} - {stats['highest_similarity']:.3f}")
        print(f"Average similarity: {stats['average_similarity']:.3f}")
        print()
        
        for i, result in enumerate(results, 1):
            print(f"Result #{i}")
            print(f"  Chunk ID: {result['chunk_id']}")
            print(f"  Similarity: {result['similarity_score']:.3f}")
            print(f"  Words: {result['word_count']}, Tokens: {result['token_count']}")
            print(f"  Preview: {result['preview']}")
            print()
    
    def print_system_stats(self):
        """Print system statistics and configuration."""
        print()
        print("=== System Statistics ===")
        if self.is_ready:
            print(f"Status: Ready")
            print(f"Index: {self.index_name}")
            print(f"Vectors in FAISS: {self.vector_store.index.ntotal}")
            print(f"Chunks available: {len(self.chunks_data)}")
            print(f"Embedding model: {EMBEDDING_MODEL}")
            print(f"Similarity metric: {self.vector_store.similarity_metric}")
            print(f"Default top_k: {SEARCH_TOP_K}")
        else:
            print("Status: Not ready")
        print("=" * 25)


# ==============================================================================
# TESTING AND EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Configure your test parameters here
    TEST_QUERY = "Tell me about ColBERT."
    TOP_K = 5
    
    print("RAG Semantic Search Demo")
    print("=" * 40)
    print(f"Query: {TEST_QUERY}")
    print(f"Top K: {TOP_K}")
    print("-" * 40)
    
    try:
        # Initialize and run search
        searcher = SemanticSearcher()
        results = searcher.search(TEST_QUERY, top_k=TOP_K, include_full_text=False)
        
        # Display results
        searcher.display_search_results(results)
        
        # Print system statistics
        searcher.print_system_stats()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
