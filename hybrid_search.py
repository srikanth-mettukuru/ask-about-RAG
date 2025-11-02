# hybrid_search.py
# Hybrid Search Implementation using Reciprocal Rank Fusion (RRF)
# Combines BM25 keyword search and semantic search for better retrieval

import json
from typing import Dict, List, Tuple
from pathlib import Path
from keyword_search import KeywordSearcher
from semantic_search import SemanticSearcher
from config import METADATA_DIR

class HybridSearcher:
    """
    Hybrid search implementation using Reciprocal Rank Fusion.
    
    Combines BM25 keyword search and semantic search to leverage both
    exact keyword matching and semantic similarity for improved retrieval.
    """
    def __init__(self, keyword_index_name: str = None, semantic_index_name: str = None, 
                 rrf_k: int = 60, beta: float = 0.7):
        """
        Initialize hybrid searcher.
        
        Args:
            keyword_index_name: Name of BM25 index (uses default if None)
            semantic_index_name: Name of semantic index (uses default if None)
            rrf_k: RRF parameter (higher values reduce impact of rank differences)
            beta: Weighting parameter (0-1). Higher values favor semantic search.
                  0.7 = 70% semantic, 30% keyword (recommended starting point)
        """
        self.keyword_searcher = KeywordSearcher(keyword_index_name) if keyword_index_name else KeywordSearcher()
        self.semantic_searcher = SemanticSearcher(semantic_index_name) if semantic_index_name else SemanticSearcher()
        self.rrf_k = rrf_k
        self.beta = beta
        self.chunks_data = None
        self.is_ready = False
    
    def load_chunks_data(self) -> bool:
        """Load chunks data for result formatting."""
        try:
            chunks_file = METADATA_DIR / "RAG_Transcript_chunks.json"
            
            if not chunks_file.exists():
                raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'chunks' in data:
                self.chunks_data = data['chunks']
            elif isinstance(data, list):
                self.chunks_data = data
            else:
                raise ValueError("Unexpected file format")
            
            # Initialize semantic searcher
            if not self.semantic_searcher.load_index_and_chunks():
                return False
            
            self.is_ready = True
            return True
            
        except Exception as e:
            print(f"Error loading chunks data: {e}")
            return False
        
    def reciprocal_rank_fusion(self, keyword_results: List[Tuple[int, float]], 
                              semantic_results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Apply Weighted Reciprocal Rank Fusion to combine rankings.
        
        Args:
            keyword_results: List of (chunk_id, score) from keyword search
            semantic_results: List of (chunk_id, similarity) from semantic search
            
        Returns:

            List of (chunk_id, weighted_rrf_score) sorted by weighted RRF score
        """
        # Create rank mappings
         # 'keyword_results' and 'semantic_results' have the 'chunk_id's already sorted from best to worst using the scores, so the 'chunk_id's can get the rank directly from their position in the list
         # the position in the sorted list becomes the rank (0 -> 1, 1 -> 2, etc.)
         # We only care about the relative ranking, not the actual scores
         # RRF uses ranks starting from 1, so we add 1 to the enumerate index. 'enumerate' gives us (index, (chunk_id, score)) where index is 0-based
        keyword_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(keyword_results)}
        semantic_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(semantic_results)}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(keyword_ranks.keys()) | set(semantic_ranks.keys())
        
        # Calculate weighted RRF scores
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            
            # Add weighted keyword contribution (1 - beta)
            if chunk_id in keyword_ranks:
                rrf_score += (1 - self.beta) * (1.0 / (self.rrf_k + keyword_ranks[chunk_id]))
            
            # Add weighted semantic contribution (beta)
            if chunk_id in semantic_ranks:
                rrf_score += self.beta * (1.0 / (self.rrf_k + semantic_ranks[chunk_id]))
            
            rrf_scores[chunk_id] = rrf_score
        
        # Sort by weighted RRF score (descending)
          # '.items()' gives us list of (chunk_id, rrf_score) tuples. Eg: [(48, 0.5), (21, 0.8), (35, 0.3), ...]
          # 'key=lambda x:x[1]' directs the sorting process to use the second element (rrf_score) of the tuple for sorting the list
          # 'reverse=True' sorts in descending order, so highest rrf_scores come first
          # this gives us a list of tuples sorted by rrf_score. each tuple of (chunk_id, rrf_score) stays together during sorting
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results    # [(21, 0.8), (48, 0.5), (35, 0.3), ...]
    
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform hybrid search using RRF and return results with chunk data.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of chunk dictionaries with chunk_id, scores and text
        """
        if not self.is_ready:
            if not self.load_chunks_data():
                raise RuntimeError("Failed to initialize hybrid search system")
            
        # Get results from both search methods
        # Use same top_k to ensure only top results from each method are considered
        
        # Keyword search
        keyword_results = self.keyword_searcher.search(query, top_k)
          # Semantic search (extract results from response structure)
        semantic_response = self.semantic_searcher.search(query, top_k, include_full_text=False)
        semantic_results = [(result['chunk_id'], result['similarity_score']) 
                           for result in semantic_response['results']]
        
        # Apply Reciprocal Rank Fusion
        rrf_results = self.reciprocal_rank_fusion(keyword_results, semantic_results)
        
        # Get top_k results
        top_rrf_results = rrf_results[:top_k]
        
        # Create chunk lookup
        chunk_lookup = {chunk.get('chunk_id', i): chunk 
                       for i, chunk in enumerate(self.chunks_data)}
          # Create keyword and semantic score lookups for detailed results
        keyword_scores = {chunk_id: score for chunk_id, score in keyword_results}
        semantic_scores = {chunk_id: score for chunk_id, score in semantic_results}
        
        # Build results with all scores
        results = []    
        for chunk_id, rrf_score in top_rrf_results:
            if chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id].copy()
                chunk['hybrid_score'] = rrf_score
                chunk['keyword_score'] = keyword_scores.get(chunk_id, 0.0)
                chunk['semantic_score'] = semantic_scores.get(chunk_id, 0.0)
                results.append(chunk)
        
        return results    #[
                          # {chunk_id:41, hybrid_score:0.01622, keyword_score:4.1093, semantic_score:0.3465, text: 'bi-encoder, it's still reasonably fast and...'},
                          # {chunk_id:40, hybrid_score:0.01621, keyword_score:3.1093, semantic_score:0.3765, text: 'document pair and you won't have...'}
                          #]


if __name__ == "__main__":
    # Example usage with weighted RRF
    query = "Tell me about ColBERT."
    top_k = 5
    
    try:
        # Use 70% semantic, 30% keyword weighting (beta=0.7)
        searcher = HybridSearcher(rrf_k=60, beta=0.7)
        results = searcher.search(query, top_k)
        
        print(f"Weighted Hybrid Search Results (70% semantic, 30% keyword)")
        print(f"Query: '{query}'")
        print(f"Results found: {len(results)}")
        print()
        
        for i, chunk in enumerate(results, 1):
            chunk_id = chunk.get('chunk_id', 'Unknown')
            hybrid_score = chunk.get('hybrid_score', 0.0)
            keyword_score = chunk.get('keyword_score', 0.0)
            semantic_score = chunk.get('semantic_score', 0.0)
            text_preview = chunk.get('text', '')[:100] + "..."
            
            print(f"{i}. Chunk {chunk_id}")
            print(f"   Weighted RRF Score: {hybrid_score:.6f}")
            print(f"   Keyword Score: {keyword_score:.4f}")
            print(f"   Semantic Score: {semantic_score:.4f}")
            print(f"   Preview: {text_preview}")
            print()
                
    except Exception as e:
        print(f"Error: {e}")
