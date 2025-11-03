# BM25 Keyword Search Module
# Simple keyword search using BM25 algorithm

import json
import math
from typing import Dict, List, Tuple
from pathlib import Path
from bm25_preprocessor import BM25TextPreprocessor
from config import BM25_INDEX_DIR, BM25_DEFAULT_INDEX_NAME, BM25_K1, BM25_B

class KeywordSearcher:
    """Simple BM25 keyword search implementation."""
    
    def __init__(self, index_name: str = BM25_DEFAULT_INDEX_NAME):
        """
        Initialize keyword searcher.
        
        Args:
            index_name: Name of the BM25 index to load
        """
        self.index_name = index_name
        self.preprocessor = BM25TextPreprocessor()
        
        # Index data
        self.inverted_index = {}
        self.document_frequencies = {}
        self.document_lengths = {}
        self.total_documents = 0
        self.average_document_length = 0.0
        self.k1 = BM25_K1  # BM25 parameter k1 to control repeated keyword impact
        self.b = BM25_B   # BM25 parameter b to control length normalization
        
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the BM25 index from disk."""
        index_file = BM25_INDEX_DIR / f"{self.index_name}.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_file}")
        
        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        self.inverted_index = index_data['inverted_index']
        self.document_frequencies = index_data['document_frequencies']
        self.document_lengths = {int(k): v for k, v in index_data['document_lengths'].items()}
        self.total_documents = index_data['total_documents']
        self.average_document_length = index_data['average_document_length']

    def calculate_bm25_score(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calculate BM25 score for a document.
        
        Args:
            query_terms: List of preprocessed query terms
            doc_id: Document ID to score
            
        Returns:
            BM25 score
        """
        if doc_id not in self.document_lengths:
            return 0.0
        
        doc_length = self.document_lengths[doc_id]
        score = 0.0
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
                  # Check if doc_id exists in inverted index for this term
            # Need to handle potential string/int mismatch here too
            doc_id_str = str(doc_id)
            if doc_id_str not in self.inverted_index[term] and doc_id not in self.inverted_index[term]:
                continue
            
            # Get term frequency, handling both int and string doc_ids
            if doc_id_str in self.inverted_index[term]:
                tf = self.inverted_index[term][doc_id_str]
            else:
                tf = self.inverted_index[term][doc_id]
            
            # Document frequency
            df = self.document_frequencies[term]
            
            # IDF calculation
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
            
            # BM25 score calculation
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.average_document_length)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for documents using BM25 scoring.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, score) tuples
        """
        # Preprocess query
        query_terms = self.preprocessor.preprocess_text(query)
        if not query_terms:
            return []
        
        # Find candidate documents
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                term_docs = set(self.inverted_index[term].keys())
                candidate_docs.update(term_docs)
        
        if not candidate_docs:
            return []
        
        # Calculate scores
        scores = []
        for doc_id in candidate_docs:
            # Convert doc_id to int to match document_lengths keys
            doc_id_int = int(doc_id)
            score = self.calculate_bm25_score(query_terms, doc_id_int)
            # Include all scores (even negative ones) since BM25 scores are relative
            scores.append((doc_id_int, score))
        
        # Sort and return top results
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]
        
        return top_results
    
    def search_with_chunks(self, query: str, chunks_data: List[Dict], 
                          top_k: int = 5) -> List[Dict]:
        """
        Search and return results with chunk data.
        
        Args:
            query: Search query
            chunks_data: List of chunk dictionaries
            top_k: Number of results
            
        Returns:
            List of chunks with scores
        """
        search_results = self.search(query, top_k)
        
        # Create chunk lookup
        chunk_lookup = {chunk.get('chunk_id', i): chunk 
                       for i, chunk in enumerate(chunks_data)}
        
        # Build results
        results = []
        for chunk_id, score in search_results:
            if chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id].copy()
                chunk['keyword_score'] = score
                results.append(chunk)

        return results


if __name__ == "__main__":
    # Example usage
    query = "Tell me about ColBERT."
    top_k = 5
    
    try:
        searcher = KeywordSearcher()
        results = searcher.search(query, top_k)
        
        print(f"Query: '{query}'")
        print(f"Results found: {len(results)}")
        
        for i, (doc_id, score) in enumerate(results, 1):
            print(f"{i}. Doc {doc_id}: {score:.4f}")
                
    except Exception as e:
        print(f"Error: {e}")
