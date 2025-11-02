# BM25 Text Preprocessing Module
# Handles tokenization, normalization, and filtering for BM25 keyword search
# Provides consistent preprocessing pipeline for both indexing and query processing

import re
import string
from typing import List, Set, Optional
from pathlib import Path
import nltk
from config import (
    BM25_USE_STEMMING, BM25_USE_STOPWORDS, 
    BM25_MIN_TERM_LENGTH, BM25_MAX_TERM_LENGTH
)

class BM25TextPreprocessor:
    """
    Text preprocessing pipeline for BM25 keyword search.
    
    Features:
    - Tokenization and normalization
    - Stop word removal (optional)
    - Stemming support (optional)
    - Term length filtering
    - Consistent preprocessing for indexing and queries
    """
    
    def __init__(self, use_stemming: bool = BM25_USE_STEMMING, 
                 use_stopwords: bool = BM25_USE_STOPWORDS):
        """
        Initialize the BM25 text preprocessor.
        
        Args:
            use_stemming: Whether to apply word stemming
            use_stopwords: Whether to remove stop words
        """
        self.use_stemming = use_stemming
        self.use_stopwords = use_stopwords
        self.min_term_length = BM25_MIN_TERM_LENGTH
        self.max_term_length = BM25_MAX_TERM_LENGTH
        
        # Initialize stemmer if needed
        self.stemmer = None
        if self.use_stemming:
            try:
                from nltk.stem import PorterStemmer
                from nltk.corpus import stopwords                
                
                # Download required NLTK data (with error handling)
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    print("Downloading NLTK stopwords...")
                    nltk.download('stopwords', quiet=True)
                
                self.stemmer = PorterStemmer()
            except ImportError:
                print("Warning: NLTK not available. Stemming disabled.")
                self.use_stemming = False
        
        # Initialize stop words set
        self.stop_words = set()
        if self.use_stopwords:
            try:
                if self.use_stemming:
                    # Use NLTK stop words if available
                    from nltk.corpus import stopwords
                    self.stop_words = set(stopwords.words('english'))
                else:
                    # Use basic stop words list
                    self.stop_words = self._get_basic_stopwords()
            except ImportError:
                # Fallback to basic stop words
                self.stop_words = self._get_basic_stopwords()
    
    def _get_basic_stopwords(self) -> Set[str]:
        """Get a basic set of English stop words."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'have', 'had', 'been', 'this', 'they',
            'we', 'you', 'your', 'or', 'but', 'not', 'can', 'could', 'would',
            'should', 'may', 'might', 'must', 'shall', 'do', 'does', 'did',
            'am', 'i', 'me', 'my', 'myself', 'us', 'our', 'ours', 'ourselves'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split on whitespace/punctuation
        # Keep alphanumeric characters and hyphens
        tokens = re.findall(r'\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b', text)
        
        return tokens
    
    def normalize_token(self, token: str) -> Optional[str]:
        """
        Normalize a single token.
        
        Args:
            token: Input token
            
        Returns:
            Normalized token or None if token should be filtered out
        """
        if not token:
            return None
        
        # Convert to lowercase
        token = token.lower().strip()
        
        # Filter by length
        if len(token) < self.min_term_length or len(token) > self.max_term_length:
            return None
        
        # Remove if it's a stop word
        if self.use_stopwords and token in self.stop_words:
            return None
        
        # Apply stemming if enabled
        if self.use_stemming and self.stemmer:
            try:
                token = self.stemmer.stem(token)
            except Exception:
                # If stemming fails, use original token
                pass
        
        return token
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Complete preprocessing pipeline for text.
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed tokens
        """
        # Step 1: Tokenize
        tokens = self.tokenize(text)
        
        # Step 2: Normalize and filter tokens
        processed_tokens = []
        for token in tokens:
            normalized = self.normalize_token(token)
            if normalized is not None:
                processed_tokens.append(normalized)

        return processed_tokens
    
    def get_term_frequencies(self, tokens: List[str]) -> dict:
        """
        Calculate term frequencies from a list of tokens.
        
        Args:
            tokens: List of preprocessed tokens
            
        Returns:
            Dictionary mapping terms to their frequencies
        """
        term_freq = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        return term_freq
    
    def get_preprocessing_stats(self) -> dict:
        """
        Get preprocessing configuration and statistics.
        
        Returns:
            Dictionary with preprocessing settings
        """
        return {
            'use_stemming': self.use_stemming,
            'use_stopwords': self.use_stopwords,
            'min_term_length': self.min_term_length,
            'max_term_length': self.max_term_length,
            'stop_words_count': len(self.stop_words),
            'stemmer_available': self.stemmer is not None
        }


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def preprocess_chunk_text(chunk_text: str, preprocessor: Optional[BM25TextPreprocessor] = None) -> List[str]:
    """
    Convenience function to preprocess chunk text.
    
    Args:
        chunk_text: Text content of a chunk
        preprocessor: Optional preprocessor instance (creates new one if None)
        
    Returns:
        List of preprocessed tokens
    """
    if preprocessor is None:
        preprocessor = BM25TextPreprocessor()
    
    return preprocessor.preprocess_text(chunk_text)


def preprocess_query_text(query: str, preprocessor: Optional[BM25TextPreprocessor] = None) -> List[str]:
    """
    Convenience function to preprocess query text.
    
    Args:
        query: Search query string
        preprocessor: Optional preprocessor instance (creates new one if None)
        
    Returns:
        List of preprocessed query terms
    """
    if preprocessor is None:
        preprocessor = BM25TextPreprocessor()
    
    return preprocessor.preprocess_text(query)


# ==============================================================================
# TESTING AND EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Test the preprocessor
    test_text = """
    This is a sample text for testing the BM25 preprocessor. It contains various 
    punctuation marks, CAPITALIZED words, numbers like 123, hyphenated-words, 
    and common stop words like 'the', 'and', 'is'. Let's see how it processes 
    different types of content including technical terms like cross-encoder and 
    embedding models.
    """
    
    print("BM25 Text Preprocessor Demo")
    print("=" * 40)
    print(f"Original text: {test_text.strip()}")
    print("-" * 40)
    
    # Test with current configuration
    preprocessor = BM25TextPreprocessor()
    tokens = preprocessor.preprocess_text(test_text)
    
    print("Preprocessed tokens:")
    print(tokens)
    print(f"Token count: {len(tokens)}")
    
    # Test term frequencies
    term_freq = preprocessor.get_term_frequencies(tokens)
    print(f"\nTerm frequencies: {term_freq}")

    # Test query preprocessing
    test_query = "What is a cross-encoder model?"
    query_tokens = preprocessor.preprocess_text(test_query)
    print(f"\nQuery: '{test_query}'")
    print(f"Query tokens: {query_tokens}")
    
    # Show preprocessing stats
    stats = preprocessor.get_preprocessing_stats()
    print(f"\nPreprocessing stats: {stats}")
    
    # Test different configurations
    print("\n" + "=" * 40)
    print("Testing different configurations:")
    print("-" * 40)
    
    # Test with stemming enabled (if available)
    preprocessor_stem = BM25TextPreprocessor(use_stemming=True, use_stopwords=False)
    tokens_stem = preprocessor_stem.preprocess_text("running models are processing")
    print(f"With stemming: {tokens_stem}")
    
    # Test with stop words enabled
    preprocessor_stop = BM25TextPreprocessor(use_stemming=False, use_stopwords=True)
    tokens_stop = preprocessor_stop.preprocess_text("this is a test with the stop words")
    print(f"With stop words: {tokens_stop}")
