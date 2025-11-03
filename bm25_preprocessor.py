# BM25 Text Preprocessing
# Preprocessing pipeline for BM25 keyword search -- converts user query text into normalized tokens for retrieval of relevant documents.

import re
from typing import List, Set, Optional
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