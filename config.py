# RAG Configuration
# Configuration settings for the RAG implementation

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
METADATA_DIR = DATA_DIR / "metadata"

# Ensure directories exist
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# FAISS Configuration  
FAISS_INDEX_DIR = EMBEDDINGS_DIR  # Use the existing embeddings directory
SEARCH_TOP_K = 5

# BM25 Configuration
BM25_K1 = 1.5          # Term frequency saturation parameter (1.2-2.0 typical range)
BM25_B = 0.75          # Field length normalization parameter (0.0-1.0 range)
BM25_INDEX_DIR = DATA_DIR / "bm25_indices"  # Storage location for BM25 indexes

# BM25 Text Preprocessing
BM25_USE_STEMMING = False      # Whether to apply stemming to terms
BM25_USE_STOPWORDS = False     # Keep all words for technical content
BM25_MIN_TERM_LENGTH = 1       # Minimum character length for terms
BM25_MAX_TERM_LENGTH = 50      # Maximum character length for terms

# BM25 Index Settings
BM25_DEFAULT_INDEX_NAME = "rag_transcript_bm25"    # Default BM25 index name

# Ensure BM25 directory exists
BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
