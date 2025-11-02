# RAG Configuration
# Configuration settings for the RAG implementation

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
METADATA_DIR = DATA_DIR / "metadata"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Chunking configuration
CHUNK_SIZE = 500  # words
OVERLAP_SIZE = 50  # words
MIN_CHUNK_SIZE = 100  # minimum words per chunk

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 100  # Process chunks in batches
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.0  # Seconds between API calls

# FAISS Configuration  
FAISS_INDEX_DIR = EMBEDDINGS_DIR  # Use the existing embeddings directory
SEARCH_TOP_K = 5

# Search configuration
MAX_CHUNKS_PER_QUERY = 5
SIMILARITY_THRESHOLD = 0.7

# Processing configuration
BATCH_SIZE = 100  # for embedding generation
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1  # seconds between API calls

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
BM25_SAVE_VOCABULARY = True    # Whether to save vocabulary separately
BM25_SAVE_STATISTICS = True    # Whether to save index statistics

# Ensure BM25 directory exists
BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
