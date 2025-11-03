# FAISS Vector Store Implementation for RAG
# Handles vector storage, indexing, and similarity search using FAISS
# Separates vector storage from metadata for optimal performance

import faiss
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from config import FAISS_INDEX_DIR, EMBEDDING_DIMENSION, SEARCH_TOP_K

class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    
    Features:
    - Optimized vector storage and search
    - Index persistence (save/load from disk)
    - Flexible similarity metrics (cosine, L2)
    - Separate metadata management
    - Index statistics and health checks
    """
    
    def __init__(self, dimension: int = EMBEDDING_DIMENSION, 
                 similarity_metric: str = "cosine"):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Vector dimension (1536 for text-embedding-3-small)
            similarity_metric: "cosine", "l2", or "ip" (inner product)
        """
        self.dimension = dimension
        self.similarity_metric = similarity_metric
        self.index = None
        self.chunk_mapping = []  # Maps FAISS index positions to chunk IDs
        self.is_trained = False
        
    def create_index(self, similarity_metric: str = None) -> faiss.Index:
        """
        Create a new FAISS index based on similarity metric.
        
        Args:
            similarity_metric: Override default similarity metric
            
        Returns:
            FAISS index object
        """
        metric = similarity_metric or self.similarity_metric
        if metric == "cosine":
            # For cosine similarity, we use inner product with normalized vectors
            # Mathematical insight: cosine(A,B) = (A·B)/(||A||×||B||) 
            # When vectors are normalized (||A||=||B||=1), cosine(A,B) = A·B
            index = faiss.IndexFlatIP(self.dimension)
        elif metric == "l2":
            # L2 (Euclidean) distance - uses different FAISS algorithm
            index = faiss.IndexFlatL2(self.dimension)
        elif metric == "ip":
            # Raw inner product (dot product) without normalization
            index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
        
        print(f"Created FAISS index: {type(index).__name__} with {metric} similarity")
        return index
    
    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: List[int]):
        """
        Add embeddings to the FAISS index.
        
        Args:
            embeddings: numpy array of shape (n_vectors, dimension)
            chunk_ids: List of chunk IDs corresponding to each embedding
        """
        if self.index is None:
            self.index = self.create_index()
        
        # Normalize vectors for cosine similarity
        if self.similarity_metric == "cosine":
            # Normalize embeddings to unit length for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)  # Avoid division by zero
        
        # Add to FAISS index
        embeddings = embeddings.astype(np.float32)  # FAISS requires float32
        self.index.add(embeddings)
        
        # Update chunk mapping
        self.chunk_mapping.extend(chunk_ids)
        self.is_trained = True
        
        print(f"Added {len(embeddings)} embeddings to FAISS index")
        print(f"Total vectors in index: {self.index.ntotal}")
    
    def similarity_search(self, query_embedding: np.ndarray, 
                         k: int = SEARCH_TOP_K) -> Tuple[List[float], List[int]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_embedding: Query vector of shape (dimension,)
            k: Number of similar vectors to return
            
        Returns:
            tuple: (similarity_scores, chunk_ids)
        """
        if self.index is None or not self.is_trained:
            raise ValueError("Index not created or trained. Add embeddings first.")
        
        # Prepare query vector
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize for cosine similarity
        if self.similarity_metric == "cosine":
            norm = np.linalg.norm(query_vector)
            query_vector = query_vector / (norm + 1e-8)
        
        # Search
        similarities, faiss_indices = self.index.search(query_vector, k)
        
        # Map FAISS indices back to chunk IDs
        chunk_ids = []
        valid_similarities = []
        
        for i, faiss_idx in enumerate(faiss_indices[0]):
            if faiss_idx != -1 and faiss_idx < len(self.chunk_mapping):
                chunk_ids.append(self.chunk_mapping[faiss_idx])
                valid_similarities.append(float(similarities[0][i]))
        
        return valid_similarities, chunk_ids
    
    def save_index(self, base_name: str) -> Dict[str, str]:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            base_name: Base filename (without extension)
            
        Returns:
            Dictionary with file paths created
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        # Create directory if it doesn't exist
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        
        # File paths
        index_path = FAISS_INDEX_DIR / f"{base_name}.faiss"
        mapping_path = FAISS_INDEX_DIR / f"{base_name}_mapping.json"
        config_path = FAISS_INDEX_DIR / f"{base_name}_config.json"
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save chunk mapping
        with open(mapping_path, 'w') as f:
            json.dump(self.chunk_mapping, f)
        
        # Save configuration
        config = {
            'dimension': self.dimension,
            'similarity_metric': self.similarity_metric,
            'index_type': type(self.index).__name__,
            'total_vectors': self.index.ntotal,
            'created_timestamp': time.time()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"FAISS index saved successfully!")
        print(f"Index file: {index_path}")
        print(f"Mapping file: {mapping_path}")
        print(f"Config file: {config_path}")
        
        return {
            'index_path': str(index_path),
            'mapping_path': str(mapping_path),
            'config_path': str(config_path)
        }
    
    def load_index(self, base_name: str) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            base_name: Base filename (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # File paths
            index_path = FAISS_INDEX_DIR / f"{base_name}.faiss"
            mapping_path = FAISS_INDEX_DIR / f"{base_name}_mapping.json"
            config_path = FAISS_INDEX_DIR / f"{base_name}_config.json"
            
            # Check if files exist
            if not all(p.exists() for p in [index_path, mapping_path, config_path]):
                print("Index files not found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load chunk mapping
            with open(mapping_path, 'r') as f:
                self.chunk_mapping = json.load(f)
            
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.dimension = config['dimension']
                self.similarity_metric = config['similarity_metric']
            
            self.is_trained = True
            
            print(f"FAISS index loaded successfully!")
            print(f"Vectors: {self.index.ntotal}")
            print(f"Dimension: {self.dimension}")
            print(f"Similarity metric: {self.similarity_metric}")
            
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        if self.index is None:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'ready' if self.is_trained else 'empty',
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'similarity_metric': self.similarity_metric,
            'index_type': type(self.index).__name__,
            'chunk_mapping_size': len(self.chunk_mapping)
        }
