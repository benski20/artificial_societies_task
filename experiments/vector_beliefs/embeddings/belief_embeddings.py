"""
Generate embeddings from belief vectors for dual-graph model.

Since the dual-graph model uses embeddings for cognitive similarity,
we need to convert belief vectors to embeddings.
"""

import numpy as np
from typing import List, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.vector_beliefs.persona.vector_persona import VectorPersona

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def generate_embeddings_from_vectors(personas: List[VectorPersona],
                                    embedding_dim: int = 384,
                                    model_name: Optional[str] = None) -> np.ndarray:
    """
    Generate embeddings from belief vectors.
    
    Uses a learned projection or direct mapping from belief space
    to embedding space for compatibility with dual-graph model.
    
    Args:
        personas: List of VectorPersona objects
        embedding_dim: Desired embedding dimension
        model_name: Model name (for consistency, not used directly)
        
    Returns:
        numpy array of shape (n_personas, embedding_dim)
    """
    belief_dim = personas[0].belief_dim
    
    # Create projection matrix: belief_dim -> embedding_dim
    # Initialize deterministically
    rng = np.random.default_rng(42)
    projection_matrix = rng.normal(0, 0.1, (belief_dim, embedding_dim))
    
    # Normalize columns
    projection_matrix = projection_matrix / (np.linalg.norm(projection_matrix, axis=0) + 1e-8)
    
    # Project belief vectors to embedding space
    embeddings = []
    for persona in personas:
        belief_vector = persona.get_belief_vector()
        embedding = belief_vector @ projection_matrix
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Normalize embeddings (for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    return embeddings


def add_embeddings_to_vector_personas(personas: List[VectorPersona],
                                     embeddings: np.ndarray) -> List[VectorPersona]:
    """
    Add embeddings to personas (for compatibility, not stored in VectorPersona).
    
    Args:
        personas: List of personas
        embeddings: Embeddings array
        
    Returns:
        List of personas (embeddings are not stored, just returned)
    """
    if len(personas) != len(embeddings):
        raise ValueError(f"Mismatch: {len(personas)} personas but {len(embeddings)} embeddings")
    
    # Note: VectorPersona doesn't store embeddings, but we return them
    # for use with dual-graph model
    return personas

