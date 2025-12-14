"""
Similarity calculations for persona embeddings.

Provides functions to calculate similarity between embeddings,
useful for cognitive affinity graph construction.
"""

import numpy as np
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise cosine similarity matrix for embeddings.
    
    Args:
        embeddings: numpy array of shape (n, embedding_dim)
        
    Returns:
        Similarity matrix of shape (n, n) with values in [0, 1]
    """
    similarity = cosine_similarity(embeddings)
    
    # Ensure values are in [0, 1] (cosine similarity is [-1, 1])
    # Normalize to [0, 1] by: (similarity + 1) / 2
    similarity = (similarity + 1) / 2
    
    return similarity


def pairwise_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score in [0, 1]
    """
    # Reshape to 2D for sklearn
    emb1 = embedding1.reshape(1, -1)
    emb2 = embedding2.reshape(1, -1)
    
    similarity = cosine_similarity(emb1, emb2)[0, 0]
    
    # Normalize to [0, 1]
    similarity = (similarity + 1) / 2
    
    return float(similarity)


def find_most_similar(embeddings: np.ndarray,
                     query_idx: int,
                     top_k: int = 10,
                     exclude_self: bool = True) -> List[tuple]:
    """
    Find most similar personas to a query persona.
    
    Args:
        embeddings: All embeddings
        query_idx: Index of query persona
        top_k: Number of similar personas to return
        exclude_self: Whether to exclude the query persona itself
        
    Returns:
        List of (index, similarity_score) tuples, sorted by similarity
    """
    similarity_matrix = cosine_similarity_matrix(embeddings)
    query_similarities = similarity_matrix[query_idx]
    
    # Get top k indices
    if exclude_self:
        query_similarities[query_idx] = -1  # Exclude self
    
    top_indices = np.argsort(query_similarities)[::-1][:top_k]
    
    results = [(idx, float(query_similarities[idx])) for idx in top_indices]
    return results


def similarity_statistics(similarity_matrix: np.ndarray) -> dict:
    """
    Calculate statistics about similarity distribution.
    
    Args:
        similarity_matrix: Pairwise similarity matrix
        
    Returns:
        Dictionary with statistics
    """
    # Exclude diagonal (self-similarity = 1.0)
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    similarities = similarity_matrix[mask]
    
    return {
        'mean': float(np.mean(similarities)),
        'median': float(np.median(similarities)),
        'std': float(np.std(similarities)),
        'min': float(np.min(similarities)),
        'max': float(np.max(similarities)),
        'q25': float(np.percentile(similarities, 25)),
        'q75': float(np.percentile(similarities, 75))
    }

