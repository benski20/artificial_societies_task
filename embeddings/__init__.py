"""
Embedding system for persona similarity calculations.

Provides functions for generating embeddings from persona narratives/beliefs
and calculating similarity between personas.
"""

from .generator import (
    generate_embeddings,
    generate_persona_text,
    add_embeddings_to_personas,
    load_personas_and_generate_embeddings
)
from .similarity import (
    cosine_similarity_matrix,
    pairwise_similarity,
    find_most_similar,
    similarity_statistics
)

__all__ = [
    'generate_embeddings',
    'generate_persona_text',
    'add_embeddings_to_personas',
    'load_personas_and_generate_embeddings',
    'cosine_similarity_matrix',
    'pairwise_similarity',
    'find_most_similar',
    'similarity_statistics'
]
