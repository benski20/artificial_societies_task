"""
Embedding generation for personas.

Generates text embeddings from persona narratives and beliefs using
sentence-transformers models.
"""

import numpy as np
from typing import List, Optional
from pathlib import Path
import json

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from persona.persona import Persona


# Default model: lightweight and fast
DEFAULT_MODEL = 'all-MiniLM-L6-v2'  # 384-dimensional embeddings


def _get_embedding_model(model_name: Optional[str] = None):
    """
    Get or initialize the embedding model.
    
    Args:
        model_name: Name of the model to use (default: all-MiniLM-L6-v2)
        
    Returns:
        SentenceTransformer model
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "sentence-transformers not installed. Install with: pip install sentence-transformers"
        )
    
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    return SentenceTransformer(model_name)


def generate_persona_text(persona: Persona) -> str:
    """
    Generate combined text from persona narrative and beliefs for embedding.
    
    Args:
        persona: Persona object
        
    Returns:
        Combined text string
    """
    parts = []
    
    # Add narrative if available
    if persona.narrative:
        parts.append(persona.narrative)
    
    # Add all beliefs
    if persona.beliefs:
        for topic, belief_text in persona.beliefs.items():
            parts.append(f"{topic}: {belief_text}")
    
    # Fallback to summary if no narrative/beliefs
    if not parts:
        parts.append(persona.get_summary())
    
    return " ".join(parts)


def generate_embeddings(personas: List[Persona],
                       model_name: Optional[str] = None,
                       batch_size: int = 32,
                       show_progress: bool = True) -> np.ndarray:
    """
    Generate embeddings for a list of personas.
    
    Args:
        personas: List of Persona objects
        model_name: Model name (default: all-MiniLM-L6-v2)
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar
        
    Returns:
        numpy array of shape (n_personas, embedding_dim)
    """
    model = _get_embedding_model(model_name)
    
    # Generate text for each persona
    texts = [generate_persona_text(p) for p in personas]
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    return embeddings


def add_embeddings_to_personas(personas: List[Persona],
                               embeddings: np.ndarray) -> List[Persona]:
    """
    Add embeddings to persona objects.
    
    Args:
        personas: List of Persona objects
        embeddings: numpy array of embeddings
        
    Returns:
        List of Personas with embeddings added
    """
    if len(personas) != len(embeddings):
        raise ValueError(f"Personas ({len(personas)}) and embeddings ({len(embeddings)}) length mismatch")
    
    for persona, embedding in zip(personas, embeddings):
        persona.embedding = embedding
    
    return personas


def load_personas_and_generate_embeddings(json_file: str,
                                         model_name: Optional[str] = None) -> tuple:
    """
    Load personas from JSON and generate embeddings.
    
    Args:
        json_file: Path to JSON file with personas
        model_name: Model name for embeddings
        
    Returns:
        Tuple of (personas, embeddings)
    """
    # Load personas from JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert to Persona objects
    personas = []
    for p_dict in data:
        persona = Persona(
            age=p_dict['age'],
            gender=p_dict['gender'],
            race=p_dict['race'],
            sexual_orientation=p_dict['sexual_orientation'],
            family_income=p_dict['family_income'],
            income_percentile=p_dict.get('income_percentile'),
            income_quintile=p_dict.get('income_quintile'),
            first_gen_college=p_dict['first_gen_college'],
            gpa=p_dict['gpa'],
            sports_participation=p_dict['sports_participation'],
            mental_health=p_dict['mental_health'],
            social_media_intensity=p_dict['social_media_intensity'],
            college_intention=p_dict['college_intention'],
            narrative=p_dict.get('narrative'),
            beliefs=p_dict.get('beliefs', {}),
            survey_responses=p_dict.get('survey_responses', {})
        )
        personas.append(persona)
    
    # Generate embeddings
    embeddings = generate_embeddings(personas, model_name=model_name)
    
    # Add embeddings to personas
    personas = add_embeddings_to_personas(personas, embeddings)
    
    return personas, embeddings

