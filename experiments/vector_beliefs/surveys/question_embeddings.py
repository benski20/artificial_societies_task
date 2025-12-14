"""
Embed survey questions and options into belief space.

Each question and option is embedded, then projected onto the belief space.
"""

import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from surveys.questions import SURVEY_QUESTIONS

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class QuestionEmbedder:
    """
    Embeds survey questions and options into belief space.
    
    Each question+option combination is embedded, then projected onto
    a belief space of specified dimension.
    """
    
    def __init__(self, 
                 belief_dim: int = 16,
                 embedding_model: Optional[str] = None):
        """
        Initialize question embedder.
        
        Args:
            belief_dim: Dimension of belief space
            embedding_model: Sentence transformer model name
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
        
        self.belief_dim = belief_dim
        self.model_name = embedding_model or 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name)
        
        # Embedding dimension from model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Projection matrix: maps embedding space to belief space
        # Initialize randomly but deterministically
        rng = np.random.default_rng(42)
        self.projection_matrix = rng.normal(0, 0.1, (self.embedding_dim, self.belief_dim))
        
        # Cache for question embeddings
        self._question_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Embed all questions
        self._embed_all_questions()
    
    def _embed_all_questions(self):
        """Pre-embed all survey questions and options."""
        for question_id, question_data in SURVEY_QUESTIONS.items():
            question_text = question_data['question']
            options = question_data['options']
            
            # Embed question text
            question_embedding = self.model.encode(question_text, convert_to_numpy=True)
            
            # Embed each option
            option_embeddings = {}
            for option in options:
                # Combine question + option for context
                full_text = f"{question_text} {option}"
                option_embedding = self.model.encode(full_text, convert_to_numpy=True)
                option_embeddings[option] = option_embedding
            
            self._question_embeddings[question_id] = {
                'question': question_embedding,
                'options': option_embeddings
            }
    
    def project_to_belief_space(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project embedding to belief space.
        
        Args:
            embedding: Embedding vector (shape: [embedding_dim])
            
        Returns:
            Belief space vector (shape: [belief_dim])
        """
        # Project: belief = embedding @ projection_matrix
        belief_vector = embedding @ self.projection_matrix
        
        # Normalize to reasonable range
        belief_vector = np.clip(belief_vector, -2.0, 2.0)
        
        return belief_vector.astype(np.float32)
    
    def get_question_belief_vector(self, question_id: str) -> np.ndarray:
        """
        Get question embedding projected to belief space.
        
        Args:
            question_id: Question identifier
            
        Returns:
            Belief space vector for question
        """
        if question_id not in self._question_embeddings:
            raise ValueError(f"Unknown question ID: {question_id}")
        
        question_embedding = self._question_embeddings[question_id]['question']
        return self.project_to_belief_space(question_embedding)
    
    def get_option_belief_vectors(self, question_id: str) -> Dict[str, np.ndarray]:
        """
        Get all option embeddings projected to belief space.
        
        Args:
            question_id: Question identifier
            
        Returns:
            Dictionary mapping option -> belief vector
        """
        if question_id not in self._question_embeddings:
            raise ValueError(f"Unknown question ID: {question_id}")
        
        option_embeddings = self._question_embeddings[question_id]['options']
        return {
            option: self.project_to_belief_space(emb)
            for option, emb in option_embeddings.items()
        }
    
    def get_all_question_ids(self) -> List[str]:
        """Get all question IDs."""
        return list(self._question_embeddings.keys())


def embed_survey_questions(belief_dim: int = 16,
                          embedding_model: Optional[str] = None) -> QuestionEmbedder:
    """
    Create question embedder and embed all questions.
    
    Args:
        belief_dim: Dimension of belief space
        embedding_model: Sentence transformer model name
        
    Returns:
        QuestionEmbedder instance
    """
    return QuestionEmbedder(belief_dim=belief_dim, embedding_model=embedding_model)

