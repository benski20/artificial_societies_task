"""
Metrics for vector-based belief system.

Reports statistics on belief vectors, projections, and responses.
"""

import numpy as np
from typing import Dict, List
from collections import Counter
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.vector_beliefs.persona.vector_persona import VectorPersona
from surveys.questions import SURVEY_QUESTIONS


def calculate_vector_metrics(personas: List[VectorPersona]) -> Dict:
    """
    Calculate statistics on belief vectors.
    
    Args:
        personas: List of personas
        
    Returns:
        Dictionary with vector statistics
    """
    vectors = np.array([p.get_belief_vector() for p in personas])
    belief_dim = vectors.shape[1]
    
    # Per-dimension statistics
    dim_means = np.mean(vectors, axis=0)
    dim_stds = np.std(vectors, axis=0)
    dim_mins = np.min(vectors, axis=0)
    dim_maxs = np.max(vectors, axis=0)
    
    # Overall statistics
    vector_norms = np.linalg.norm(vectors, axis=1)
    
    # Pairwise cosine similarities
    normalized_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = normalized_vectors @ normalized_vectors.T
    
    # Exclude diagonal
    mask = ~np.eye(len(personas), dtype=bool)
    similarities = similarity_matrix[mask]
    
    return {
        'n_personas': len(personas),
        'belief_dim': belief_dim,
        'vector_norms': {
            'mean': float(np.mean(vector_norms)),
            'std': float(np.std(vector_norms)),
            'min': float(np.min(vector_norms)),
            'max': float(np.max(vector_norms))
        },
        'dimension_statistics': {
            'mean': dim_means.tolist(),
            'std': dim_stds.tolist(),
            'min': dim_mins.tolist(),
            'max': dim_maxs.tolist()
        },
        'pairwise_similarity': {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities))
        }
    }


def calculate_response_metrics(personas: List[VectorPersona],
                               question_id: str) -> Dict:
    """
    Calculate metrics for survey responses.
    
    Args:
        personas: List of personas
        question_id: Question identifier
        
    Returns:
        Dictionary with response metrics
    """
    # Get responses
    responses = [p.survey_responses.get(question_id) for p in personas]
    responses = [r for r in responses if r is not None]
    
    if not responses:
        return {'error': 'No responses found'}
    
    # Response distribution
    response_counts = Counter(responses)
    total = len(responses)
    
    distribution = {option: count / total for option, count in response_counts.items()}
    
    # Probability statistics
    probabilities = []
    for persona in personas:
        if question_id in persona.response_probabilities:
            probs = persona.response_probabilities[question_id]
            probabilities.append(probs)
    
    prob_stats = {}
    if probabilities:
        prob_array = np.array(probabilities)
        prob_stats = {
            'mean_probabilities': prob_array.mean(axis=0).tolist(),
            'std_probabilities': prob_array.std(axis=0).tolist(),
            'entropy': {
                'mean': float(np.mean([-np.sum(p * np.log(p + 1e-8)) for p in probabilities])),
                'std': float(np.std([-np.sum(p * np.log(p + 1e-8)) for p in probabilities]))
            }
        }
    
    return {
        'question_id': question_id,
        'total_responses': total,
        'distribution': distribution,
        'response_counts': dict(response_counts),
        'probability_statistics': prob_stats
    }


def calculate_all_metrics(personas: List[VectorPersona],
                          question_ids: List[str]) -> Dict:
    """
    Calculate all metrics for the system.
    
    Args:
        personas: List of personas
        question_ids: List of question IDs
        
    Returns:
        Dictionary with all metrics
    """
    return {
        'vector_metrics': calculate_vector_metrics(personas),
        'response_metrics': {
            qid: calculate_response_metrics(personas, qid)
            for qid in question_ids
        }
    }

