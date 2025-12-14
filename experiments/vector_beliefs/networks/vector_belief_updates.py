"""
Belief vector update mechanism using dual-graph model.

Updates belief vectors directly (no text conversion needed).
"""

import numpy as np
from typing import List, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.vector_beliefs.persona.vector_persona import VectorPersona
from networks.dual_graph import DualGraphModel


def calculate_susceptibility(persona: VectorPersona, base_susceptibility: float = 0.5) -> float:
    """
    Calculate persona's susceptibility to influence.
    
    Same factors as original system:
    - Poor mental health → more susceptible
    - High social media → more susceptible
    - Lower GPA → more susceptible
    
    Args:
        persona: VectorPersona object
        base_susceptibility: Base susceptibility level (0.0-1.0)
        
    Returns:
        Susceptibility score (0.0-1.0)
    """
    susceptibility = base_susceptibility
    
    # Mental health adjustment
    mental_health_map = {'Good': -0.2, 'Fair': 0.0, 'Poor': 0.3}
    susceptibility += mental_health_map.get(persona.mental_health, 0.0)
    
    # Social media intensity adjustment
    social_media_map = {
        'Low': -0.1,
        'Moderate': 0.0,
        'High': 0.15,
        'Almost Constant': 0.25
    }
    susceptibility += social_media_map.get(persona.social_media_intensity, 0.0)
    
    # GPA adjustment (lower GPA = more susceptible)
    gpa_factor = (4.0 - persona.gpa) / 4.0 * 0.1
    susceptibility += gpa_factor
    
    # Clamp to [0, 1]
    susceptibility = max(0.0, min(1.0, susceptibility))
    
    return susceptibility


def update_vector_belief(persona: VectorPersona,
                         neighbor_vectors: List[np.ndarray],
                         influence_weights: List[float],
                         susceptibility: float,
                         noise_scale: float = 0.02) -> np.ndarray:
    """
    Update belief vector based on neighbor influences.
    
    Args:
        persona: Persona to update
        neighbor_vectors: List of neighbor belief vectors
        influence_weights: List of influence weights (from dual-graph)
        susceptibility: Persona's susceptibility
        noise_scale: Scale of random noise
        
    Returns:
        Updated belief vector
    """
    if not neighbor_vectors or not influence_weights:
        return persona.get_belief_vector()
    
    if len(neighbor_vectors) != len(influence_weights):
        raise ValueError(
            f"Mismatch: {len(neighbor_vectors)} vectors but {len(influence_weights)} weights"
        )
    
    current_vector = persona.get_belief_vector()
    belief_dim = current_vector.shape[0]
    
    # Calculate weighted average of neighbor vectors
    total_weight = sum(influence_weights)
    if total_weight == 0:
        return current_vector
    
    # Weighted sum
    weighted_sum = np.zeros(belief_dim, dtype=np.float32)
    for neighbor_vec, weight in zip(neighbor_vectors, influence_weights):
        weighted_sum += neighbor_vec * weight
    
    # Average neighbor vector
    avg_neighbor_vector = weighted_sum / total_weight
    
    # Calculate vector difference
    vector_difference = avg_neighbor_vector - current_vector
    
    # Apply update: weighted by susceptibility and influence strength
    avg_influence = total_weight / len(neighbor_vectors)
    update = vector_difference * susceptibility * avg_influence * 0.5  # Scale factor
    
    # Add small random noise
    noise = np.random.normal(0, noise_scale, belief_dim)
    update += noise
    
    # Apply update
    new_vector = current_vector + update
    
    # Clamp to reasonable range
    new_vector = np.clip(new_vector, -2.0, 2.0)
    
    return new_vector.astype(np.float32)


def update_persona_belief_vector(persona: VectorPersona,
                                dual_model: DualGraphModel,
                                all_personas: List[VectorPersona]) -> VectorPersona:
    """
    Update a persona's belief vector based on neighbor influences.
    
    Uses dual-graph model to determine which neighbors can influence.
    
    Args:
        persona: Persona to update
        dual_model: Dual-graph model
        all_personas: All personas (for accessing neighbor vectors)
        
    Returns:
        Updated persona
    """
    persona_idx = all_personas.index(persona)
    
    # Get susceptibility
    susceptibility = calculate_susceptibility(persona)
    
    # Get influencers using dual-graph model
    influencers = dual_model.get_influence_neighbors(persona_idx)
    
    if not influencers:
        return persona  # No influencers, no change
    
    # Extract neighbor vectors and weights
    neighbor_vectors = []
    influence_weights = []
    
    for neighbor_idx, influence_weight in influencers:
        neighbor = all_personas[neighbor_idx]
        neighbor_vector = neighbor.get_belief_vector()
        neighbor_vectors.append(neighbor_vector)
        influence_weights.append(influence_weight)
    
    # Update belief vector
    new_vector = update_vector_belief(
        persona,
        neighbor_vectors,
        influence_weights,
        susceptibility
    )
    
    # Update persona
    persona.update_belief_vector(new_vector)
    
    return persona


def update_all_vector_beliefs(personas: List[VectorPersona],
                             dual_model: DualGraphModel,
                             random_seed: Optional[int] = None) -> List[VectorPersona]:
    """
    Update belief vectors for all personas in one round.
    
    Args:
        personas: List of all personas
        dual_model: Dual-graph model
        random_seed: Random seed for reproducibility
        
    Returns:
        Updated list of personas
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    updated_personas = []
    for persona in personas:
        updated_persona = update_persona_belief_vector(
            persona, dual_model, personas
        )
        updated_personas.append(updated_persona)
    
    return updated_personas

