"""
Belief update mechanism for personas.

Implements belief propagation based on dual-graph influence model.
Personas update their beliefs based on neighbors' beliefs weighted by influence.
"""

import numpy as np
from typing import List, Dict, Optional
from persona.persona import Persona
from .dual_graph import DualGraphModel


def calculate_susceptibility(persona: Persona, base_susceptibility: float = 0.5) -> float:
    """
    Calculate persona's susceptibility to influence.
    
    Factors that increase susceptibility:
    - Poor mental health
    - High social media intensity
    - Lower GPA (less confidence)
    
    Args:
        persona: Persona object
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
    gpa_factor = (4.0 - persona.gpa) / 4.0 * 0.1  # Max 0.1 boost for low GPA
    susceptibility += gpa_factor
    
    # Clamp to [0, 1]
    susceptibility = max(0.0, min(1.0, susceptibility))
    
    return susceptibility


def update_belief_vector(current_belief: float,
                        neighbor_beliefs: List[tuple],
                        influence_weights: List[float],
                        susceptibility: float,
                        noise: float = 0.02) -> float:
    """
    Update a single belief value based on neighbor influences.
    
    Uses weighted average with susceptibility modulation.
    
    Args:
        current_belief: Current belief value (0.0-1.0)
        neighbor_beliefs: List of (neighbor_index, neighbor_belief) tuples
        influence_weights: List of influence weights corresponding to neighbors
        susceptibility: Persona's susceptibility to influence (0.0-1.0)
        noise: Random noise factor
        
    Returns:
        Updated belief value (0.0-1.0)
    """
    if not neighbor_beliefs or not influence_weights:
        return current_belief
    
    # Calculate weighted average of neighbor beliefs
    total_weight = 0.0
    weighted_sum = 0.0
    
    for (neighbor_idx, neighbor_belief), weight in zip(neighbor_beliefs, influence_weights):
        weighted_sum += neighbor_belief * weight
        total_weight += weight
    
    if total_weight == 0:
        return current_belief
    
    # Average neighbor belief
    avg_neighbor_belief = weighted_sum / total_weight
    
    # Calculate belief difference
    belief_difference = avg_neighbor_belief - current_belief
    
    # Apply update: weighted by susceptibility and influence strength
    # Scale update by average influence weight (normalized)
    avg_influence = total_weight / len(neighbor_beliefs) if neighbor_beliefs else 0
    update = belief_difference * susceptibility * avg_influence * 0.5  # Scale factor for realism
    
    # Add small random noise
    noise_factor = np.random.normal(0, noise)
    update += noise_factor
    
    # Apply update
    new_belief = current_belief + update
    
    # Clamp to [0, 1]
    new_belief = max(0.0, min(1.0, new_belief))
    
    return new_belief


def belief_to_numeric(belief_text: str, topic: str) -> float:
    """
    Convert belief text to numeric value (0.0-1.0) for a topic.
    
    This is a simplified mapping - in reality, you'd use sentiment analysis
    or more sophisticated NLP to extract belief strength.
    
    Args:
        belief_text: Text description of belief
        topic: Topic name
        
    Returns:
        Numeric belief value (0.0-1.0)
    """
    # Simple keyword-based mapping
    belief_lower = belief_text.lower()
    
    # Positive indicators
    positive_words = ['important', 'critical', 'essential', 'strongly', 'very', 
                     'believe', 'support', 'agree', 'valuable', 'crucial']
    negative_words = ['not', 'unimportant', 'disagree', 'oppose', 'against', 
                     'unnecessary', 'waste', 'reject']
    
    positive_count = sum(1 for word in positive_words if word in belief_lower)
    negative_count = sum(1 for word in negative_words if word in belief_lower)
    
    # Base value from counts
    if positive_count > negative_count:
        base_value = 0.5 + min(0.4, positive_count * 0.1)
    elif negative_count > positive_count:
        base_value = 0.5 - min(0.4, negative_count * 0.1)
    else:
        base_value = 0.5
    
    # Add some randomness for realism
    base_value += np.random.normal(0, 0.1)
    
    return max(0.0, min(1.0, base_value))


def numeric_to_belief_text(numeric_value: float, topic: str) -> str:
    """
    Convert numeric belief value back to text description.
    
    Args:
        numeric_value: Belief value (0.0-1.0)
        topic: Topic name
        
    Returns:
        Text description of belief
    """
    if numeric_value >= 0.7:
        intensity = "strongly believes"
        sentiment = "very important" if "college" in topic.lower() else "very supportive"
    elif numeric_value >= 0.5:
        intensity = "believes"
        sentiment = "important" if "college" in topic.lower() else "supportive"
    elif numeric_value >= 0.3:
        intensity = "somewhat believes"
        sentiment = "somewhat important" if "college" in topic.lower() else "somewhat supportive"
    else:
        intensity = "does not strongly believe"
        sentiment = "not very important" if "college" in topic.lower() else "not very supportive"
    
    return f"The persona {intensity} that {topic} is {sentiment}."


def update_persona_beliefs(persona: Persona,
                          dual_model: DualGraphModel,
                          all_personas: List[Persona],
                          topics: List[str]) -> Persona:
    """
    Update a persona's beliefs based on neighbor influences from dual-graph model.
    
    This function uses the dual-graph model to determine which neighbors can influence
    this persona. Influence only occurs when BOTH conditions are met:
    1. Social connection exists (Graph A: Watts-Strogatz network)
    2. Cognitive similarity exists (Graph B: embedding similarity)
    
    Influence weight = social_connected × cognitive_similarity × reduction_factor
    
    Args:
        persona: Persona to update
        dual_model: Dual-graph model (combines social + cognitive graphs)
        all_personas: All personas (for accessing neighbor beliefs)
        topics: List of belief topics to update
        
    Returns:
        Updated persona with modified beliefs
    """
    persona_idx = all_personas.index(persona)
    
    # Get susceptibility (how open to influence)
    susceptibility = calculate_susceptibility(persona)
    
    # Get influencers using dual-graph model
    # This returns only neighbors that:
    # - Are socially connected (Graph A)
    # - Have cognitive similarity (Graph B)
    # - Have non-zero influence weight (social × cognitive × reduction)
    influencers = dual_model.get_influence_neighbors(persona_idx)
    
    if not influencers:
        return persona  # No influencers, no change
    
    # Create a dictionary for quick lookup
    influencer_dict = {idx: weight for idx, weight in influencers}
    
    # Update each belief topic
    for topic in topics:
        if topic not in persona.beliefs:
            continue
        
        # Convert current belief to numeric
        current_belief_numeric = belief_to_numeric(persona.beliefs[topic], topic)
        
        # Get neighbor beliefs WITH matching influence weights
        # Only include neighbors that:
        # 1. Are in the dual-graph influence network (social + cognitive)
        # 2. Have beliefs for this topic
        neighbor_beliefs = []
        matching_weights = []
        
        for neighbor_idx, influence_weight in influencers:
            neighbor = all_personas[neighbor_idx]
            if topic in neighbor.beliefs:
                neighbor_belief_numeric = belief_to_numeric(neighbor.beliefs[topic], topic)
                neighbor_beliefs.append((neighbor_idx, neighbor_belief_numeric))
                matching_weights.append(influence_weight)  # Match weight to belief
        
        # Update belief using dual-graph influence weights
        # These weights already incorporate:
        # - Social connection (Graph A): binary (connected or not)
        # - Cognitive similarity (Graph B): 0.0-1.0 from embeddings
        # - Reduction factor: 0.15 (15% of raw influence for realism)
        # Final weight = social_connected × cognitive_similarity × 0.15
        if neighbor_beliefs and matching_weights:
            new_belief_numeric = update_belief_vector(
                current_belief_numeric,
                neighbor_beliefs,
                matching_weights,  # Dual-graph influence weights
                susceptibility
            )
            
            # Convert back to text (simplified - in production, regenerate with LLM)
            persona.beliefs[topic] = numeric_to_belief_text(new_belief_numeric, topic)
    
    return persona


def update_all_beliefs(personas: List[Persona],
                      dual_model: DualGraphModel,
                      topics: List[str],
                      random_seed: Optional[int] = None) -> List[Persona]:
    """
    Update beliefs for all personas in one round.
    
    Args:
        personas: List of all personas
        dual_model: Dual-graph model
        topics: List of belief topics to update
        random_seed: Random seed for reproducibility
        
    Returns:
        Updated list of personas
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    updated_personas = []
    for persona in personas:
        updated_persona = update_persona_beliefs(
            persona, dual_model, personas, topics
        )
        updated_personas.append(updated_persona)
    
    return updated_personas

