"""
Survey response generation logic.

Provides functions for generating realistic survey responses based on
persona attributes, using both probabilistic models and LLM augmentation.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from persona.persona import Persona
from .questions import SURVEY_QUESTIONS, get_survey_question
from llm.prompts import generate_survey_response_prompt
from llm.augmentation import _call_llm


def _sigmoid(x: float) -> float:
    """Sigmoid function for logistic modeling."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def _normalize_income(income: float) -> float:
    """Normalize income to [0, 1] range."""
    # Use percentiles from research data
    min_income = 25000  # 20th percentile
    max_income = 111000  # 80th percentile
    normalized = (income - min_income) / (max_income - min_income)
    return np.clip(normalized, 0, 1)


def _get_belief_strength(persona: Persona, question_id: str) -> float:
    """
    Get belief strength for a survey question from persona's beliefs.
    
    Args:
        persona: Persona object
        question_id: Survey question ID
        
    Returns:
        Belief strength (0.0-1.0), or None if no matching belief
    """
    from networks.belief_updates import belief_to_numeric
    
    # Map question IDs to belief topics
    question_to_topic = {
        'college_importance': 'college importance and future plans',
        'social_media_stress': 'social media impact on daily life',
        'school_start_times': 'school start times and sleep'
    }
    
    topic = question_to_topic.get(question_id)
    if not topic or topic not in persona.beliefs:
        return None
    
    return belief_to_numeric(persona.beliefs[topic], topic)


def probabilistic_respond(persona: Persona, question_id: str,
                         random_state: Optional[np.random.Generator] = None) -> str:
    """
    Generate survey response using probabilistic model based on persona attributes.
    
    Args:
        persona: Persona object
        question_id: Survey question identifier
        random_state: Optional random number generator
        
    Returns:
        Selected response option
    """
    if random_state is None:
        random_state = np.random.default_rng()
    
    question_data = get_survey_question(question_id)
    options = question_data['options']
    
    # Get belief strength if available (from updated beliefs)
    belief_strength = _get_belief_strength(persona, question_id)
    
    # Calculate logit based on question type
    if question_id == 'college_importance':
        # Higher GPA, income, and college intention → more important
        base_logit = (
            1.5 * (persona.gpa / 4.0) +  # GPA contribution
            0.8 * _normalize_income(persona.family_income) +  # Income contribution
            0.5 * (1 if persona.college_intention in ['4-year college', '2-year college'] else 0) +
            -0.3 * (1 if persona.first_gen_college else 0)  # First-gen may have concerns
        )
        
        # Incorporate belief strength if available (beliefs override base attributes)
        if belief_strength is not None:
            # Map belief strength (0-1) to logit scale (-2 to +2)
            belief_logit = (belief_strength - 0.5) * 4.0  # Scale to [-2, +2]
            logit = 0.3 * base_logit + 0.7 * belief_logit  # 70% weight on beliefs
        else:
            logit = base_logit
        
        logit += random_state.normal(0, 0.3)  # Noise
        
        # Map logit to response probabilities
        # Higher logit → more important
        probs = [
            _sigmoid(logit - 1.0),  # Very Important
            _sigmoid(logit - 0.3) - _sigmoid(logit - 1.0),  # Somewhat Important
            _sigmoid(logit + 0.3) - _sigmoid(logit - 0.3),  # Not Very Important
            1 - _sigmoid(logit + 0.3)  # Not Important At All
        ]
        
    elif question_id == 'social_media_stress':
        # Higher social media intensity, worse mental health → more stress
        intensity_map = {'Low': 0, 'Moderate': 1, 'High': 2, 'Almost Constant': 3}
        mental_health_map = {'Good': 0, 'Fair': 1, 'Poor': 2}
        
        base_logit = (
            0.6 * (intensity_map.get(persona.social_media_intensity, 1) / 3.0) +
            0.8 * (mental_health_map.get(persona.mental_health, 1) / 2.0)
        )
        
        # Incorporate belief strength if available
        if belief_strength is not None:
            belief_logit = (belief_strength - 0.5) * 4.0
            logit = 0.3 * base_logit + 0.7 * belief_logit
        else:
            logit = base_logit
        
        logit += random_state.normal(0, 0.3)
        
        # Map logit to 5-point scale (Strongly Agree to Strongly Disagree)
        # Use cumulative probabilities
        thresholds = [-1.2, -0.4, 0.4, 1.2]
        cum_probs = [_sigmoid(logit - t) for t in thresholds]
        cum_probs.append(1.0)  # Last cumulative probability is always 1
        
        # Convert to individual probabilities
        probs = [cum_probs[0]]  # First probability
        for i in range(1, len(cum_probs)):
            probs.append(cum_probs[i] - cum_probs[i-1])
        
    elif question_id == 'school_start_times':
        # More stress, worse mental health → more support for later start times
        mental_health_map = {'Good': 0, 'Fair': 1, 'Poor': 2}
        intensity_map = {'Low': 0, 'Moderate': 1, 'High': 2, 'Almost Constant': 3}
        
        base_logit = (
            0.7 * (mental_health_map.get(persona.mental_health, 1) / 2.0) +
            0.4 * (intensity_map.get(persona.social_media_intensity, 1) / 3.0) +
            -0.3 * (1 if persona.sports_participation else 0)  # Athletes may prefer earlier
        )
        
        # Incorporate belief strength if available
        if belief_strength is not None:
            belief_logit = (belief_strength - 0.5) * 4.0
            logit = 0.3 * base_logit + 0.7 * belief_logit
        else:
            logit = base_logit
        
        logit += random_state.normal(0, 0.3)
        
        # Map to 5-point scale using cumulative probabilities
        thresholds = [-1.2, -0.4, 0.4, 1.2]
        cum_probs = [_sigmoid(logit - t) for t in thresholds]
        cum_probs.append(1.0)  # Last cumulative probability is always 1
        
        # Convert to individual probabilities
        probs = [cum_probs[0]]  # First probability
        for i in range(1, len(cum_probs)):
            probs.append(cum_probs[i] - cum_probs[i-1])
        
    else:
        # Default: uniform distribution
        probs = [1.0 / len(options)] * len(options)
    
    # Ensure probabilities are non-negative and normalize
    probs = np.array(probs)
    probs = np.maximum(probs, 0)  # Ensure non-negative
    probs_sum = probs.sum()
    if probs_sum > 0:
        probs = probs / probs_sum
    else:
        # Fallback to uniform if all probabilities are zero/negative
        probs = np.ones(len(options)) / len(options)
    
    # Sample from distribution
    selected_idx = random_state.choice(len(options), p=probs)
    return options[selected_idx]


def respond_to_survey(persona: Persona, question_id: str,
                     use_llm: bool = False,
                     model_name: Optional[str] = None,
                     random_state: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """
    Generate survey response for a persona.
    
    Can use probabilistic model, LLM, or hybrid approach.
    
    Args:
        persona: Persona object
        question_id: Survey question identifier
        use_llm: Whether to use LLM for response (default: False, uses probabilistic)
        model_name: LLM model name (if use_llm=True)
        random_state: Optional random number generator
        
    Returns:
        Dictionary with 'response' and optionally 'explanation'
    """
    question_data = get_survey_question(question_id)
    
    if use_llm:
        # Use LLM to generate response
        prompt = generate_survey_response_prompt(
            persona, 
            question_data['question'],
            question_data['options']
        )
        try:
            response = _call_llm(prompt, model_name, max_tokens=50)
            # Validate response is in options
            if response not in question_data['options']:
                # Fall back to probabilistic if LLM response invalid
                response = probabilistic_respond(persona, question_id, random_state)
            return {'response': response, 'method': 'llm'}
        except Exception as e:
            # Fall back to probabilistic on error
            response = probabilistic_respond(persona, question_id, random_state)
            return {'response': response, 'method': 'probabilistic', 'error': str(e)}
    else:
        # Use probabilistic model
        response = probabilistic_respond(persona, question_id, random_state)
        return {'response': response, 'method': 'probabilistic'}


def generate_survey_responses(personas: List[Persona],
                             question_ids: Optional[List[str]] = None,
                             use_llm: bool = False,
                             model_name: Optional[str] = None,
                             random_seed: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate survey responses for all personas.
    
    Args:
        personas: List of Persona objects
        question_ids: List of question IDs (default: all questions)
        use_llm: Whether to use LLM for responses
        model_name: LLM model name
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping question_id to list of responses
    """
    if question_ids is None:
        question_ids = list(SURVEY_QUESTIONS.keys())
    
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()
    
    results = {}
    
    for question_id in question_ids:
        responses = []
        for persona in personas:
            response = respond_to_survey(
                persona, question_id, use_llm, model_name, rng
            )
            responses.append(response)
        results[question_id] = responses
    
    return results

