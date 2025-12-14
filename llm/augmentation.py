"""
LLM augmentation functions for persona generation.

Uses LLMs (GPT-4o-mini) to augment statistically-generated personas with rich narratives, beliefs (probabilistic), and explanations.
"""

import os
from typing import List, Optional, Dict
from pathlib import Path
try:
    from dotenv import load_dotenv
    # Try loading from config/.env first, then root .env
    env_path = Path(__file__).parent.parent / 'config' / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    load_dotenv()  # Also try root .env
except ImportError:
    # dotenv is optional - only needed for LLM features
    pass

from persona.persona import Persona
from .prompts import (
    generate_persona_narrative_prompt,
    generate_belief_explanation_prompt,
    generate_embedding_text
)

# Try to import OpenAI and Anthropic
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False



def _get_llm_client(model_name: Optional[str] = None):
    """
    Get appropriate LLM based on model name or default.
    
    Args:
        model_name: Model name ('gpt-4o-mini')
        
    Returns:
        LLM client object
    """
    if model_name is None:
        model_name = os.getenv('DEFAULT_LLM_MODEL', 'gpt-4o-mini')
    
    if 'gpt' in model_name.lower() or 'openai' in model_name.lower():
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(api_key=api_key), 'openai'
    
    else:
        # Default to OpenAI
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(api_key=api_key), 'openai'


def _call_llm(prompt: str, model_name: Optional[str] = None, 
              max_tokens: int = 500) -> str:
    """
    Call LLM with prompt and return response.
    
    Args:
        prompt: Prompt text
        model_name: Model name (optional)
        max_tokens: Maximum tokens in response
        
    Returns:
        LLM response text
    """
    client, provider = _get_llm_client(model_name)
    
    if provider == 'openai':
        if model_name is None:
            model_name = os.getenv('DEFAULT_LLM_MODEL', 'gpt-4o-mini')
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates realistic, authentic personas for research purposes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    
    elif provider == 'anthropic':
        if model_name is None:
            model_name = os.getenv('DEFAULT_LLM_MODEL', 'claude-3-haiku-20240307')
        
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def augment_single_persona(persona: Persona, 
                          model_name: Optional[str] = None,
                          topics: Optional[List[str]] = None) -> Persona:
    """
    Augment a single persona with LLM-generated narrative and beliefs.
    
    Args:
        persona: Persona object to augment
        model_name: LLM model name (optional)
        topics: List of topics for belief generation (default: common topics)
        
    Returns:
        Augmented Persona object (modified in place)
    """
    if topics is None:
        topics = [
            'college importance and future plans',
            'social media impact on daily life',
            'school start times and sleep',
            'mental health and stress',
            'academic pressure and expectations'
        ]
    
    # Generate narrative
    narrative_prompt = generate_persona_narrative_prompt(persona)
    persona.narrative = _call_llm(narrative_prompt, model_name, max_tokens=400)
    
    # Generate beliefs for each topic
    persona.beliefs = {}
    for topic in topics:
        belief_prompt = generate_belief_explanation_prompt(persona, topic)
        belief = _call_llm(belief_prompt, model_name, max_tokens=150)
        persona.beliefs[topic] = belief
    
    return persona


def augment_personas_with_llm(personas: List[Persona],
                              model_name: Optional[str] = None,
                              topics: Optional[List[str]] = None,
                              batch_size: int = 10,
                              verbose: bool = True) -> List[Persona]:
    """
    Augment a list of personas with LLM-generated narratives and beliefs.
    
    Args:
        personas: List of Persona objects to augment
        model_name: LLM model name (optional)
        topics: List of topics for belief generation
        batch_size: Number of personas to process before progress update
        verbose: Whether to print progress
        
    Returns:
        List of augmented Persona objects
    """
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(personas, desc="Augmenting personas with LLM")
    else:
        iterator = personas
    
    augmented = []
    for i, persona in enumerate(iterator):
        try:
            augmented_persona = augment_single_persona(persona, model_name, topics)
            augmented.append(augmented_persona)
        except Exception as e:
            if verbose:
                print(f"Error augmenting persona {i}: {e}")
            # Keep original persona if augmentation fails
            augmented.append(persona)
    
    return augmented

