"""
LLM augmentation layer for persona generation.

Provides functions to augment statistically-generated personas with rich narratives, beliefs, and explanations using LLMs.
"""

from .augmentation import augment_personas_with_llm, augment_single_persona
from .prompts import (
    generate_persona_narrative_prompt,
    generate_belief_explanation_prompt,
    generate_survey_response_prompt
)

__all__ = [
    'augment_personas_with_llm',
    'augment_single_persona',
    'generate_persona_narrative_prompt',
    'generate_belief_explanation_prompt',
    'generate_survey_response_prompt'
]

