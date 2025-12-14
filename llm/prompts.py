"""
Prompt templates for LLM augmentation.

Contains prompt templates for generating persona narratives, beliefs, and survey responses.
"""

from typing import Dict, Any
from persona.persona import Persona


def generate_persona_narrative_prompt(persona: Persona) -> str:
    """
    Generate prompt for creating persona narrative/background story.
    
    Args:
        persona: Persona object with structured attributes
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Create a realistic 2-3 paragraph background story for a U.S. high school student.

Demographics:
- Age: {persona.age}
- Gender: {persona.gender}
- Race/Ethnicity: {persona.race}
- Sexual Orientation: {persona.sexual_orientation}
- Family Income: ${persona.family_income:,.0f}
- First-Generation College Student: {'Yes' if persona.first_gen_college else 'No'}

Academic:
- GPA: {persona.gpa:.2f}
- Sports Participation: {'Yes' if persona.sports_participation else 'No'}

Personal:
- Mental Health: {persona.mental_health}
- Social Media Intensity: {persona.social_media_intensity}
- College Plan: {persona.college_intention}

Generate a realistic background story that explains how these attributes connect naturally. 
Include details about:
- Family background and dynamics
- School life and academic experiences
- Interests, hobbies, and social connections
- Aspirations and concerns about the future

Be authentic and avoid stereotypes. Write in third person. The story should feel like a real person's background, not a statistical profile."""
    
    return prompt


def generate_belief_explanation_prompt(persona: Persona, topic: str) -> str:
    """
    Generate prompt for explaining persona's beliefs on a specific topic.
    
    Args:
        persona: Persona object
        topic: Topic to explain beliefs about (e.g., "college importance", "social media impact")
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Persona Background:
{persona.narrative if persona.narrative else persona.get_summary()}

Structured Attributes:
- GPA: {persona.gpa:.2f}
- Family Income: ${persona.family_income:,.0f}
- First-Gen College: {'Yes' if persona.first_gen_college else 'No'}
- Mental Health: {persona.mental_health}
- College Plan: {persona.college_intention}

Topic: {topic}

Explain this persona's perspective and beliefs about {topic} in 2-3 sentences. 
Connect their views to their background, socioeconomic context, and personal 
circumstances. Be specific and authentic."""
    
    return prompt


def generate_survey_response_prompt(persona: Persona, question: str, 
                                   response_options: list) -> str:
    """
    Generate prompt for survey response generation.
    
    Args:
        persona: Persona object
        question: Survey question text
        response_options: List of possible response options
        
    Returns:
        Formatted prompt string
    """
    options_str = "\n".join([f"- {opt}" for opt in response_options])
    
    prompt = f"""Persona Background:
{persona.narrative if persona.narrative else persona.get_summary()}

Structured Attributes:
- GPA: {persona.gpa:.2f}
- Family Income: ${persona.family_income:,.0f}
- First-Gen College: {'Yes' if persona.first_gen_college else 'No'}
- Mental Health: {persona.mental_health}
- College Plan: {persona.college_intention}

Survey Question: "{question}"

Based on this persona's background and attributes, provide a realistic single-select response. Consider their lived experience, socioeconomic context, and personal circumstances.

Response Options:
{options_str}

Respond with ONLY the exact text of one of the response options above, nothing else."""
    
    return prompt


def generate_embedding_text(persona: Persona) -> str:
    """
    Generate text for embedding computation from persona.
    
    Args:
        persona: Persona object
        
    Returns:
        Combined text for embedding
    """
    parts = []
    
    if persona.narrative:
        parts.append(persona.narrative)
    
    if persona.beliefs:
        parts.append(" ".join(persona.beliefs.values()))
    
    # Add structured summary as fallback
    if not parts:
        parts.append(persona.get_summary())
    
    return " ".join(parts)

