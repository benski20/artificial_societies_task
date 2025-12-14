"""
Survey system for persona responses.

Contains survey question definitions and response generation logic.
"""

from .questions import SURVEY_QUESTIONS, get_survey_question
from .responses import (
    respond_to_survey,
    generate_survey_responses,
    probabilistic_respond
)

__all__ = [
    'SURVEY_QUESTIONS',
    'get_survey_question',
    'respond_to_survey',
    'generate_survey_responses',
    'probabilistic_respond'
]

