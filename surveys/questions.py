"""
Survey question definitions.

Contains the survey questions used in the persona evaluation.
"""

from typing import Dict, List

# Survey questions with response options
SURVEY_QUESTIONS = {
    'college_importance': {
        'question': 'How important is college to your future?',
        'options': [
            'Very Important',
            'Somewhat Important',
            'Not Very Important',
            'Not Important At All'
        ]
    },
    'social_media_stress': {
        'question': 'Do you feel social media increases stress?',
        'options': [
            'Strongly Agree',
            'Somewhat Agree',
            'Neither Agree nor Disagree',
            'Somewhat Disagree',
            'Strongly Disagree'
        ]
    },
    'school_start_times': {
        'question': 'Would you support later school start times?',
        'options': [
            'Strongly Support',
            'Support',
            'Neutral',
            'Oppose',
            'Strongly Oppose'
        ]
    }
}


def get_survey_question(question_id: str) -> Dict[str, any]:
    """
    Get survey question by ID.
    
    Args:
        question_id: Question identifier
        
    Returns:
        Dictionary with 'question' and 'options' keys
    """
    if question_id not in SURVEY_QUESTIONS:
        raise ValueError(f"Unknown question ID: {question_id}")
    return SURVEY_QUESTIONS[question_id]


def list_survey_questions() -> List[str]:
    """Return list of all survey question IDs."""
    return list(SURVEY_QUESTIONS.keys())

