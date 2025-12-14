"""
Evaluation and validation framework.

Contains functions for validating persona distributions and comparing
generation methods.
"""

from .validation import (
    validate_demographics,
    validate_correlations,
    validate_distributions,
    print_validation_report
)
from .metrics import (
    calculate_distribution_metrics,
    compare_methods,
    analyze_survey_responses
)

__all__ = [
    'validate_demographics',
    'validate_correlations',
    'validate_distributions',
    'print_validation_report',
    'calculate_distribution_metrics',
    'compare_methods',
    'analyze_survey_responses'
]

