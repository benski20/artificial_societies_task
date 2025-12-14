"""
Persona generation methods.

Contains three different methodologies for generating synthetic personas.
"""

from .marginal import generate_marginal_personas
from .constraint_based import generate_constraint_based_personas
from .stratified import generate_stratified_personas

__all__ = [
    'generate_marginal_personas',
    'generate_constraint_based_personas',
    'generate_stratified_personas'
]

