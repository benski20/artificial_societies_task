"""
Method 1: Marginal Distributions

Generate personas by independently sampling from each demographic variable's marginal distribution without considering correlations.

Advantages:
- Simplicity: Easy to implement and understand
- Accuracy for Marginals: Perfectly captures individual variable distributions
- Speed: Fast generation

Disadvantages:
- No Correlations: Misses realistic relationships
- Unrealistic Combinations: Can generate implausible personas
"""

import numpy as np
from typing import List, Optional
from persona.persona import Persona
from data.research_data import (
    GENDER_DISTRIBUTION,
    RACE_ETHNICITY_DISTRIBUTION,
    SEXUAL_ORIENTATION_DISTRIBUTION,
    FIRST_GEN_COLLEGE_DISTRIBUTION,
    SPORTS_PARTICIPATION_DISTRIBUTION,
    MENTAL_HEALTH_DISTRIBUTION,
    SOCIAL_MEDIA_INTENSITY_DISTRIBUTION,
    COLLEGE_INTENTION_DISTRIBUTION
)
from data.distributions import (
    sample_categorical,
    sample_lognormal_income,
    sample_gpa
)


def generate_marginal_personas(n: int = 100, 
                               age_range: tuple = (16, 18),
                               random_seed: Optional[int] = None) -> List[Persona]:
    """
    Generate personas using marginal distributions method.
    
    Each variable is sampled independently from its marginal distribution.
    
    Args:
        n: Number of personas to generate
        age_range: Tuple of (min_age, max_age)
        random_seed: Optional random seed for reproducibility
        
    Returns:
        List of Persona objects
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()
    
    personas = []
    
    for i in range(n):
        # 1. Sample age
        age = rng.integers(age_range[0], age_range[1] + 1)
        
        # 2. Sample gender
        gender = sample_categorical(GENDER_DISTRIBUTION, n=1, random_state=rng)[0]
        
        # 3. Sample race/ethnicity
        race = sample_categorical(RACE_ETHNICITY_DISTRIBUTION, n=1, random_state=rng)[0]
        
        # 4. Sample sexual orientation (independent of gender)
        sexual_orientation = sample_categorical(
            SEXUAL_ORIENTATION_DISTRIBUTION, n=1, random_state=rng
        )[0]
        
        # 5. Sample family income
        income = sample_lognormal_income(n=1, random_state=rng)[0]
        
        # 6. Sample first-gen college status (independent of income)
        first_gen = sample_categorical(
            FIRST_GEN_COLLEGE_DISTRIBUTION, n=1, random_state=rng
        )[0] == 'Yes'
        
        # 7. Sample GPA (gender-dependent but no other correlations)
        gpa = sample_gpa(gender, n=1, random_state=rng)[0]
        
        # 8. Sample sports participation
        sports = sample_categorical(
            SPORTS_PARTICIPATION_DISTRIBUTION, n=1, random_state=rng
        )[0] == 'Yes'
        
        # 9. Sample mental health
        mental_health = sample_categorical(
            MENTAL_HEALTH_DISTRIBUTION, n=1, random_state=rng
        )[0]
        
        # 10. Sample social media intensity
        social_media = sample_categorical(
            SOCIAL_MEDIA_INTENSITY_DISTRIBUTION, n=1, random_state=rng
        )[0]
        
        # 11. Sample college intention
        college_intention = sample_categorical(
            COLLEGE_INTENTION_DISTRIBUTION, n=1, random_state=rng
        )[0]
        
        # Create persona
        persona = Persona(
            age=age,
            gender=gender,
            race=race,
            sexual_orientation=sexual_orientation,
            family_income=float(income),
            first_gen_college=first_gen,
            gpa=float(gpa),
            sports_participation=sports,
            mental_health=mental_health,
            social_media_intensity=social_media,
            college_intention=college_intention
        )
        
        personas.append(persona)
    
    return personas

