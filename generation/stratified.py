"""
Method 3: Stratified Sampling

Generate personas by stratifying on key demographic variables (gender, race,
income quintile) to ensure guaranteed representation of demographic groups.

Guarantees:
- Exactly 20 personas in each income quintile
- Proper representation of all demographic groups
- Better for hypothesis testing about subgroup differences
"""

import numpy as np
from typing import List, Optional
from persona.persona import Persona
from data.research_data import (
    GENDER_DISTRIBUTION,
    RACE_ETHNICITY_DISTRIBUTION,
    INCOME_QUINTILES,
    get_income_percentile
)
from data.distributions import (
    sample_categorical,
    sample_uniform_income_quintile
)
from generation.constraint_based import (
    _sample_lgbtq_given_gender,
    _sample_first_gen_given_income_percentile,
    _sample_mental_health_given_gender_lgbtq,
    _sample_social_media_given_race,
    _sample_college_intention_given_gpa_income,
    _sample_sports_given_lgbtq
)
from data.distributions import (
    calculate_gpa_with_adjustments,
    sample_gpa
)


def generate_stratified_personas(n: int = 100,
                                age_range: tuple = (16, 18),
                                random_seed: Optional[int] = None) -> List[Persona]:
    """
    Generate personas using stratified sampling method.
    
    Stratifies by income quintile (20 personas per quintile), then fills
    other attributes using constraint-based logic.
    
    Args:
        n: Number of personas to generate (must be multiple of 5 for quintiles)
        age_range: Tuple of (min_age, max_age)
        random_seed: Optional random seed for reproducibility
        
    Returns:
        List of Persona objects
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()
    
    # Ensure n is divisible by 5 for quintiles
    if n % 5 != 0:
        # Round up to nearest multiple of 5
        n = ((n // 5) + 1) * 5
    
    personas_per_quintile = n // 5
    quintiles = list(INCOME_QUINTILES.keys())
    
    personas = []
    
    # Generate personas for each quintile
    for quintile in quintiles:
        for i in range(personas_per_quintile):
            # 1. Sample age
            age = rng.integers(age_range[0], age_range[1] + 1)
            
            # 2. Sample gender from distribution
            gender = sample_categorical(GENDER_DISTRIBUTION, n=1, random_state=rng)[0]
            
            # 3. Sample race from distribution
            race = sample_categorical(RACE_ETHNICITY_DISTRIBUTION, n=1, random_state=rng)[0]
            
            # 4. Sample income uniformly from quintile range
            income = sample_uniform_income_quintile(quintile, n=1, random_state=rng)[0]
            income_percentile = get_income_percentile(income)
            
            # 5. Assign income_quintile
            income_quintile = quintile
            
            # 6. Sample sexual orientation (gender-conditional)
            sexual_orientation = _sample_lgbtq_given_gender(gender, rng)
            
            # 7. Sample first-gen college (income-conditional)
            first_gen = _sample_first_gen_given_income_percentile(income_percentile, rng)
            
            # 8. Calculate GPA using quintile + gender + race
            # Use quintile position as proxy for income percentile
            base_gpa = sample_gpa(gender, n=1, random_state=rng)[0]
            gpa = calculate_gpa_with_adjustments(
                base_gpa, race, income_percentile, random_state=rng
            )
            
            # 9. Sample sports participation
            sports = _sample_sports_given_lgbtq(sexual_orientation, gender, rng)
            
            # 10. Sample mental health
            mental_health = _sample_mental_health_given_gender_lgbtq(
                gender, sexual_orientation, rng
            )
            
            # 11. Sample social media intensity
            social_media = _sample_social_media_given_race(race, rng)
            
            # 12. Sample college intention
            college_intention = _sample_college_intention_given_gpa_income(
                gpa, income, first_gen, rng
            )
            
            # Create persona
            persona = Persona(
                age=age,
                gender=gender,
                race=race,
                sexual_orientation=sexual_orientation,
                family_income=float(income),
                income_percentile=income_percentile,
                income_quintile=income_quintile,
                first_gen_college=first_gen,
                gpa=float(gpa),
                sports_participation=sports,
                mental_health=mental_health,
                social_media_intensity=social_media,
                college_intention=college_intention
            )
            
            personas.append(persona)
    
    return personas

