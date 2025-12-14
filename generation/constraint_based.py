"""
Method 2: Constraint-Based Generation

Generate personas using probabilistic constraints that reflect realistic relationships between variables.

Method for persona development: 

Key Constraints:
1. Gender → LGBTQ+ Identification (Females: 30%, Males: 21%)
2. Income Percentile → GPA (Top 30%: +0.3, Bottom 30%: -0.15)
3. Income Percentile → First-Gen Status (Top 30%: 30%, Bottom 30%: 60%)
4. Race → GPA Adjustment (Asian: +0.25, Black: -0.30)
5. Gender → Mental Health (Females worse: 43% poor vs 28% poor)
6. LGBTQ+ Status → Mental Health (worse outcomes)
7. Race → Social Media Intensity (Black/Hispanic higher)
8. GPA times Income → College Intentions
"""

import numpy as np
from typing import List, Optional
from persona.persona import Persona
from data.research_data import (
    GENDER_DISTRIBUTION,
    RACE_ETHNICITY_DISTRIBUTION,
    LGBTQ_BY_GENDER,
    FIRST_GEN_BY_INCOME_PERCENTILE,
    MENTAL_HEALTH_BY_GENDER,
    MENTAL_HEALTH_LGBTQ_ADJUSTMENTS,
    SOCIAL_MEDIA_RACE_FACTORS,
    COLLEGE_INTENTION_BY_GPA_INCOME,
    get_income_percentile,
    get_gpa_category,
    get_income_category
)
from data.distributions import (
    sample_categorical,
    sample_lognormal_income,
    sample_gpa,
    calculate_gpa_with_adjustments
)


def _sample_lgbtq_given_gender(gender: str, rng: np.random.Generator) -> str:
    """Sample LGBTQ+ status conditional on gender."""
    prob_lgbtq = LGBTQ_BY_GENDER[gender]
    if rng.random() < prob_lgbtq:
        return 'LGBTQ+'
    else:
        return 'Heterosexual'


def _sample_first_gen_given_income_percentile(income_percentile: str, 
                                             rng: np.random.Generator) -> bool:
    """Sample first-gen status conditional on income percentile."""
    prob_first_gen = FIRST_GEN_BY_INCOME_PERCENTILE[income_percentile]
    return rng.random() < prob_first_gen


def _sample_mental_health_given_gender_lgbtq(gender: str, lgbtq: str,
                                             rng: np.random.Generator) -> str:
    """Sample mental health conditional on gender and LGBTQ+ status."""
    from data.research_data import MENTAL_HEALTH_DISTRIBUTION
    
    # Start with gender-conditional distribution
    base_dist = MENTAL_HEALTH_BY_GENDER[gender].copy()
    
    # Apply LGBTQ+ adjustments
    if lgbtq == 'LGBTQ+':
        base_dist['Good'] = max(0, base_dist['Good'] + MENTAL_HEALTH_LGBTQ_ADJUSTMENTS['Good'])
        base_dist['Poor'] = max(0, base_dist['Poor'] + MENTAL_HEALTH_LGBTQ_ADJUSTMENTS['Poor'])
        # Normalize
        total = sum(base_dist.values())
        base_dist = {k: v / total for k, v in base_dist.items()}
    
    return sample_categorical(base_dist, n=1, random_state=rng)[0]


def _sample_social_media_given_race(race: str, rng: np.random.Generator) -> str:
    """Sample social media intensity with race-based adjustments."""
    from data.research_data import SOCIAL_MEDIA_INTENSITY_DISTRIBUTION
    
    base_dist = SOCIAL_MEDIA_INTENSITY_DISTRIBUTION.copy()
    factor = SOCIAL_MEDIA_RACE_FACTORS.get(race, 1.0)
    
    # Adjust probabilities (higher factor = more likely to be high intensity)
    if factor > 1.0:
        # Increase high intensity probabilities
        base_dist['High'] *= factor
        base_dist['Almost Constant'] *= factor
        # Decrease low intensity
        base_dist['Low'] /= factor
    elif factor < 1.0:
        # Decrease high intensity
        base_dist['High'] *= factor
        base_dist['Almost Constant'] *= factor
        # Increase low intensity
        base_dist['Low'] /= factor
    
    # Normalize
    total = sum(base_dist.values())
    base_dist = {k: v / total for k, v in base_dist.items()}
    
    return sample_categorical(base_dist, n=1, random_state=rng)[0]


def _sample_college_intention_given_gpa_income(gpa: float, income: float,
                                               first_gen: bool,
                                               rng: np.random.Generator) -> str:
    """Sample college intention based on GPA × Income constraints."""
    gpa_cat = get_gpa_category(gpa)
    income_cat = get_income_category(income)
    
    # Determine which distribution to use
    if gpa_cat == 'High' and income_cat == 'High':
        dist_key = 'High GPA + High Income'
    elif gpa_cat == 'Low':
        dist_key = 'Low GPA'
    else:
        dist_key = 'Moderate GPA + Moderate Income'
    
    base_dist = COLLEGE_INTENTION_BY_GPA_INCOME[dist_key].copy()
    
    # Adjust for first-gen (slightly lower 4-year college probability)
    if first_gen:
        base_dist['4-year college'] *= 0.9
        base_dist['2-year college'] *= 1.1
        # Normalize
        total = sum(base_dist.values())
        base_dist = {k: v / total for k, v in base_dist.items()}
    
    return sample_categorical(base_dist, n=1, random_state=rng)[0]


def _sample_sports_given_lgbtq(lgbtq: str, gender: str, rng: np.random.Generator) -> bool:
    """Sample sports participation (lower for LGBTQ+ students)."""
    from data.research_data import SPORTS_BY_GENDER
    
    base_prob = SPORTS_BY_GENDER[gender]
    
    # Reduce probability for LGBTQ+ students
    if lgbtq == 'LGBTQ+':
        from data.research_data import SPORTS_LGBTQ_REDUCTION
        base_prob *= (1 - SPORTS_LGBTQ_REDUCTION)
    
    return rng.random() < base_prob


def generate_constraint_based_personas(n: int = 100,
                                      age_range: tuple = (16, 18),
                                      random_seed: Optional[int] = None) -> List[Persona]:
    """
    Generate personas using constraint-based method.
    
    This method applies probabilistic constraints to capture realistic
    correlations between variables.
    
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
        
        # 2. Sample gender and race (independent)
        gender = sample_categorical(GENDER_DISTRIBUTION, n=1, random_state=rng)[0]
        race = sample_categorical(RACE_ETHNICITY_DISTRIBUTION, n=1, random_state=rng)[0]
        
        # 3. Sample income from lognormal
        income = sample_lognormal_income(n=1, random_state=rng)[0]
        income_percentile = get_income_percentile(income)
        
        # 4. Sample sexual orientation using gender-conditional probabilities
        sexual_orientation = _sample_lgbtq_given_gender(gender, rng)
        
        # 5. Sample first-gen college using income-conditional probabilities
        first_gen = _sample_first_gen_given_income_percentile(income_percentile, rng)
        
        # 6. Calculate GPA using gender, income_percentile, and race adjustments
        base_gpa = sample_gpa(gender, n=1, random_state=rng)[0]
        gpa = calculate_gpa_with_adjustments(
            base_gpa, race, income_percentile, random_state=rng
        )
        
        # 7. Sample sports participation (LGBTQ+ lower)
        sports = _sample_sports_given_lgbtq(sexual_orientation, gender, rng)
        
        # 8. Sample mental health using gender + LGBTQ+ adjustments
        mental_health = _sample_mental_health_given_gender_lgbtq(
            gender, sexual_orientation, rng
        )
        
        # 9. Sample social media intensity using race + gender factors
        social_media = _sample_social_media_given_race(race, rng)
        
        # 10. Sample college intention using gpa × income constraints
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
            first_gen_college=first_gen,
            gpa=float(gpa),
            sports_participation=sports,
            mental_health=mental_health,
            social_media_intensity=social_media,
            college_intention=college_intention
        )
        
        personas.append(persona)
    
    return personas

