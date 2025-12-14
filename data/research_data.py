"""
Research-based constants and distributions for U.S. High School Students (ages 16-18).

All data sourced from:
- CDC Adolescent Behaviors and Experiences Survey (ABES) 2021
- Pew Research Center Social Media & Technology surveys 2024-2025
- National Center for Education Statistics (NCES)
- U.S. Bureau of Labor Statistics (BLS)
"""

from typing import Dict, List, Tuple
import numpy as np

# ============================================================================
# DEMOGRAPHIC DISTRIBUTIONS
# ============================================================================

GENDER_DISTRIBUTION = {
    'Female': 0.51,
    'Male': 0.49
}

RACE_ETHNICITY_DISTRIBUTION = {
    'White': 0.39,
    'Hispanic': 0.28,
    'Black': 0.15,
    'Asian': 0.055,
    'Two or more races': 0.05,
    'Other': 0.009
}

SEXUAL_ORIENTATION_DISTRIBUTION = {
    'Heterosexual': 0.74,
    'LGBTQ+': 0.26
}

# Gender-conditional LGBTQ+ probabilities
LGBTQ_BY_GENDER = {
    'Female': 0.30,  # 30% of females identify as LGBTQ+
    'Male': 0.21     # 21% of males identify as LGBTQ+
}

# ============================================================================
# SOCIOECONOMIC STATUS
# ============================================================================

# Income distribution parameters (lognormal)
INCOME_DISTRIBUTION = {
    'mean': 10.8,      # log(median ~ $60k)
    'sigma': 0.8,
    'median': 60000,
    'percentiles': {
        20: 25000,
        80: 111000,
        99: 512000
    }
}

# Income quintile ranges for stratified sampling
INCOME_QUINTILES = {
    'Bottom 20%': (0, 30000),
    '20-40%': (30000, 45000),
    '40-60%': (45000, 60000),
    '60-80%': (60000, 85000),
    'Top 20%': (85000, None)  # None means no upper bound
}

FIRST_GEN_COLLEGE_DISTRIBUTION = {
    'Yes': 0.45,
    'No': 0.55
}

# Income-conditional first-gen probabilities
FIRST_GEN_BY_INCOME_PERCENTILE = {
    'Top 30%': 0.30,
    'Middle 40%': 0.45,
    'Bottom 30%': 0.60
}

# ============================================================================
# ACADEMIC PERFORMANCE
# ============================================================================

# GPA distributions by gender
GPA_DISTRIBUTION = {
    'overall_mean': 3.0,
    'female_mean': 3.1,
    'male_mean': 2.91,
    'std': 0.5,
    'min': 0.0,
    'max': 4.0
}

# GPA adjustments by race
GPA_RACE_ADJUSTMENTS = {
    'Asian': 0.25,
    'White': 0.10,
    'Hispanic': -0.15,
    'Black': -0.30,
    'Two or more races': 0.0,  # Default
    'Other': 0.0
}

# GPA adjustments by income percentile
GPA_INCOME_ADJUSTMENTS = {
    'Top 30%': 0.3,
    'Middle 40%': 0.15,
    'Bottom 30%': -0.15
}

# ============================================================================
# BEHAVIORAL DISTRIBUTIONS
# ============================================================================

SPORTS_PARTICIPATION_DISTRIBUTION = {
    'Yes': 0.54,
    'No': 0.46
}

SPORTS_BY_GENDER = {
    'Female': 0.43,  # 43% of females participate
    'Male': 0.57     # 57% of males participate
}

# Sports participation lower for LGBTQ+ students
SPORTS_LGBTQ_REDUCTION = 0.15  # 15% reduction in probability

MENTAL_HEALTH_DISTRIBUTION = {
    'Good': 0.40,
    'Fair': 0.23,
    'Poor': 0.37
}

# Gender-conditional mental health
MENTAL_HEALTH_BY_GENDER = {
    'Female': {'Good': 0.35, 'Fair': 0.22, 'Poor': 0.43},
    'Male': {'Good': 0.48, 'Fair': 0.24, 'Poor': 0.28}
}

# LGBTQ+ mental health adjustments
MENTAL_HEALTH_LGBTQ_ADJUSTMENTS = {
    'Good': -0.15,  # 15% reduction in good mental health
    'Poor': 0.10    # 10% increase in poor mental health
}

SOCIAL_MEDIA_INTENSITY_DISTRIBUTION = {
    'Low': 0.15,
    'Moderate': 0.20,
    'High': 0.32,
    'Almost Constant': 0.33
}

# Race-based social media intensity factors
SOCIAL_MEDIA_RACE_FACTORS = {
    'Black': 1.15,
    'Hispanic': 1.10,
    'Asian': 0.90,
    'White': 1.0,
    'Two or more races': 1.0,
    'Other': 1.0
}

# ============================================================================
# COLLEGE INTENTIONS
# ============================================================================

COLLEGE_INTENTION_DISTRIBUTION = {
    '4-year college': 0.35,
    '2-year college': 0.10,
    'Vocational/trade school': 0.15,
    'Work/other': 0.20,
    'Unsure': 0.20
}

# College intention by GPA × Income
COLLEGE_INTENTION_BY_GPA_INCOME = {
    'High GPA + High Income': {'4-year college': 0.60, '2-year college': 0.15, 'Other': 0.25},
    'Moderate GPA + Moderate Income': {'4-year college': 0.40, '2-year college': 0.20, 'Other': 0.40},
    'Low GPA': {'4-year college': 0.10, '2-year college': 0.10, 'Other': 0.80}
}

# ============================================================================
# CORRELATIONS (for validation)
# ============================================================================

EXPECTED_CORRELATIONS = {
    'income_gpa': (0.35, 0.45),  # Moderate positive
    'income_first_gen': (-0.60, -0.50),  # Strong negative
    'gpa_college_intention': (0.50, 0.70)  # Strong positive
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_income_percentile(income: float) -> str:
    """Determine income percentile category."""
    if income >= INCOME_DISTRIBUTION['percentiles'][80]:
        return 'Top 30%'
    elif income >= INCOME_DISTRIBUTION['percentiles'][20]:
        return 'Middle 40%'
    else:
        return 'Bottom 30%'

def get_gpa_category(gpa: float) -> str:
    """Categorize GPA for college intention logic."""
    if gpa >= 3.5:
        return 'High'
    elif gpa >= 2.5:
        return 'Moderate'
    else:
        return 'Low'

def get_income_category(income: float) -> str:
    """Categorize income for college intention logic."""
    if income >= 80000:
        return 'High'
    elif income >= 40000:
        return 'Moderate'
    else:
        return 'Low'

