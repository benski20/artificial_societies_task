"""
Distribution functions for sampling persona attributes.

Provides functions for generating random values from various distributions
used in persona generation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .research_data import (
    INCOME_DISTRIBUTION,
    GPA_DISTRIBUTION,
    INCOME_QUINTILES
)


def sample_categorical(distribution: Dict[str, float], n: int = 1, 
                      random_state: Optional[np.random.Generator] = None) -> List[str]:
    """
    Sample from a categorical distribution.
    
    Args:
        distribution: Dictionary mapping categories to probabilities
        n: Number of samples to generate
        random_state: Optional random number generator
        
    Returns:
        List of sampled categories
    """
    if random_state is None:
        random_state = np.random.default_rng()
    
    categories = list(distribution.keys())
    probabilities = list(distribution.values())
    
    # Normalize probabilities
    probabilities = np.array(probabilities)
    probabilities = probabilities / probabilities.sum()
    
    samples = random_state.choice(categories, size=n, p=probabilities)
    return samples.tolist() if n > 1 else [samples.item()]


def sample_lognormal_income(n: int = 1, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample family income from lognormal distribution.
    
    Args:
        n: Number of samples
        random_state: Optional random number generator
        
    Returns:
        Array of income values (in dollars)
    """
    if random_state is None:
        random_state = np.random.default_rng()
    
    mean = INCOME_DISTRIBUTION['mean']
    sigma = INCOME_DISTRIBUTION['sigma']
    
    # Sample from lognormal
    log_income = random_state.normal(mean, sigma, n)
    income = np.exp(log_income)
    
    return income


def sample_gpa(gender: str, n: int = 1, 
               random_state: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample GPA from normal distribution, adjusted by gender.
    
    Args:
        gender: 'Female' or 'Male'
        n: Number of samples
        random_state: Optional random number generator
        
    Returns:
        Array of GPA values (clipped to [0, 4.0])
    """
    if random_state is None:
        random_state = np.random.default_rng()
    
    # Get gender-specific mean
    if gender == 'Female':
        mean = GPA_DISTRIBUTION['female_mean']
    else:
        mean = GPA_DISTRIBUTION['male_mean']
    
    std = GPA_DISTRIBUTION['std']
    min_gpa = GPA_DISTRIBUTION['min']
    max_gpa = GPA_DISTRIBUTION['max']
    
    # Sample from normal distribution
    gpa = random_state.normal(mean, std, n)
    
    # Clip to valid range
    gpa = np.clip(gpa, min_gpa, max_gpa)
    
    return gpa


def sample_uniform_income_quintile(quintile: str, n: int = 1,
                                  random_state: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample income uniformly within a specific quintile range.
    
    Args:
        quintile: One of 'Bottom 20%', '20-40%', '40-60%', '60-80%', 'Top 20%'
        n: Number of samples
        random_state: Optional random number generator
        
    Returns:
        Array of income values
    """
    if random_state is None:
        random_state = np.random.default_rng()
    
    if quintile not in INCOME_QUINTILES:
        raise ValueError(f"Unknown quintile: {quintile}")
    
    min_income, max_income = INCOME_QUINTILES[quintile]
    
    if max_income is None:
        # For top quintile, use lognormal tail (above 85k)
        # Sample from lognormal and filter
        samples = []
        while len(samples) < n:
            candidate = sample_lognormal_income(1, random_state)[0]
            if candidate >= min_income:
                samples.append(candidate)
        return np.array(samples)
    else:
        # Uniform sampling within range
        return random_state.uniform(min_income, max_income, n)


def calculate_gpa_with_adjustments(base_gpa: float, race: str, income_percentile: str,
                                   random_state: Optional[np.random.Generator] = None) -> float:
    """
    Calculate GPA with race and income adjustments, then add noise.
    
    Args:
        base_gpa: Base GPA (from gender distribution)
        race: Race/ethnicity category
        income_percentile: Income percentile category
        random_state: Optional random number generator
        
    Returns:
        Adjusted GPA (clipped to [0, 4.0])
    """
    from .research_data import GPA_RACE_ADJUSTMENTS, GPA_INCOME_ADJUSTMENTS
    
    if random_state is None:
        random_state = np.random.default_rng()
    
    # Apply adjustments
    race_adj = GPA_RACE_ADJUSTMENTS.get(race, 0.0)
    income_adj = GPA_INCOME_ADJUSTMENTS.get(income_percentile, 0.0)
    
    adjusted_gpa = base_gpa + race_adj + income_adj
    
    # Add noise
    noise = random_state.normal(0, GPA_DISTRIBUTION['std'], 1)[0]
    final_gpa = adjusted_gpa + noise
    
    # Clip to valid range
    final_gpa = np.clip(final_gpa, GPA_DISTRIBUTION['min'], GPA_DISTRIBUTION['max'])
    
    return float(final_gpa)


def sample_conditional_categorical(base_distribution: Dict[str, float],
                                   adjustments: Dict[str, float],
                                   condition: str,
                                   random_state: Optional[np.random.Generator] = None) -> str:
    """
    Sample from categorical distribution with conditional adjustments.
    
    Args:
        base_distribution: Base probability distribution
        adjustments: Adjustments to apply based on condition
        condition: Condition key to look up adjustments
        random_state: Optional random number generator
        
    Returns:
        Sampled category
    """
    if random_state is None:
        random_state = np.random.default_rng()
    
    # Start with base distribution
    adjusted_dist = base_distribution.copy()
    
    # Apply adjustments if condition exists
    if condition in adjustments:
        adj = adjustments[condition]
        # Adjust probabilities (simple additive adjustment)
        # Normalize to ensure probabilities sum to 1
        for key in adjusted_dist:
            adjusted_dist[key] += adj.get(key, 0)
        
        # Ensure non-negative and normalize
        adjusted_dist = {k: max(0, v) for k, v in adjusted_dist.items()}
        total = sum(adjusted_dist.values())
        adjusted_dist = {k: v / total for k, v in adjusted_dist.items()}
    
    return sample_categorical(adjusted_dist, n=1, random_state=random_state)[0]

