"""
Validation functions for persona generation.

Validates that generated personas match expected distributions and correlations.
"""

import numpy as np
from typing import List, Dict, Tuple
from persona.persona import Persona
from data.research_data import (
    GENDER_DISTRIBUTION,
    RACE_ETHNICITY_DISTRIBUTION,
    SEXUAL_ORIENTATION_DISTRIBUTION,
    FIRST_GEN_COLLEGE_DISTRIBUTION,
    GPA_DISTRIBUTION,
    EXPECTED_CORRELATIONS,
    get_income_percentile
)


def validate_demographics(personas: List[Persona]) -> Dict[str, Dict[str, float]]:
    """
    Validate demographic distributions against expected values.
    
    Args:
        personas: List of Persona objects
        
    Returns:
        Dictionary with validation results for each demographic
    """
    n = len(personas)
    results = {}
    
    # Gender distribution
    gender_counts = {}
    for persona in personas:
        gender_counts[persona.gender] = gender_counts.get(persona.gender, 0) + 1
    
    results['gender'] = {
        'observed': {k: v / n for k, v in gender_counts.items()},
        'expected': GENDER_DISTRIBUTION,
        'difference': {k: (gender_counts.get(k, 0) / n) - GENDER_DISTRIBUTION.get(k, 0)
                     for k in set(list(gender_counts.keys()) + list(GENDER_DISTRIBUTION.keys()))}
    }
    
    # Race distribution
    race_counts = {}
    for persona in personas:
        race_counts[persona.race] = race_counts.get(persona.race, 0) + 1
    
    results['race'] = {
        'observed': {k: v / n for k, v in race_counts.items()},
        'expected': RACE_ETHNICITY_DISTRIBUTION,
        'difference': {k: (race_counts.get(k, 0) / n) - RACE_ETHNICITY_DISTRIBUTION.get(k, 0)
                     for k in set(list(race_counts.keys()) + list(RACE_ETHNICITY_DISTRIBUTION.keys()))}
    }
    
    # LGBTQ+ distribution
    lgbtq_count = sum(1 for p in personas if p.sexual_orientation == 'LGBTQ+')
    results['lgbtq'] = {
        'observed': lgbtq_count / n,
        'expected': SEXUAL_ORIENTATION_DISTRIBUTION['LGBTQ+'],
        'difference': (lgbtq_count / n) - SEXUAL_ORIENTATION_DISTRIBUTION['LGBTQ+']
    }
    
    # First-gen college
    first_gen_count = sum(1 for p in personas if p.first_gen_college)
    results['first_gen'] = {
        'observed': first_gen_count / n,
        'expected': FIRST_GEN_COLLEGE_DISTRIBUTION['Yes'],
        'difference': (first_gen_count / n) - FIRST_GEN_COLLEGE_DISTRIBUTION['Yes']
    }
    
    # Average GPA
    avg_gpa = np.mean([p.gpa for p in personas])
    results['gpa'] = {
        'observed': avg_gpa,
        'expected': GPA_DISTRIBUTION['overall_mean'],
        'difference': avg_gpa - GPA_DISTRIBUTION['overall_mean'],
        'is_percentage': False  # GPA is not a percentage
    }
    
    return results


def validate_correlations(personas: List[Persona]) -> Dict[str, Dict[str, float]]:
    """
    Validate correlations between variables.
    
    Args:
        personas: List of Persona objects
        
    Returns:
        Dictionary with correlation validation results
    """
    results = {}
    
    # Income-GPA correlation
    incomes = np.array([p.family_income for p in personas])
    gpas = np.array([p.gpa for p in personas])
    income_gpa_corr = np.corrcoef(incomes, gpas)[0, 1]
    
    expected_range = EXPECTED_CORRELATIONS['income_gpa']
    results['income_gpa'] = {
        'observed': income_gpa_corr,
        'expected_range': expected_range,
        'within_range': expected_range[0] <= income_gpa_corr <= expected_range[1]
    }
    
    # Income-First-Gen correlation (should be negative)
    first_gen_binary = np.array([1 if p.first_gen_college else 0 for p in personas])
    income_first_gen_corr = np.corrcoef(incomes, first_gen_binary)[0, 1]
    
    expected_range = EXPECTED_CORRELATIONS['income_first_gen']
    results['income_first_gen'] = {
        'observed': income_first_gen_corr,
        'expected_range': expected_range,
        'within_range': expected_range[0] <= income_first_gen_corr <= expected_range[1]
    }
    
    # GPA-College Intention correlation
    college_intention_binary = np.array([
        1 if p.college_intention in ['4-year college', '2-year college'] else 0
        for p in personas
    ])
    gpa_college_corr = np.corrcoef(gpas, college_intention_binary)[0, 1]
    
    expected_range = EXPECTED_CORRELATIONS['gpa_college_intention']
    results['gpa_college_intention'] = {
        'observed': gpa_college_corr,
        'expected_range': expected_range,
        'within_range': expected_range[0] <= gpa_college_corr <= expected_range[1]
    }
    
    return results


def validate_distributions(personas: List[Persona]) -> Dict[str, any]:
    """
    Comprehensive validation of all distributions and correlations.
    
    Args:
        personas: List of Persona objects
        
    Returns:
        Dictionary with all validation results
    """
    return {
        'demographics': validate_demographics(personas),
        'correlations': validate_correlations(personas),
        'n_personas': len(personas)
    }


def print_validation_report(validation_results: Dict[str, any]):
    """
    Print a formatted validation report.
    
    Args:
        validation_results: Results from validate_distributions()
    """
    print("=" * 60)
    print("PERSONA GENERATION VALIDATION REPORT")
    print("=" * 60)
    print(f"\nTotal Personas: {validation_results['n_personas']}\n")
    
    # Demographics
    print("DEMOGRAPHIC DISTRIBUTIONS")
    print("-" * 60)
    demos = validation_results['demographics']
    
    for demo_name, demo_data in demos.items():
        if isinstance(demo_data['observed'], dict):
            print(f"\n{demo_name.upper()}:")
            for key in demo_data['expected'].keys():
                obs = demo_data['observed'].get(key, 0)
                exp = demo_data['expected'][key]
                diff = demo_data['difference'].get(key, 0)
                status = "✓" if abs(diff) < 0.05 else "✗"
                print(f"  {key:20s} Observed: {obs:6.2%}  Expected: {exp:6.2%}  Diff: {diff:+7.2%} {status}")
        else:
            obs = demo_data['observed']
            exp = demo_data['expected']
            diff = demo_data['difference']
            is_percentage = demo_data.get('is_percentage', True)
            status = "✓" if abs(diff) < (0.05 if is_percentage else 0.1) else "✗"
            if is_percentage:
                print(f"{demo_name.upper():20s} Observed: {obs:6.2%}  Expected: {exp:6.2%}  Diff: {diff:+7.2%} {status}")
            else:
                print(f"{demo_name.upper():20s} Observed: {obs:6.2f}  Expected: {exp:6.2f}  Diff: {diff:+7.2f} {status}")
    
    # Correlations
    print("\n\nCORRELATIONS")
    print("-" * 60)
    corrs = validation_results['correlations']
    
    for corr_name, corr_data in corrs.items():
        obs = corr_data['observed']
        exp_range = corr_data['expected_range']
        status = "✓" if corr_data['within_range'] else "✗"
        print(f"{corr_name:25s} Observed: {obs:+.3f}  Expected: [{exp_range[0]:+.2f}, {exp_range[1]:+.2f}] {status}")
    
    print("\n" + "=" * 60)

