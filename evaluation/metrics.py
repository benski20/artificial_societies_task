"""
Metrics and comparison functions for evaluating generation methods.
"""

import numpy as np
from typing import List, Dict, Any
from persona.persona import Persona
from collections import Counter


def calculate_distribution_metrics(personas: List[Persona]) -> Dict[str, Any]:
    """
    Calculate various distribution metrics for personas.
    
    Args:
        personas: List of Persona objects
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {
        'n_personas': len(personas),
        'avg_gpa': np.mean([p.gpa for p in personas]),
        'median_income': np.median([p.family_income for p in personas]),
        'avg_age': np.mean([p.age for p in personas]),
        'gender_distribution': Counter([p.gender for p in personas]),
        'race_distribution': Counter([p.race for p in personas]),
        'lgbtq_percentage': sum(1 for p in personas if p.sexual_orientation == 'LGBTQ+') / len(personas),
        'first_gen_percentage': sum(1 for p in personas if p.first_gen_college) / len(personas),
        'sports_percentage': sum(1 for p in personas if p.sports_participation) / len(personas),
        'mental_health_distribution': Counter([p.mental_health for p in personas]),
        'social_media_distribution': Counter([p.social_media_intensity for p in personas]),
        'college_intention_distribution': Counter([p.college_intention for p in personas])
    }
    
    return metrics


def compare_methods(method_results: Dict[str, List[Persona]]) -> Dict[str, Any]:
    """
    Compare results from different generation methods.
    
    Args:
        method_results: Dictionary mapping method name to list of personas
        
    Returns:
        Comparison results
    """
    comparisons = {}
    
    for method_name, personas in method_results.items():
        comparisons[method_name] = calculate_distribution_metrics(personas)
    
    return comparisons


def analyze_survey_responses(survey_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, int]]:
    """
    Analyze survey response distributions.
    
    Args:
        survey_results: Dictionary mapping question_id to list of responses
        
    Returns:
        Dictionary with response counts for each question
    """
    analysis = {}
    
    for question_id, responses in survey_results.items():
        response_counts = Counter([r['response'] for r in responses])
        analysis[question_id] = dict(response_counts)
    
    return analysis


def print_comparison_report(comparison_results: Dict[str, Any]):
    """
    Print a formatted comparison report.
    
    Args:
        comparison_results: Results from compare_methods()
    """
    print("=" * 80)
    print("METHOD COMPARISON REPORT")
    print("=" * 80)
    
    methods = list(comparison_results.keys())
    
    # Key metrics to compare
    metrics_to_compare = [
        'avg_gpa',
        'median_income',
        'lgbtq_percentage',
        'first_gen_percentage',
        'sports_percentage'
    ]
    
    print("\nKEY METRICS")
    print("-" * 80)
    print(f"{'Metric':<25}", end="")
    for method in methods:
        print(f"{method:>20}", end="")
    print()
    print("-" * 80)
    
    for metric in metrics_to_compare:
        print(f"{metric:<25}", end="")
        for method in methods:
            value = comparison_results[method].get(metric, 'N/A')
            if isinstance(value, float):
                print(f"{value:>20.3f}", end="")
            else:
                print(f"{str(value):>20}", end="")
        print()
    
    print("\n" + "=" * 80)

