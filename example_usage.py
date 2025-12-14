"""
Example usage script demonstrating how to use the persona generation system.

This script shows basic usage patterns without command-line arguments.
"""

from generation.constraint_based import generate_constraint_based_personas
from llm.augmentation import augment_single_persona
from surveys.responses import respond_to_survey, generate_survey_responses
from evaluation.validation import validate_distributions, print_validation_report


def example_basic_generation():
    """Example: Basic persona generation without LLM."""
    print("=" * 60)
    print("Example 1: Basic Persona Generation")
    print("=" * 60)
    
    # Generate 10 personas using constraint-based method
    personas = generate_constraint_based_personas(n=10, random_seed=42)
    
    print(f"\nGenerated {len(personas)} personas:")
    for i, persona in enumerate(personas[:3], 1):  # Show first 3
        print(f"\nPersona {i}:")
        print(f"  Age: {persona.age}, Gender: {persona.gender}, Race: {persona.race}")
        print(f"  Income: ${persona.family_income:,.0f}")
        print(f"  GPA: {persona.gpa:.2f}")
        print(f"  First-Gen College: {persona.first_gen_college}")
        print(f"  College Plan: {persona.college_intention}")
    
    return personas


def example_llm_augmentation():
    """Example: LLM augmentation (requires API key)."""
    print("\n\n" + "=" * 60)
    print("Example 2: LLM Augmentation")
    print("=" * 60)
    
    # Generate a single persona
    personas = generate_constraint_based_personas(n=1, random_seed=42)
    persona = personas[0]
    
    print(f"\nOriginal persona:")
    print(f"  {persona.get_summary()}")
    
    # Augment with LLM (will fail if API key not set)
    try:
        augmented = augment_single_persona(persona, model_name='gpt-4o-mini')
        
        print(f"\nAugmented persona:")
        print(f"\nNarrative:")
        print(f"  {augmented.narrative[:200]}...")
        
        print(f"\nBeliefs:")
        for topic, belief in list(augmented.beliefs.items())[:2]:
            print(f"  {topic}: {belief[:100]}...")
    except Exception as e:
        print(f"\n⚠ LLM augmentation failed (API key may not be set): {e}")
        print("  Continuing with original persona...")
        augmented = persona
    
    return augmented


def example_survey_responses():
    """Example: Generating survey responses."""
    print("\n\n" + "=" * 60)
    print("Example 3: Survey Response Generation")
    print("=" * 60)
    
    # Generate personas
    personas = generate_constraint_based_personas(n=20, random_seed=42)
    
    # Generate survey responses for all personas
    survey_results = generate_survey_responses(
        personas,
        use_llm=False,  # Use probabilistic model
        random_seed=42
    )
    
    print("\nSurvey Response Summary:")
    from surveys.questions import SURVEY_QUESTIONS
    from collections import Counter
    
    for question_id, responses in survey_results.items():
        question_text = SURVEY_QUESTIONS[question_id]['question']
        response_counts = Counter([r['response'] for r in responses])
        
        print(f"\n{question_text}:")
        for response, count in response_counts.most_common():
            percentage = (count / len(responses)) * 100
            print(f"  {response}: {count} ({percentage:.1f}%)")
    
    return survey_results


def example_validation():
    """Example: Validating generated personas."""
    print("\n\n" + "=" * 60)
    print("Example 4: Validation")
    print("=" * 60)
    
    # Generate personas
    personas = generate_constraint_based_personas(n=100, random_seed=42)
    
    # Validate
    validation_results = validate_distributions(personas)
    print_validation_report(validation_results)
    
    return validation_results


def example_comparison():
    """Example: Comparing different generation methods."""
    print("\n\n" + "=" * 60)
    print("Example 5: Method Comparison")
    print("=" * 60)
    
    from generation.marginal import generate_marginal_personas
    from generation.stratified import generate_stratified_personas
    from evaluation.metrics import compare_methods, print_comparison_report
    
    # Generate personas using all three methods
    print("\nGenerating personas with all three methods...")
    marginal_personas = generate_marginal_personas(n=100, random_seed=42)
    constraint_personas = generate_constraint_based_personas(n=100, random_seed=42)
    stratified_personas = generate_stratified_personas(n=100, random_seed=42)
    
    # Compare
    method_results = {
        'marginal': marginal_personas,
        'constraint': constraint_personas,
        'stratified': stratified_personas
    }
    
    comparison_results = compare_methods(method_results)
    print_comparison_report(comparison_results)
    
    return comparison_results


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SYNTHETIC PERSONA GENERATION - EXAMPLE USAGE")
    print("=" * 60)
    
    # Run examples
    example_basic_generation()
    example_llm_augmentation()
    example_survey_responses()
    example_validation()
    example_comparison()
    
    print("\n\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

