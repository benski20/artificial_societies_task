"""
Main orchestration script for synthetic persona generation.

This script demonstrates the complete pipeline:
1. Generate personas using different methods
2. Augment with LLM narratives
3. Generate survey responses
4. Validate and compare results
"""

import argparse
import json
from pathlib import Path
from typing import List

from generation.marginal import generate_marginal_personas
from generation.constraint_based import generate_constraint_based_personas
from generation.stratified import generate_stratified_personas
from llm.augmentation import augment_personas_with_llm
from surveys.responses import generate_survey_responses
from evaluation.validation import validate_distributions, print_validation_report
from evaluation.metrics import compare_methods, analyze_survey_responses, print_comparison_report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic personas for U.S. high school students'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['marginal', 'constraint', 'stratified', 'all'],
        default='constraint',
        help='Generation method to use (default: constraint)'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=100,
        help='Number of personas to generate (default: 100)'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Augment personas with LLM narratives'
    )
    parser.add_argument(
        '--surveys',
        action='store_true',
        help='Generate survey responses'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate generated personas'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default=None,
        help='LLM model name (e.g., gpt-4o-mini)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("SYNTHETIC PERSONA GENERATION PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Method: {args.method}")
    print(f"  Number of personas: {args.n}")
    print(f"  Random seed: {args.seed}")
    print(f"  LLM augmentation: {args.augment}")
    print(f"  Generate surveys: {args.surveys}")
    print(f"  Validate: {args.validate}\n")
    
    # Generate personas
    print("Step 1: Generating personas...")
    print("-" * 80)
    
    method_results = {}
    
    if args.method == 'all':
        methods_to_run = ['marginal', 'constraint', 'stratified']
    else:
        methods_to_run = [args.method]
    
    for method_name in methods_to_run:
        print(f"\nGenerating {args.n} personas using {method_name} method...")
        
        if method_name == 'marginal':
            personas = generate_marginal_personas(n=args.n, random_seed=args.seed)
        elif method_name == 'constraint':
            personas = generate_constraint_based_personas(n=args.n, random_seed=args.seed)
        elif method_name == 'stratified':
            personas = generate_stratified_personas(n=args.n, random_seed=args.seed)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        method_results[method_name] = personas
        print(f"✓ Generated {len(personas)} personas")
    
    # LLM augmentation
    if args.augment:
        print("\n\nStep 2: Augmenting personas with LLM...")
        print("-" * 80)
        
        for method_name, personas in method_results.items():
            print(f"\nAugmenting {method_name} personas...")
            try:
                augmented = augment_personas_with_llm(
                    personas,
                    model_name=args.llm_model,
                    verbose=True
                )
                method_results[method_name] = augmented
                print(f"✓ Augmented {len(augmented)} personas")
            except Exception as e:
                print(f"✗ Error during augmentation: {e}")
                print("  Continuing without LLM augmentation...")
    
    # Survey responses
    survey_results_all = {}
    if args.surveys:
        print("\n\nStep 3: Generating survey responses...")
        print("-" * 80)
        
        for method_name, personas in method_results.items():
            print(f"\nGenerating survey responses for {method_name} method...")
            survey_results = generate_survey_responses(
                personas,
                use_llm=False,  # Use probabilistic model by default
                random_seed=args.seed
            )
            survey_results_all[method_name] = survey_results
            
            # Print summary
            analysis = analyze_survey_responses(survey_results)
            for question_id, counts in analysis.items():
                print(f"  {question_id}:")
                for response, count in counts.items():
                    print(f"    {response}: {count}")
    
    # Validation
    if args.validate:
        print("\n\nStep 4: Validating personas...")
        print("-" * 80)
        
        for method_name, personas in method_results.items():
            print(f"\nValidation for {method_name} method:")
            validation_results = validate_distributions(personas)
            print_validation_report(validation_results)
    
    # Comparison (if multiple methods)
    if len(method_results) > 1:
        print("\n\nStep 5: Comparing methods...")
        print("-" * 80)
        comparison_results = compare_methods(method_results)
        print_comparison_report(comparison_results)
    
    # Save results
    print("\n\nStep 6: Saving results...")
    print("-" * 80)
    
    for method_name, personas in method_results.items():
        # Save personas as JSON
        output_file = output_dir / f"{method_name}_personas.json"
        personas_dict = [p.to_dict() for p in personas]
        
        with open(output_file, 'w') as f:
            json.dump(personas_dict, f, indent=2)
        print(f"✓ Saved {len(personas)} personas to {output_file}")
        
        # Save survey results if available
        if method_name in survey_results_all:
            survey_file = output_dir / f"{method_name}_survey_results.json"
            with open(survey_file, 'w') as f:
                json.dump(survey_results_all[method_name], f, indent=2)
            print(f"✓ Saved survey results to {survey_file}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

