"""
Run multi-round simulation with belief updates and survey responses.

This script:
1. Loads personas
2. Creates dual-graph model with reduced influence
3. Runs multiple rounds of belief updates
4. Tracks survey response changes over time
5. Saves results
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from persona.persona import Persona
from networks import run_simulation, save_simulation_results, print_simulation_summary


def load_personas_from_json(filepath: str) -> list:
    """Load personas from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    personas = []
    for p_dict in data:
        persona = Persona(
            age=p_dict['age'],
            gender=p_dict['gender'],
            race=p_dict['race'],
            sexual_orientation=p_dict['sexual_orientation'],
            family_income=p_dict['family_income'],
            income_percentile=p_dict.get('income_percentile'),
            income_quintile=p_dict.get('income_quintile'),
            first_gen_college=p_dict['first_gen_college'],
            gpa=p_dict['gpa'],
            sports_participation=p_dict['sports_participation'],
            mental_health=p_dict['mental_health'],
            social_media_intensity=p_dict['social_media_intensity'],
            college_intention=p_dict['college_intention'],
            narrative=p_dict.get('narrative'),
            beliefs=p_dict.get('beliefs', {}),
            survey_responses=p_dict.get('survey_responses', {})
        )
        personas.append(persona)
    
    return personas


def main():
    """Run the simulation."""
    print("=" * 80)
    print("MULTI-ROUND SIMULATION WITH BELIEF UPDATES")
    print("=" * 80)
    
    # Load personas
    personas_file = Path('outputs/constraint_personas.json')
    if not personas_file.exists():
        print(f"Error: {personas_file} not found!")
        print("Please run persona generation first:")
        print("  python3 main.py --method constraint --n 100 --augment --surveys")
        return
    
    print(f"\nLoading personas from {personas_file}...")
    personas = load_personas_from_json(str(personas_file))
    print(f"✓ Loaded {len(personas)} personas")
    
    # Simulation parameters
    n_rounds = 5
    influence_reduction_factor = 0.15  # Reduce influence to 15% (more realistic)
    
    print(f"\nSimulation Parameters:")
    print(f"  Rounds: {n_rounds}")
    print(f"  Influence reduction factor: {influence_reduction_factor}")
    print(f"  (Influence weights reduced to {influence_reduction_factor*100:.0f}% of raw values)")
    
    # Run simulation
    print("\n" + "=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)
    
    final_personas, results = run_simulation(
        personas=personas,
        n_rounds=n_rounds,
        influence_reduction_factor=influence_reduction_factor,
        update_embeddings=True,  # Update embeddings after each round
        random_seed=42
    )
    
    # Print summary
    print_simulation_summary(results)
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_dir = Path('outputs/simulation')
    save_simulation_results(results, output_dir, prefix="multi_round")
    
    # Save final personas
    final_personas_dict = [p.to_dict() for p in final_personas]
    with open(output_dir / 'final_personas.json', 'w') as f:
        json.dump(final_personas_dict, f, indent=2)
    print(f"✓ Saved final personas to {output_dir / 'final_personas.json'}")
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nKey Files:")
    print(f"  - {output_dir}/multi_round_survey_evolution.json")
    print(f"  - {output_dir}/multi_round_influence_stats.json")
    print(f"  - {output_dir}/final_personas.json")


if __name__ == '__main__':
    main()

