"""
Run multi-round simulation with vector-based belief system.

This script:
1. Loads personas and converts to VectorPersona
2. Creates dual-graph model
3. Runs multiple rounds of belief vector updates
4. Tracks survey response changes over time
5. Reports metrics
6. Saves results
"""

import json
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from persona.persona import Persona
from experiments.vector_beliefs.persona.vector_persona import VectorPersona
from experiments.vector_beliefs.surveys.question_embeddings import embed_survey_questions
from experiments.vector_beliefs.surveys.vector_responses import generate_all_responses
from experiments.vector_beliefs.networks.vector_belief_updates import update_all_vector_beliefs
from experiments.vector_beliefs.embeddings.belief_embeddings import generate_embeddings_from_vectors
from experiments.vector_beliefs.utils.metrics import calculate_all_metrics
from networks.dual_graph import create_dual_graph_model
from surveys.questions import SURVEY_QUESTIONS


def load_personas_from_json(filepath: str) -> List[Persona]:
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


def convert_to_vector_personas(personas: List[Persona], belief_dim: int = 16) -> List[VectorPersona]:
    """Convert Persona objects to VectorPersona objects."""
    vector_personas = []
    for persona in personas:
        vector_persona = VectorPersona(
            age=persona.age,
            gender=persona.gender,
            race=persona.race,
            sexual_orientation=persona.sexual_orientation,
            family_income=persona.family_income,
            income_percentile=persona.income_percentile,
            income_quintile=persona.income_quintile,
            first_gen_college=persona.first_gen_college,
            gpa=persona.gpa,
            sports_participation=persona.sports_participation,
            mental_health=persona.mental_health,
            social_media_intensity=persona.social_media_intensity,
            college_intention=persona.college_intention,
            narrative=persona.narrative,
            belief_dim=belief_dim
        )
        vector_personas.append(vector_persona)
    
    return vector_personas


def collect_survey_responses(personas: List[VectorPersona],
                            question_id: str) -> Dict[str, int]:
    """Collect survey responses for a question."""
    responses = [p.survey_responses.get(question_id) for p in personas]
    responses = [r for r in responses if r is not None]
    return dict(Counter(responses))


def run_vector_simulation(personas: List[VectorPersona],
                         n_rounds: int = 5,
                         belief_dim: int = 16,
                         social_k: int = 4,
                         social_p: float = 0.2,
                         influence_reduction_factor: float = 0.15,
                         temperature: float = 1.0,
                         random_seed: Optional[int] = None) -> tuple:
    """
    Run multi-round simulation with vector-based beliefs.
    
    Returns:
        Tuple of (final_personas, results_dict)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    question_ids = list(SURVEY_QUESTIONS.keys())
    
    # Initialize question embedder
    print("Embedding survey questions...")
    question_embedder = embed_survey_questions(belief_dim=belief_dim)
    
    # Generate initial embeddings from belief vectors
    print("Generating initial embeddings from belief vectors...")
    embeddings = generate_embeddings_from_vectors(personas, embedding_dim=384)
    
    # Create dual-graph model
    print("Creating dual-graph model...")
    dual_model = create_dual_graph_model(
        len(personas),
        embeddings,
        social_k=social_k,
        social_p=social_p,
        influence_reduction_factor=influence_reduction_factor,
        seed=random_seed
    )
    
    # Generate initial survey responses
    print("Generating initial survey responses...")
    personas = generate_all_responses(
        personas, question_embedder, question_ids,
        temperature=temperature, random_seed=random_seed
    )
    
    # Track results
    results = {
        'survey_evolution': {qid: [] for qid in question_ids},
        'influence_stats': [],
        'vector_metrics': []
    }
    
    # Record round 0
    for question_id in question_ids:
        responses = collect_survey_responses(personas, question_id)
        results['survey_evolution'][question_id].append({
            'round': 0,
            'distribution': responses,
            'total': len(personas)
        })
    
    # Record initial metrics
    stats = dual_model.statistics()
    results['influence_stats'].append({
        'round': 0,
        **stats
    })
    
    metrics = calculate_all_metrics(personas, question_ids)
    results['vector_metrics'].append({
        'round': 0,
        **metrics
    })
    
    # Run rounds
    for round_num in range(1, n_rounds + 1):
        print(f"\nRound {round_num}/{n_rounds}...")
        
        # Update belief vectors
        print("  Updating belief vectors...")
        personas = update_all_vector_beliefs(
            personas, dual_model, random_seed=random_seed + round_num if random_seed else None
        )
        
        # Update embeddings (belief vectors changed)
        print("  Updating embeddings...")
        embeddings = generate_embeddings_from_vectors(personas, embedding_dim=384)
        dual_model.update_cognitive_graph(embeddings)
        
        # Generate survey responses
        print("  Generating survey responses...")
        personas = generate_all_responses(
            personas, question_embedder, question_ids,
            temperature=temperature, random_seed=random_seed + round_num * 100 if random_seed else None
        )
        
        # Record results
        for question_id in question_ids:
            responses = collect_survey_responses(personas, question_id)
            results['survey_evolution'][question_id].append({
                'round': round_num,
                'distribution': responses,
                'total': len(personas)
            })
        
        stats = dual_model.statistics()
        results['influence_stats'].append({
            'round': round_num,
            **stats
        })
        
        metrics = calculate_all_metrics(personas, question_ids)
        results['vector_metrics'].append({
            'round': round_num,
            **metrics
        })
    
    return personas, results


def print_simulation_summary(results: Dict):
    """Print simulation summary."""
    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal rounds: {len(results['survey_evolution'][list(results['survey_evolution'].keys())[0]])}")
    
    print("\nSurvey Response Evolution:")
    print("-" * 80)
    
    for question_id, evolution in results['survey_evolution'].items():
        question_text = SURVEY_QUESTIONS[question_id]['question']
        print(f"\n{question_text}:")
        
        first_round = evolution[0]
        last_round = evolution[-1]
        
        print(f"  Round {first_round['round']}:")
        for option, count in sorted(first_round['distribution'].items()):
            pct = count / first_round['total'] * 100
            print(f"    {option}: {count} ({pct:.1f}%)")
        
        print(f"  Round {last_round['round']}:")
        for option, count in sorted(last_round['distribution'].items()):
            pct = count / last_round['total'] * 100
            print(f"    {option}: {count} ({pct:.1f}%)")
        
        # Calculate changes
        first_dist = first_round['distribution']
        last_dist = last_round['distribution']
        all_options = set(list(first_dist.keys()) + list(last_dist.keys()))
        changes = {
            opt: last_dist.get(opt, 0) - first_dist.get(opt, 0)
            for opt in all_options
        }
        changes = {k: v for k, v in changes.items() if v != 0}
        
        if changes:
            print(f"  Change (Round {first_round['round']} → Round {last_round['round']}):")
            for option, change in sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"    {option}: {first_dist.get(option, 0)} → {last_dist.get(option, 0)} ({change:+d})")
    
    # Vector metrics
    print("\n" + "=" * 80)
    print("VECTOR METRICS")
    print("=" * 80)
    
    final_metrics = results['vector_metrics'][-1]['vector_metrics']
    print(f"\nBelief Vector Statistics:")
    print(f"  Dimension: {final_metrics['belief_dim']}")
    print(f"  Vector Norms:")
    print(f"    Mean: {final_metrics['vector_norms']['mean']:.4f}")
    print(f"    Std: {final_metrics['vector_norms']['std']:.4f}")
    print(f"    Range: [{final_metrics['vector_norms']['min']:.4f}, {final_metrics['vector_norms']['max']:.4f}]")
    print(f"  Pairwise Similarity:")
    print(f"    Mean: {final_metrics['pairwise_similarity']['mean']:.4f}")
    print(f"    Std: {final_metrics['pairwise_similarity']['std']:.4f}")


def save_results(personas: List[VectorPersona],
                results: Dict,
                output_dir: Path):
    """Save simulation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save survey evolution
    with open(output_dir / "survey_evolution.json", 'w') as f:
        json.dump(results['survey_evolution'], f, indent=2)
    
    # Save influence stats
    with open(output_dir / "influence_stats.json", 'w') as f:
        json.dump(results['influence_stats'], f, indent=2)
    
    # Save vector metrics
    with open(output_dir / "vector_metrics.json", 'w') as f:
        json.dump(results['vector_metrics'], f, indent=2)
    
    # Save final personas
    personas_dict = [p.to_dict() for p in personas]
    with open(output_dir / "final_personas.json", 'w') as f:
        json.dump(personas_dict, f, indent=2)
    
    print(f"\n✓ Saved results to {output_dir}/")


def main():
    """Main function."""
    print("=" * 80)
    print("VECTOR-BASED BELIEF SIMULATION")
    print("=" * 80)
    
    # Load personas
    personas_file = Path('outputs/constraint_personas.json')
    print(f"\nLoading personas from {personas_file}...")
    personas = load_personas_from_json(str(personas_file))
    print(f"✓ Loaded {len(personas)} personas")
    
    # Convert to vector personas
    print("\nConverting to vector personas...")
    belief_dim = 16
    vector_personas = convert_to_vector_personas(personas, belief_dim=belief_dim)
    print(f"✓ Created {len(vector_personas)} vector personas (belief_dim={belief_dim})")
    
    # Simulation parameters
    n_rounds = 5
    influence_reduction_factor = 0.15
    temperature = 1.0
    
    print(f"\nSimulation Parameters:")
    print(f"  Rounds: {n_rounds}")
    print(f"  Belief dimension: {belief_dim}")
    print(f"  Influence reduction factor: {influence_reduction_factor}")
    print(f"  Response temperature: {temperature}")
    
    # Run simulation
    print("\n" + "=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)
    
    final_personas, results = run_vector_simulation(
        vector_personas,
        n_rounds=n_rounds,
        belief_dim=belief_dim,
        influence_reduction_factor=influence_reduction_factor,
        temperature=temperature,
        random_seed=42
    )
    
    # Print summary
    print_simulation_summary(results)
    
    # Save results
    output_dir = Path('outputs/vector_beliefs_simulation')
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    save_results(final_personas, results, output_dir)
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()

