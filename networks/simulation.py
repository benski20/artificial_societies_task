"""
Multi-round simulation pipeline.

Runs multiple rounds of belief updates and survey responses to track
how opinions change over time.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from collections import defaultdict

from persona.persona import Persona
from embeddings import generate_embeddings, add_embeddings_to_personas
from networks.dual_graph import DualGraphModel, create_dual_graph_model
from networks.belief_updates import update_all_beliefs
from surveys.responses import generate_survey_responses
from surveys.questions import SURVEY_QUESTIONS


class SimulationResults:
    """Container for simulation results."""
    
    def __init__(self):
        self.rounds = []
        self.survey_responses_by_round = defaultdict(list)
        self.belief_changes = defaultdict(list)
        self.influence_stats = []
    
    def add_round(self, round_num: int, personas: List[Persona], 
                 survey_responses: Dict, dual_model: DualGraphModel):
        """Add results from a simulation round."""
        self.rounds.append({
            'round': round_num,
            'n_personas': len(personas)
        })
        
        # Store survey responses
        for question_id, responses in survey_responses.items():
            self.survey_responses_by_round[question_id].append(responses)
        
        # Store influence stats
        stats = dual_model.statistics()
        self.influence_stats.append(stats)
    
    def get_response_distribution(self, question_id: str, round_num: int) -> Dict:
        """Get response distribution for a question in a specific round."""
        if round_num >= len(self.survey_responses_by_round[question_id]):
            return {}
        
        responses = self.survey_responses_by_round[question_id][round_num]
        from collections import Counter
        return dict(Counter([r['response'] for r in responses]))


def run_simulation(personas: List[Persona],
                  n_rounds: int = 5,
                  topics: Optional[List[str]] = None,
                  social_k: int = 4,
                  social_p: float = 0.2,
                  influence_reduction_factor: float = 0.15,
                  update_embeddings: bool = True,
                  random_seed: Optional[int] = None) -> Tuple[List[Persona], SimulationResults]:
    """
    Run multi-round simulation of belief updates.
    
    Args:
        personas: Initial list of personas
        n_rounds: Number of simulation rounds
        topics: Belief topics to update (default: all topics)
        social_k: Watts-Strogatz k parameter
        social_p: Watts-Strogatz p parameter
        influence_reduction_factor: Factor to reduce influence weights
        update_embeddings: Whether to update embeddings after each round
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (final_personas, simulation_results)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if topics is None:
        # Default topics from persona beliefs
        if personas and personas[0].beliefs:
            topics = list(personas[0].beliefs.keys())
        else:
            topics = [
                'college importance and future plans',
                'social media impact on daily life',
                'school start times and sleep',
                'mental health and stress',
                'academic pressure and expectations'
            ]
    
    results = SimulationResults()
    current_personas = personas.copy()
    
    # Initial embeddings
    print("Generating initial embeddings...")
    embeddings = generate_embeddings(current_personas, show_progress=False)
    current_personas = add_embeddings_to_personas(current_personas, embeddings)
    
    # Create initial dual-graph model
    dual_model = create_dual_graph_model(
        n_personas=len(current_personas),
        embeddings=embeddings,
        social_k=social_k,
        social_p=social_p,
        influence_reduction_factor=influence_reduction_factor,
        seed=random_seed
    )
    
    # Initial survey responses
    print("Generating initial survey responses...")
    initial_responses = generate_survey_responses(
        current_personas, use_llm=False, random_seed=random_seed
    )
    results.add_round(0, current_personas, initial_responses, dual_model)
    
    # Run simulation rounds
    for round_num in range(1, n_rounds + 1):
        print(f"\nRound {round_num}/{n_rounds}...")
        
        # Update beliefs
        print("  Updating beliefs...")
        current_personas = update_all_beliefs(
            current_personas, dual_model, topics, random_seed=random_seed
        )
        
        # Update embeddings if requested
        if update_embeddings:
            print("  Updating embeddings...")
            new_embeddings = generate_embeddings(current_personas, show_progress=False)
            current_personas = add_embeddings_to_personas(current_personas, new_embeddings)
            
            # Update cognitive graph
            dual_model.update_cognitive_graph(new_embeddings)
        
        # Generate survey responses
        print("  Generating survey responses...")
        round_responses = generate_survey_responses(
            current_personas, use_llm=False, random_seed=random_seed
        )
        
        # Store results
        results.add_round(round_num, current_personas, round_responses, dual_model)
    
    return current_personas, results


def save_simulation_results(results: SimulationResults,
                           output_dir: Path,
                           prefix: str = "simulation"):
    """Save simulation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save survey response evolution
    survey_evolution = {}
    for question_id in SURVEY_QUESTIONS.keys():
        question_evolution = []
        for round_num in range(len(results.survey_responses_by_round[question_id])):
            dist = results.get_response_distribution(question_id, round_num)
            question_evolution.append({
                'round': round_num,
                'distribution': dist,
                'total': sum(dist.values())
            })
        survey_evolution[question_id] = question_evolution
    
    with open(output_dir / f"{prefix}_survey_evolution.json", 'w') as f:
        json.dump(survey_evolution, f, indent=2)
    
    # Save influence statistics
    with open(output_dir / f"{prefix}_influence_stats.json", 'w') as f:
        json.dump(results.influence_stats, f, indent=2)
    
    print(f"✓ Saved simulation results to {output_dir}/")


def print_simulation_summary(results: SimulationResults):
    """Print summary of simulation results."""
    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal rounds: {len(results.rounds)}")
    
    # Show survey response changes
    print("\nSurvey Response Evolution:")
    print("-" * 80)
    
    for question_id in SURVEY_QUESTIONS.keys():
        question_text = SURVEY_QUESTIONS[question_id]['question']
        print(f"\n{question_text}:")
        
        for round_num in range(min(3, len(results.survey_responses_by_round[question_id]))):
            dist = results.get_response_distribution(question_id, round_num)
            print(f"  Round {round_num}:")
            for response, count in sorted(dist.items(), key=lambda x: -x[1])[:3]:
                percentage = (count / sum(dist.values())) * 100
                print(f"    {response}: {count} ({percentage:.1f}%)")
        
        # Show change from first to last
        if len(results.survey_responses_by_round[question_id]) > 1:
            first_dist = results.get_response_distribution(question_id, 0)
            last_dist = results.get_response_distribution(question_id, -1)
            
            print(f"  Change (Round 0 → Round {len(results.rounds)-1}):")
            for response in set(list(first_dist.keys()) + list(last_dist.keys())):
                first_count = first_dist.get(response, 0)
                last_count = last_dist.get(response, 0)
                change = last_count - first_count
                if change != 0:
                    print(f"    {response}: {first_count} → {last_count} ({change:+d})")

