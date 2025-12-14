"""
Visualize vector-based simulation results.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from surveys.questions import SURVEY_QUESTIONS


def load_simulation_results(results_dir: Path):
    """Load simulation results."""
    survey_file = results_dir / "survey_evolution.json"
    metrics_file = results_dir / "vector_metrics.json"
    
    with open(survey_file, 'r') as f:
        survey_evolution = json.load(f)
    
    with open(metrics_file, 'r') as f:
        vector_metrics = json.load(f)
    
    return survey_evolution, vector_metrics


def plot_survey_evolution(survey_evolution: dict, output_dir: Path):
    """Plot survey response evolution over rounds."""
    
    for question_id, evolution in survey_evolution.items():
        question_text = SURVEY_QUESTIONS[question_id]['question']
        options = SURVEY_QUESTIONS[question_id]['options']
        
        # Extract data
        rounds = []
        response_counts = {opt: [] for opt in options}
        
        for round_data in evolution:
            rounds.append(round_data['round'])
            dist = round_data['distribution']
            total = round_data['total']
            
            for opt in options:
                count = dist.get(opt, 0)
                percentage = (count / total * 100) if total > 0 else 0
                response_counts[opt].append(percentage)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each response option
        for opt in options:
            ax.plot(rounds, response_counts[opt], marker='o', label=opt, linewidth=2)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title(f'Survey Response Evolution (Vector-Based): {question_text}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rounds)
        
        plt.tight_layout()
        
        # Save
        safe_question_id = question_id.replace('_', '_')
        filename = f"survey_evolution_{safe_question_id}.png"
        fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Saved {filename}")


def plot_vector_metrics(vector_metrics: list, output_dir: Path):
    """Plot belief vector metrics over rounds."""
    
    rounds = [m['round'] for m in vector_metrics]
    
    # Extract metrics
    vector_norms_mean = [m['vector_metrics']['vector_norms']['mean'] for m in vector_metrics]
    pairwise_sim_mean = [m['vector_metrics']['pairwise_similarity']['mean'] for m in vector_metrics]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Vector norms
    ax1.plot(rounds, vector_norms_mean, marker='o', linewidth=2, color='blue')
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Mean Vector Norm', fontsize=11)
    ax1.set_title('Belief Vector Norms Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rounds)
    
    # Pairwise similarity
    ax2.plot(rounds, pairwise_sim_mean, marker='o', linewidth=2, color='green')
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Mean Pairwise Similarity', fontsize=11)
    ax2.set_title('Belief Vector Similarity Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds)
    
    plt.tight_layout()
    fig.savefig(output_dir / "vector_metrics.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved vector_metrics.png")


def plot_change_summary(survey_evolution: dict, output_dir: Path):
    """Plot summary of changes from first to last round."""
    
    changes = {}
    
    for question_id, evolution in survey_evolution.items():
        if len(evolution) < 2:
            continue
        
        first_round = evolution[0]['distribution']
        last_round = evolution[-1]['distribution']
        
        question_text = SURVEY_QUESTIONS[question_id]['question']
        changes[question_text] = {}
        
        all_responses = set(list(first_round.keys()) + list(last_round.keys()))
        for response in all_responses:
            first_count = first_round.get(response, 0)
            last_count = last_round.get(response, 0)
            change = last_count - first_count
            if change != 0:
                changes[question_text][response] = change
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    y_pos = 0
    color_idx = 0
    
    for question_text, response_changes in changes.items():
        if not response_changes:
            continue
        
        # Sort by absolute change
        sorted_changes = sorted(response_changes.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for response, change in sorted_changes:
            color = 'green' if change > 0 else 'red'
            ax.barh(y_pos, change, color=color, alpha=0.7)
            ax.text(change + (0.5 if change > 0 else -0.5), y_pos, 
                   f"{response}: {change:+d}", 
                   va='center', fontsize=9)
            y_pos += 1
        
        y_pos += 0.5  # Space between questions
    
    ax.set_yticks([])
    ax.set_xlabel('Change in Count (Round 0 → Final Round)', fontsize=12)
    ax.set_title('Survey Response Changes Over Simulation (Vector-Based)', 
                fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig.savefig(output_dir / "change_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved change_summary.png")


def main():
    """Main function."""
    print("=" * 80)
    print("VECTOR-BASED SIMULATION VISUALIZATION")
    print("=" * 80)
    
    results_dir = Path('outputs/vector_beliefs_simulation')
    
    if not (results_dir / "survey_evolution.json").exists():
        print(f"Error: Simulation results not found in {results_dir}")
        print("Please run simulation first:")
        print("  python3 experiments/vector_beliefs/run_vector_simulation.py")
        return
    
    print(f"\nLoading simulation results from {results_dir}...")
    survey_evolution, vector_metrics = load_simulation_results(results_dir)
    
    # Create output directory
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    print("-" * 80)
    
    # Individual plots
    plot_survey_evolution(survey_evolution, viz_dir)
    
    # Vector metrics
    plot_vector_metrics(vector_metrics, viz_dir)
    
    # Change summary
    plot_change_summary(survey_evolution, viz_dir)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {viz_dir}/")


if __name__ == '__main__':
    main()

