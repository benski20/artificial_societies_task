"""
Visualize metrics comparisons between original and vector systems.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from surveys.questions import SURVEY_QUESTIONS

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def load_metrics():
    """Load metrics for both systems."""
    metrics_dir = Path('metrics/reports')
    
    with open(metrics_dir / 'original_metrics.json') as f:
        original = json.load(f)
    
    with open(metrics_dir / 'vector_metrics.json') as f:
        vector = json.load(f)
    
    return original, vector


def plot_embedding_spread_comparison(original: Dict, vector: Dict, output_dir: Path):
    """Plot embedding spread comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean pairwise distance
    ax = axes[0, 0]
    systems = ['Original', 'Vector']
    distances = [
        original['embedding_spread']['mean_pairwise_distance'],
        vector['embedding_spread']['mean_pairwise_distance']
    ]
    ax.bar(systems, distances, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax.set_ylabel('Mean Pairwise Distance', fontsize=11)
    ax.set_title('Embedding Spread: Mean Pairwise Distance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Distance range
    ax = axes[0, 1]
    ranges = [
        original['embedding_spread']['distance_range'],
        vector['embedding_spread']['distance_range']
    ]
    ax.bar(systems, ranges, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax.set_ylabel('Distance Range', fontsize=11)
    ax.set_title('Embedding Spread: Distance Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # PCA variance explained
    ax = axes[1, 0]
    orig_pca = original['embedding_spread']['pca_total_variance_explained']
    vec_pca = vector['embedding_spread']['pca_total_variance_explained']
    ax.bar(systems, [orig_pca, vec_pca], color=['#3498db', '#e74c3c'], alpha=0.7)
    ax.set_ylabel('Variance Explained (First 2 PCs)', fontsize=11)
    ax.set_title('PCA: Variance Explained', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Dimension variance
    ax = axes[1, 1]
    orig_dim_var = original['embedding_spread']['mean_dimension_variance']
    vec_dim_var = vector['embedding_spread']['mean_dimension_variance']
    ax.bar(systems, [orig_dim_var, vec_dim_var], color=['#3498db', '#e74c3c'], alpha=0.7)
    ax.set_ylabel('Mean Dimension Variance', fontsize=11)
    ax.set_title('Dimension-Wise Variance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'embedding_spread_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved embedding_spread_comparison.png")


def plot_response_distributions(original: Dict, vector: Dict, output_dir: Path):
    """Plot aggregate response distributions."""
    for question_id in SURVEY_QUESTIONS.keys():
        question_text = SURVEY_QUESTIONS[question_id]['question']
        options = SURVEY_QUESTIONS[question_id]['options']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original system
        ax = axes[0]
        orig_dist = original['aggregate_distributions'][question_id]['round_final']
        orig_counts = [orig_dist.get(opt, 0) for opt in options]
        orig_pcts = [c / sum(orig_counts) * 100 if sum(orig_counts) > 0 else 0 for c in orig_counts]
        
        bars = ax.bar(options, orig_pcts, color='#3498db', alpha=0.7)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'Original System: {question_text}', fontsize=12, fontweight='bold')
        ax.set_xticklabels(options, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, pct in zip(bars, orig_pcts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Vector system
        ax = axes[1]
        vec_dist = vector['aggregate_distributions'][question_id]['round_final']
        vec_counts = [vec_dist.get(opt, 0) for opt in options]
        vec_pcts = [c / sum(vec_counts) * 100 if sum(vec_counts) > 0 else 0 for c in vec_counts]
        
        bars = ax.bar(options, vec_pcts, color='#e74c3c', alpha=0.7)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'Vector System: {question_text}', fontsize=12, fontweight='bold')
        ax.set_xticklabels(options, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, pct in zip(bars, vec_pcts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        safe_qid = question_id.replace('_', '_')
        fig.savefig(output_dir / f'response_distribution_{safe_qid}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved response_distribution_{safe_qid}.png")


def plot_entropy_over_time(original: Dict, vector: Dict, output_dir: Path):
    """Plot response entropy over time."""
    fig, axes = plt.subplots(len(SURVEY_QUESTIONS), 1, figsize=(12, 5 * len(SURVEY_QUESTIONS)))
    
    if len(SURVEY_QUESTIONS) == 1:
        axes = [axes]
    
    for idx, (question_id, question_text) in enumerate(SURVEY_QUESTIONS.items()):
        ax = axes[idx]
        
        # Original system
        orig_entropy = original['response_entropy'][question_id]
        orig_rounds = [e['round'] for e in orig_entropy]
        orig_entropies = [e['entropy'] for e in orig_entropy]
        orig_max = orig_entropy[0]['max_entropy']
        
        ax.plot(orig_rounds, orig_entropies, marker='o', label='Original', 
               linewidth=2, color='#3498db')
        ax.axhline(y=orig_max, color='#3498db', linestyle='--', alpha=0.3, label='Max (Original)')
        
        # Vector system
        vec_entropy = vector['response_entropy'][question_id]
        vec_rounds = [e['round'] for e in vec_entropy]
        vec_entropies = [e['entropy'] for e in vec_entropy]
        vec_max = vec_entropy[0]['max_entropy']
        
        ax.plot(vec_rounds, vec_entropies, marker='s', label='Vector', 
               linewidth=2, color='#e74c3c')
        ax.axhline(y=vec_max, color='#e74c3c', linestyle='--', alpha=0.3, label='Max (Vector)')
        
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Entropy (bits)', fontsize=11)
        ax.set_title(f'Response Entropy Over Time: {question_text}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(orig_rounds)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'entropy_over_time.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved entropy_over_time.png")


def plot_opinion_clustering(original: Dict, vector: Dict, output_dir: Path):
    """Plot opinion clustering metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Homophily effect
    ax = axes[0, 0]
    systems = ['Original', 'Vector']
    homophily = [
        original['opinion_clustering']['homophily_effect'],
        vector['opinion_clustering']['homophily_effect']
    ]
    colors = ['green' if h > 0 else 'red' for h in homophily]
    ax.bar(systems, homophily, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Homophily Effect', fontsize=11)
    ax.set_title('Opinion Clustering: Homophily Effect', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Connected vs unconnected similarity
    ax = axes[0, 1]
    orig_conn = original['opinion_clustering']['connected_similarity']['mean']
    orig_unconn = original['opinion_clustering']['unconnected_similarity']['mean']
    vec_conn = vector['opinion_clustering']['connected_similarity']['mean']
    vec_unconn = vector['opinion_clustering']['unconnected_similarity']['mean']
    
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [orig_conn, vec_conn], width, label='Connected', 
           color='#3498db', alpha=0.7)
    ax.bar(x + width/2, [orig_unconn, vec_unconn], width, label='Unconnected', 
           color='#e74c3c', alpha=0.7)
    ax.set_ylabel('Mean Similarity', fontsize=11)
    ax.set_title('Connected vs Unconnected Similarity', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Clustering silhouette scores
    ax = axes[1, 0]
    n_clusters = [3, 5, 10]
    orig_silhouettes = [
        original['opinion_clustering']['clustering_analysis'][f'{n}_clusters']['silhouette_score']
        for n in n_clusters
    ]
    vec_silhouettes = [
        vector['opinion_clustering']['clustering_analysis'][f'{n}_clusters']['silhouette_score']
        for n in n_clusters
    ]
    
    x = np.arange(len(n_clusters))
    width = 0.35
    ax.bar(x - width/2, orig_silhouettes, width, label='Original', color='#3498db', alpha=0.7)
    ax.bar(x + width/2, vec_silhouettes, width, label='Vector', color='#e74c3c', alpha=0.7)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Clustering Quality (Silhouette Score)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n} clusters' for n in n_clusters])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Convergence comparison
    ax = axes[1, 1]
    convergence_data = []
    labels = []
    for qid in SURVEY_QUESTIONS.keys():
        orig_conv = original['convergence'][qid]['convergence_round']
        vec_conv = vector['convergence'][qid]['convergence_round']
        convergence_data.append([
            orig_conv if orig_conv else 10,  # Use 10 if no convergence
            vec_conv if vec_conv else 10
        ])
        labels.append(qid.replace('_', '\n'))
    
    x = np.arange(len(labels))
    width = 0.35
    orig_conv_rounds = [d[0] for d in convergence_data]
    vec_conv_rounds = [d[1] for d in convergence_data]
    ax.bar(x - width/2, orig_conv_rounds, width, label='Original', color='#3498db', alpha=0.7)
    ax.bar(x + width/2, vec_conv_rounds, width, label='Vector', color='#e74c3c', alpha=0.7)
    ax.set_ylabel('Convergence Round', fontsize=11)
    ax.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 6])
    
    plt.tight_layout()
    fig.savefig(output_dir / 'opinion_clustering_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved opinion_clustering_comparison.png")


def plot_network_metrics(original: Dict, vector: Dict, output_dir: Path):
    """Plot network structure metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    systems = ['Original', 'Vector']
    
    # In-degree centrality
    ax = axes[0, 0]
    orig_in = original['network_metrics']['in_degree_centrality']['mean']
    vec_in = vector['network_metrics']['in_degree_centrality']['mean']
    ax.bar(systems, [orig_in, vec_in], color=['#3498db', '#e74c3c'], alpha=0.7)
    ax.set_ylabel('Mean In-Degree Centrality', fontsize=11)
    ax.set_title('In-Degree Centrality', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Out-degree centrality
    ax = axes[0, 1]
    orig_out = original['network_metrics']['out_degree_centrality']['mean']
    vec_out = vector['network_metrics']['out_degree_centrality']['mean']
    ax.bar(systems, [orig_out, vec_out], color=['#3498db', '#e74c3c'], alpha=0.7)
    ax.set_ylabel('Mean Out-Degree Centrality', fontsize=11)
    ax.set_title('Out-Degree Centrality', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Influence strength
    ax = axes[1, 0]
    orig_in_str = original['network_metrics']['in_strength']['mean']
    vec_in_str = vector['network_metrics']['in_strength']['mean']
    orig_out_str = original['network_metrics']['out_strength']['mean']
    vec_out_str = vector['network_metrics']['out_strength']['mean']
    
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [orig_in_str, vec_in_str], width, label='In-Strength', 
           color='#3498db', alpha=0.7)
    ax.bar(x + width/2, [orig_out_str, vec_out_str], width, label='Out-Strength', 
           color='#e74c3c', alpha=0.7)
    ax.set_ylabel('Mean Strength', fontsize=11)
    ax.set_title('Influence Strength', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Network density
    ax = axes[1, 1]
    orig_density = original['network_metrics']['density']
    vec_density = vector['network_metrics']['density']
    ax.bar(systems, [orig_density, vec_density], color=['#3498db', '#e74c3c'], alpha=0.7)
    ax.set_ylabel('Network Density', fontsize=11)
    ax.set_title('Influence Network Density', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'network_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved network_metrics_comparison.png")


def plot_response_evolution_comparison(original: Dict, vector: Dict, output_dir: Path):
    """Plot response evolution side-by-side."""
    for question_id in SURVEY_QUESTIONS.keys():
        question_text = SURVEY_QUESTIONS[question_id]['question']
        options = SURVEY_QUESTIONS[question_id]['options']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original system evolution
        ax = axes[0]
        orig_evolution = original['aggregate_distributions'][question_id]['all_rounds']
        rounds = list(range(len(orig_evolution)))
        
        for opt in options:
            percentages = []
            for round_dist in orig_evolution:
                total = sum(round_dist.values())
                pct = (round_dist.get(opt, 0) / total * 100) if total > 0 else 0
                percentages.append(pct)
            ax.plot(rounds, percentages, marker='o', label=opt, linewidth=2)
        
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'Original System: {question_text}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rounds)
        
        # Vector system evolution
        ax = axes[1]
        vec_evolution = vector['aggregate_distributions'][question_id]['all_rounds']
        rounds = list(range(len(vec_evolution)))
        
        for opt in options:
            percentages = []
            for round_dist in vec_evolution:
                total = sum(round_dist.values())
                pct = (round_dist.get(opt, 0) / total * 100) if total > 0 else 0
                percentages.append(pct)
            ax.plot(rounds, percentages, marker='s', label=opt, linewidth=2)
        
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'Vector System: {question_text}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rounds)
        
        plt.tight_layout()
        safe_qid = question_id.replace('_', '_')
        fig.savefig(output_dir / f'response_evolution_{safe_qid}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved response_evolution_{safe_qid}.png")


def main():
    """Main function."""
    print("=" * 80)
    print("METRICS VISUALIZATION")
    print("=" * 80)
    
    # Load metrics
    print("\nLoading metrics...")
    original, vector = load_metrics()
    
    # Create output directory
    output_dir = Path('metrics/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    # Generate all plots
    plot_embedding_spread_comparison(original, vector, output_dir)
    plot_response_distributions(original, vector, output_dir)
    plot_entropy_over_time(original, vector, output_dir)
    plot_opinion_clustering(original, vector, output_dir)
    plot_network_metrics(original, vector, output_dir)
    plot_response_evolution_comparison(original, vector, output_dir)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == '__main__':
    main()

