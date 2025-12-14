"""
Generate comprehensive comparison report between original and vector systems.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from surveys.questions import SURVEY_QUESTIONS


def load_metrics():
    """Load metrics for both systems."""
    metrics_dir = Path('metrics/reports')
    
    with open(metrics_dir / 'original_metrics.json') as f:
        original = json.load(f)
    
    with open(metrics_dir / 'vector_metrics.json') as f:
        vector = json.load(f)
    
    return original, vector


def generate_comparison_report(original: dict, vector: dict) -> str:
    """Generate comprehensive comparison report."""
    lines = []
    lines.append("=" * 80)
    lines.append("COMPREHENSIVE METRICS COMPARISON REPORT")
    lines.append("Original Text-Based System vs Vector-Based System")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"Total Personas: {original['n_personas']} (both systems)")
    
    # 1. Belief Space Diversity
    lines.append("\n" + "=" * 80)
    lines.append("1. BELIEF SPACE DIVERSITY (Embedding Spread)")
    lines.append("=" * 80)
    
    orig_emb = original['embedding_spread']
    vec_emb = vector['embedding_spread']
    
    lines.append("\nMean Pairwise Distance:")
    lines.append(f"  Original: {orig_emb['mean_pairwise_distance']:.4f}")
    lines.append(f"  Vector: {vec_emb['mean_pairwise_distance']:.4f}")
    lines.append(f"  Difference: {vec_emb['mean_pairwise_distance'] - orig_emb['mean_pairwise_distance']:+.4f}")
    lines.append(f"  → Vector system has {'more' if vec_emb['mean_pairwise_distance'] > orig_emb['mean_pairwise_distance'] else 'less'} diversity")
    
    lines.append("\nDistance Range:")
    lines.append(f"  Original: {orig_emb['distance_range']:.4f}")
    lines.append(f"  Vector: {vec_emb['distance_range']:.4f}")
    
    lines.append("\nPCA Variance Explained (First 2 PCs):")
    lines.append(f"  Original: {orig_emb['pca_total_variance_explained']:.1%}")
    lines.append(f"  Vector: {vec_emb['pca_total_variance_explained']:.1%}")
    
    if vector.get('belief_vector_diversity'):
        vec_belief = vector['belief_vector_diversity']
        lines.append("\nBelief Vector Diversity (Vector System Only):")
        lines.append(f"  Mean Pairwise Distance: {vec_belief['mean_pairwise_distance']:.4f}")
        lines.append(f"  Distance Range: {vec_belief['distance_range']:.4f}")
    
    # 2. Aggregate Response Distributions
    lines.append("\n" + "=" * 80)
    lines.append("2. AGGREGATE RESPONSE DISTRIBUTIONS")
    lines.append("=" * 80)
    
    for question_id, question_text in SURVEY_QUESTIONS.items():
        lines.append(f"\n{question_text}:")
        
        orig_final = original['aggregate_distributions'][question_id]['round_final']
        vec_final = vector['aggregate_distributions'][question_id]['round_final']
        
        lines.append("  Original System (Final Round):")
        for opt, count in sorted(orig_final.items(), key=lambda x: x[1], reverse=True):
            pct = count / sum(orig_final.values()) * 100
            lines.append(f"    {opt}: {count} ({pct:.1f}%)")
        
        lines.append("  Vector System (Final Round):")
        for opt, count in sorted(vec_final.items(), key=lambda x: x[1], reverse=True):
            pct = count / sum(vec_final.values()) * 100
            lines.append(f"    {opt}: {count} ({pct:.1f}%)")
        
        # Calculate difference
        all_options = set(list(orig_final.keys()) + list(vec_final.keys()))
        lines.append("  Differences:")
        for opt in all_options:
            orig_count = orig_final.get(opt, 0)
            vec_count = vec_final.get(opt, 0)
            diff = vec_count - orig_count
            if diff != 0:
                lines.append(f"    {opt}: {diff:+d}")
    
    # 3. Response Entropy
    lines.append("\n" + "=" * 80)
    lines.append("3. RESPONSE ENTROPY (Uncertainty)")
    lines.append("=" * 80)
    lines.append("\nHigher entropy = more uncertainty/diversity in responses")
    lines.append("Lower entropy = more certainty/concentration")
    
    for question_id, question_text in SURVEY_QUESTIONS.items():
        lines.append(f"\n{question_text}:")
        
        orig_entropy = original['response_entropy'][question_id]
        vec_entropy = vector['response_entropy'][question_id]
        
        # Initial entropy
        orig_init = orig_entropy[0]['entropy']
        vec_init = vec_entropy[0]['entropy']
        lines.append(f"  Initial Entropy:")
        lines.append(f"    Original: {orig_init:.4f} bits")
        lines.append(f"    Vector: {vec_init:.4f} bits")
        
        # Final entropy
        orig_final = orig_entropy[-1]['entropy']
        vec_final = vec_entropy[-1]['entropy']
        lines.append(f"  Final Entropy:")
        lines.append(f"    Original: {orig_final:.4f} bits")
        lines.append(f"    Vector: {vec_final:.4f} bits")
        
        # Change
        orig_change = orig_final - orig_init
        vec_change = vec_final - vec_init
        lines.append(f"  Change:")
        lines.append(f"    Original: {orig_change:+.4f} bits")
        lines.append(f"    Vector: {vec_change:+.4f} bits")
        
        # Max entropy
        orig_max = orig_entropy[0]['max_entropy']
        vec_max = vec_entropy[0]['max_entropy']
        lines.append(f"  Max Possible Entropy: {orig_max:.4f} bits")
        lines.append(f"  Relative Entropy (Final):")
        lines.append(f"    Original: {orig_final/orig_max:.1%} of max")
        lines.append(f"    Vector: {vec_final/vec_max:.1%} of max")
    
    # 4. Opinion Clustering
    lines.append("\n" + "=" * 80)
    lines.append("4. OPINION CLUSTERING (Network Homophily)")
    lines.append("=" * 80)
    lines.append("\nHomophily Effect = Connected Similarity - Unconnected Similarity")
    lines.append("Positive = beliefs cluster along social connections (homophily)")
    lines.append("Negative = beliefs don't cluster (heterophily)")
    
    orig_clust = original['opinion_clustering']
    vec_clust = vector['opinion_clustering']
    
    lines.append("\nConnected Similarity (socially connected pairs):")
    lines.append(f"  Original: {orig_clust['connected_similarity']['mean']:.4f}")
    lines.append(f"  Vector: {vec_clust['connected_similarity']['mean']:.4f}")
    
    lines.append("\nUnconnected Similarity (not socially connected):")
    lines.append(f"  Original: {orig_clust['unconnected_similarity']['mean']:.4f}")
    lines.append(f"  Vector: {vec_clust['unconnected_similarity']['mean']:.4f}")
    
    lines.append("\nHomophily Effect:")
    orig_homophily = orig_clust['homophily_effect']
    vec_homophily = vec_clust['homophily_effect']
    lines.append(f"  Original: {orig_homophily:+.4f}")
    lines.append(f"  Vector: {vec_homophily:+.4f}")
    
    if orig_homophily > 0:
        lines.append(f"  → Original system shows homophily (beliefs cluster)")
    else:
        lines.append(f"  → Original system shows heterophily (beliefs don't cluster)")
    
    if vec_homophily > 0:
        lines.append(f"  → Vector system shows homophily (beliefs cluster)")
    else:
        lines.append(f"  → Vector system shows heterophily (beliefs don't cluster)")
    
    lines.append("\nClustering Quality (Silhouette Score):")
    for n_clusters in [3, 5, 10]:
        orig_sil = orig_clust['clustering_analysis'][f'{n_clusters}_clusters']['silhouette_score']
        vec_sil = vec_clust['clustering_analysis'][f'{n_clusters}_clusters']['silhouette_score']
        lines.append(f"  {n_clusters} clusters:")
        lines.append(f"    Original: {orig_sil:.4f}")
        lines.append(f"    Vector: {vec_sil:.4f}")
    
    # 5. Convergence Metrics
    lines.append("\n" + "=" * 80)
    lines.append("5. CONVERGENCE METRICS")
    lines.append("=" * 80)
    
    for question_id, question_text in SURVEY_QUESTIONS.items():
        lines.append(f"\n{question_text}:")
        
        orig_conv = original['convergence'][question_id]
        vec_conv = vector['convergence'][question_id]
        
        lines.append(f"  Convergence Round (95% stability):")
        lines.append(f"    Original: Round {orig_conv['convergence_round'] if orig_conv['convergence_round'] else 'N/A (no convergence)'}")
        lines.append(f"    Vector: Round {vec_conv['convergence_round'] if vec_conv['convergence_round'] else 'N/A (no convergence)'}")
        
        lines.append(f"  Mean Stability:")
        lines.append(f"    Original: {orig_conv['mean_stability']:.1%}")
        lines.append(f"    Vector: {vec_conv['mean_stability']:.1%}")
        
        lines.append(f"  Final Polarization (variance):")
        lines.append(f"    Original: {orig_conv['final_polarization']:.4f}")
        lines.append(f"    Vector: {vec_conv['final_polarization']:.4f}")
        
        lines.append(f"  Total Change:")
        lines.append(f"    Original: {orig_conv['total_change']:.1%}")
        lines.append(f"    Vector: {vec_conv['total_change']:.1%}")
    
    # 6. Network Metrics
    lines.append("\n" + "=" * 80)
    lines.append("6. INFLUENCE NETWORK METRICS")
    lines.append("=" * 80)
    
    orig_net = original['network_metrics']
    vec_net = vector['network_metrics']
    
    lines.append("\nNetwork Structure:")
    lines.append(f"  Density:")
    lines.append(f"    Original: {orig_net['density']:.4f}")
    lines.append(f"    Vector: {vec_net['density']:.4f}")
    
    lines.append("\nIn-Degree Centrality (who influences you):")
    lines.append(f"  Original: {orig_net['in_degree_centrality']['mean']:.4f}")
    lines.append(f"  Vector: {vec_net['in_degree_centrality']['mean']:.4f}")
    
    lines.append("\nOut-Degree Centrality (who you influence):")
    lines.append(f"  Original: {orig_net['out_degree_centrality']['mean']:.4f}")
    lines.append(f"  Vector: {vec_net['out_degree_centrality']['mean']:.4f}")
    
    lines.append("\nInfluence Strength:")
    lines.append(f"  In-Strength (Original): {orig_net['in_strength']['mean']:.4f}")
    lines.append(f"  In-Strength (Vector): {vec_net['in_strength']['mean']:.4f}")
    lines.append(f"  Out-Strength (Original): {orig_net['out_strength']['mean']:.4f}")
    lines.append(f"  Out-Strength (Vector): {vec_net['out_strength']['mean']:.4f}")
    
    # 7. Key Insights
    lines.append("\n" + "=" * 80)
    lines.append("7. KEY INSIGHTS & INTERPRETATIONS")
    lines.append("=" * 80)
    
    lines.append("\nBelief Diversity:")
    if vec_emb['mean_pairwise_distance'] > orig_emb['mean_pairwise_distance']:
        lines.append("  → Vector system maintains more diverse beliefs")
        lines.append("  → Beliefs are more spread out in vector space")
    else:
        lines.append("  → Original system maintains more diverse beliefs")
    
    lines.append("\nResponse Uncertainty:")
    avg_orig_entropy = np.mean([original['response_entropy'][qid][-1]['entropy'] 
                                for qid in SURVEY_QUESTIONS.keys()])
    avg_vec_entropy = np.mean([vector['response_entropy'][qid][-1]['entropy'] 
                              for qid in SURVEY_QUESTIONS.keys()])
    if avg_vec_entropy > avg_orig_entropy:
        lines.append("  → Vector system shows higher uncertainty (more realistic)")
        lines.append("  → Responses are more probabilistic and diverse")
    else:
        lines.append("  → Original system shows higher uncertainty")
    
    lines.append("\nOpinion Clustering:")
    if vec_homophily > orig_homophily:
        lines.append("  → Vector system shows stronger homophily")
        lines.append("  → Beliefs cluster more along social connections")
    else:
        lines.append("  → Original system shows stronger homophily")
    
    lines.append("\nConvergence:")
    orig_converged = sum(1 for qid in SURVEY_QUESTIONS.keys() 
                        if original['convergence'][qid]['convergence_round'] is not None)
    vec_converged = sum(1 for qid in SURVEY_QUESTIONS.keys() 
                       if vector['convergence'][qid]['convergence_round'] is not None)
    lines.append(f"  Original: {orig_converged}/{len(SURVEY_QUESTIONS)} questions converged")
    lines.append(f"  Vector: {vec_converged}/{len(SURVEY_QUESTIONS)} questions converged")
    if vec_converged < orig_converged:
        lines.append("  → Vector system shows slower convergence (more dynamic)")
    else:
        lines.append("  → Original system shows slower convergence")
    
    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Main function."""
    print("=" * 80)
    print("SYSTEM COMPARISON REPORT GENERATOR")
    print("=" * 80)
    
    # Load metrics
    print("\nLoading metrics...")
    original, vector = load_metrics()
    
    # Generate report
    print("Generating comparison report...")
    report = generate_comparison_report(original, vector)
    
    # Save report
    output_dir = Path('metrics/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / 'system_comparison_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved comparison report to {report_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("REPORT SUMMARY")
    print("=" * 80)
    print(report[:2000])  # Print first part
    print("\n... (full report saved to file)")


if __name__ == '__main__':
    import numpy as np
    main()

