"""
Calculate comprehensive metrics for both systems.

Metrics include:
1. Belief space diversity (embedding spread)
2. Aggregate response distributions
3. Opinion clustering
4. Response entropy
5. Influence network metrics
6. Convergence metrics
"""

import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from persona.persona import Persona
from experiments.vector_beliefs.persona.vector_persona import VectorPersona
from surveys.questions import SURVEY_QUESTIONS
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score


def load_personas(filepath: str, system_type: str = 'original') -> List:
    """Load personas from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if system_type == 'vector':
        return [VectorPersona.from_dict(p_dict) for p_dict in data]
    else:
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
                first_gen_college=p_dict.get('first_gen_college', False),
                gpa=p_dict.get('gpa', 0.0),
                sports_participation=p_dict.get('sports_participation', False),
                mental_health=p_dict.get('mental_health', 'Fair'),
                social_media_intensity=p_dict.get('social_media_intensity', 'Moderate'),
                college_intention=p_dict.get('college_intention', 'Unsure'),
                narrative=p_dict.get('narrative'),
                beliefs=p_dict.get('beliefs', {}),
                survey_responses=p_dict.get('survey_responses', {})
            )
            personas.append(persona)
        return personas


def calculate_embedding_spread(embeddings: np.ndarray) -> Dict:
    """
    Calculate diversity/spread of embeddings in belief space.
    
    Metrics:
    - Mean pairwise distance
    - Variance of distances
    - Spread (max - min distance)
    - Dimension-wise variance
    """
    # Pairwise distances
    distances = pdist(embeddings, metric='euclidean')
    
    # Distance statistics
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    
    # Dimension-wise variance
    dim_variances = np.var(embeddings, axis=0)
    
    # Principal component analysis (first 2 PCs explain how much variance)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(10, embeddings.shape[1]))
    pca.fit(embeddings)
    
    return {
        'mean_pairwise_distance': float(mean_distance),
        'std_pairwise_distance': float(std_distance),
        'min_distance': float(min_distance),
        'max_distance': float(max_distance),
        'distance_range': float(max_distance - min_distance),
        'dimension_variances': dim_variances.tolist(),
        'mean_dimension_variance': float(np.mean(dim_variances)),
        'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
        'pca_total_variance_explained': float(np.sum(pca.explained_variance_ratio_[:2]))
    }


def calculate_belief_vector_diversity(personas: List[VectorPersona]) -> Dict:
    """Calculate diversity of belief vectors."""
    vectors = np.array([p.get_belief_vector() for p in personas])
    return calculate_embedding_spread(vectors)


def calculate_response_entropy(responses: List[str]) -> float:
    """
    Calculate Shannon entropy of response distribution.
    
    Higher entropy = more uncertainty/diversity
    Lower entropy = more certainty/concentration
    """
    if not responses:
        return 0.0
    
    counts = Counter(responses)
    total = len(responses)
    probabilities = [count / total for count in counts.values()]
    
    # Shannon entropy: -sum(p * log2(p))
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    return float(entropy)


def calculate_response_entropy_over_time(survey_evolution: Dict) -> Dict:
    """Calculate entropy for each question over all rounds."""
    entropies = {}
    
    for question_id, evolution in survey_evolution.items():
        entropies[question_id] = []
        for round_data in evolution:
            distribution = round_data['distribution']
            responses = []
            for option, count in distribution.items():
                responses.extend([option] * count)
            
            entropy = calculate_response_entropy(responses)
            entropies[question_id].append({
                'round': round_data['round'],
                'entropy': entropy,
                'max_entropy': np.log2(len(distribution))  # Max possible entropy
            })
    
    return entropies


def calculate_opinion_clustering(personas: List,
                                 embeddings: np.ndarray,
                                 dual_model) -> Dict:
    """
    Analyze if beliefs cluster along the network.
    
    Measures:
    - Do socially connected people have similar beliefs?
    - Clustering coefficient
    - Homophily (similarity between connected nodes)
    """
    # Get social connections
    social_edges = list(dual_model.social_graph.edges())
    
    # Calculate similarity between connected nodes
    connected_similarities = []
    unconnected_similarities = []
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-8)
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T
    
    # Get connected pairs
    connected_pairs = set()
    for edge in social_edges:
        i, j = edge
        connected_pairs.add((min(i, j), max(i, j)))
        similarity = similarity_matrix[i, j]
        connected_similarities.append(float(similarity))
    
    # Get unconnected pairs (sample)
    n_personas = len(personas)
    all_pairs = set()
    for i in range(n_personas):
        for j in range(i + 1, n_personas):
            all_pairs.add((i, j))
    
    unconnected_pairs = all_pairs - connected_pairs
    # Sample to avoid too many comparisons
    sample_size = min(len(unconnected_pairs), len(connected_similarities) * 2)
    sampled_unconnected = np.random.choice(len(unconnected_pairs), sample_size, replace=False)
    
    for idx in sampled_unconnected:
        i, j = list(unconnected_pairs)[idx]
        similarity = similarity_matrix[i, j]
        unconnected_similarities.append(float(similarity))
    
    # Clustering analysis
    # Use hierarchical clustering to find belief clusters
    distance_matrix = squareform(pdist(embeddings, metric='euclidean'))
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    # Try different numbers of clusters
    cluster_metrics = {}
    for n_clusters in [3, 5, 10]:
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        silhouette = silhouette_score(embeddings, clusters)
        cluster_metrics[f'{n_clusters}_clusters'] = {
            'silhouette_score': float(silhouette),
            'cluster_sizes': [int(np.sum(clusters == i)) for i in range(1, n_clusters + 1)]
        }
    
    return {
        'connected_similarity': {
            'mean': float(np.mean(connected_similarities)),
            'std': float(np.std(connected_similarities)),
            'median': float(np.median(connected_similarities))
        },
        'unconnected_similarity': {
            'mean': float(np.mean(unconnected_similarities)),
            'std': float(np.std(unconnected_similarities)),
            'median': float(np.median(unconnected_similarities))
        },
        'homophily_effect': float(np.mean(connected_similarities) - np.mean(unconnected_similarities)),
        'clustering_analysis': cluster_metrics
    }


def calculate_convergence_metrics(survey_evolution: Dict) -> Dict:
    """
    Calculate convergence and stability metrics.
    
    Measures:
    - How quickly responses stabilize
    - Polarization (variance in responses)
    - Stability (change between consecutive rounds)
    """
    convergence = {}
    
    for question_id, evolution in survey_evolution.items():
        if len(evolution) < 2:
            continue
        
        # Calculate stability (change between rounds)
        stability_scores = []
        for i in range(1, len(evolution)):
            prev_dist = evolution[i-1]['distribution']
            curr_dist = evolution[i]['distribution']
            
            # Total variation distance
            all_options = set(list(prev_dist.keys()) + list(curr_dist.keys()))
            total_prev = sum(prev_dist.values())
            total_curr = sum(curr_dist.values())
            
            tv_distance = 0.0
            for opt in all_options:
                prev_pct = prev_dist.get(opt, 0) / total_prev if total_prev > 0 else 0
                curr_pct = curr_dist.get(opt, 0) / total_curr if total_curr > 0 else 0
                tv_distance += abs(prev_pct - curr_pct)
            
            stability_scores.append(1.0 - tv_distance / 2.0)  # Normalize to [0, 1]
        
        # Calculate convergence round (when stability reaches threshold)
        convergence_round = None
        for i, stability in enumerate(stability_scores):
            if stability > 0.95:  # 95% stable
                convergence_round = i + 1
                break
        
        # Calculate polarization (variance in response distribution)
        final_dist = evolution[-1]['distribution']
        total = sum(final_dist.values())
        probabilities = [count / total for count in final_dist.values() if total > 0]
        polarization = np.var(probabilities) if probabilities else 0.0
        
        convergence[question_id] = {
            'stability_scores': stability_scores,
            'mean_stability': float(np.mean(stability_scores)) if stability_scores else 0.0,
            'convergence_round': convergence_round,
            'final_polarization': float(polarization),
            'total_change': float(1.0 - stability_scores[-1]) if stability_scores else 0.0
        }
    
    return convergence


def calculate_influence_network_metrics(dual_model) -> Dict:
    """Calculate network structure metrics."""
    import networkx as nx
    
    # Influence matrix
    influence_matrix = dual_model.get_influence_matrix()
    
    # Create influence graph
    G = nx.DiGraph()
    n = len(influence_matrix)
    for i in range(n):
        for j in range(n):
            if influence_matrix[i, j] > 0:
                G.add_edge(i, j, weight=influence_matrix[i, j])
    
    # Centrality measures
    in_degree_centrality = dict(nx.in_degree_centrality(G))
    out_degree_centrality = dict(nx.out_degree_centrality(G))
    
    # Weighted centrality
    in_strength = {node: sum(data['weight'] for _, _, data in G.in_edges(node, data=True)) 
                   for node in G.nodes()}
    out_strength = {node: sum(data['weight'] for _, _, data in G.out_edges(node, data=True)) 
                    for node in G.nodes()}
    
    return {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'in_degree_centrality': {
            'mean': float(np.mean(list(in_degree_centrality.values()))),
            'std': float(np.std(list(in_degree_centrality.values()))),
            'max': float(np.max(list(in_degree_centrality.values()))),
            'min': float(np.min(list(in_degree_centrality.values())))
        },
        'out_degree_centrality': {
            'mean': float(np.mean(list(out_degree_centrality.values()))),
            'std': float(np.std(list(out_degree_centrality.values()))),
            'max': float(np.max(list(out_degree_centrality.values()))),
            'min': float(np.min(list(out_degree_centrality.values())))
        },
        'in_strength': {
            'mean': float(np.mean(list(in_strength.values()))),
            'std': float(np.std(list(in_strength.values()))),
            'max': float(np.max(list(in_strength.values()))),
            'min': float(np.min(list(in_strength.values())))
        },
        'out_strength': {
            'mean': float(np.mean(list(out_strength.values()))),
            'std': float(np.std(list(out_strength.values()))),
            'max': float(np.max(list(out_strength.values()))),
            'min': float(np.min(list(out_strength.values())))
        }
    }


def calculate_all_metrics(system_type: str = 'original') -> Dict:
    """
    Calculate all metrics for a system.
    
    Args:
        system_type: 'original' or 'vector'
        
    Returns:
        Dictionary with all metrics
    """
    print(f"\nCalculating metrics for {system_type} system...")
    
    # Load data
    if system_type == 'vector':
        personas_file = Path('outputs/vector_beliefs_simulation/final_personas.json')
        survey_file = Path('outputs/vector_beliefs_simulation/survey_evolution.json')
    else:
        personas_file = Path('outputs/simulation/final_personas.json')
        survey_file = Path('outputs/simulation/multi_round_survey_evolution.json')
    
    print(f"  Loading personas...")
    personas = load_personas(str(personas_file), system_type)
    
    print(f"  Loading survey evolution...")
    with open(survey_file) as f:
        survey_evolution = json.load(f)
    
    # Generate embeddings
    print(f"  Generating embeddings...")
    if system_type == 'vector':
        from experiments.vector_beliefs.embeddings.belief_embeddings import generate_embeddings_from_vectors
        embeddings = generate_embeddings_from_vectors(personas, embedding_dim=384)
    else:
        from embeddings.generator import generate_embeddings
        embeddings = generate_embeddings(personas, show_progress=False)
    
    # Reconstruct dual-graph
    print(f"  Reconstructing dual-graph...")
    from networks.dual_graph import create_dual_graph_model
    dual_model = create_dual_graph_model(
        len(personas), embeddings,
        influence_reduction_factor=0.15, seed=42
    )
    
    # Calculate metrics
    print(f"  Calculating embedding spread...")
    embedding_metrics = calculate_embedding_spread(embeddings)
    
    if system_type == 'vector':
        print(f"  Calculating belief vector diversity...")
        belief_diversity = calculate_belief_vector_diversity(personas)
    else:
        belief_diversity = None
    
    print(f"  Calculating response entropy...")
    entropy_metrics = calculate_response_entropy_over_time(survey_evolution)
    
    print(f"  Calculating opinion clustering...")
    clustering_metrics = calculate_opinion_clustering(personas, embeddings, dual_model)
    
    print(f"  Calculating convergence metrics...")
    convergence_metrics = calculate_convergence_metrics(survey_evolution)
    
    print(f"  Calculating network metrics...")
    network_metrics = calculate_influence_network_metrics(dual_model)
    
    # Aggregate response distributions
    print(f"  Calculating aggregate distributions...")
    aggregate_distributions = {}
    for question_id, evolution in survey_evolution.items():
        aggregate_distributions[question_id] = {
            'round_0': evolution[0]['distribution'],
            'round_final': evolution[-1]['distribution'],
            'all_rounds': [r['distribution'] for r in evolution]
        }
    
    return {
        'system_type': system_type,
        'n_personas': len(personas),
        'embedding_spread': embedding_metrics,
        'belief_vector_diversity': belief_diversity,
        'response_entropy': entropy_metrics,
        'opinion_clustering': clustering_metrics,
        'convergence': convergence_metrics,
        'network_metrics': network_metrics,
        'aggregate_distributions': aggregate_distributions
    }


def main():
    """Main function."""
    print("=" * 80)
    print("COMPREHENSIVE METRICS CALCULATION")
    print("=" * 80)
    
    # Calculate for both systems
    print("\n" + "=" * 80)
    print("ORIGINAL SYSTEM")
    print("=" * 80)
    original_metrics = calculate_all_metrics('original')
    
    print("\n" + "=" * 80)
    print("VECTOR SYSTEM")
    print("=" * 80)
    vector_metrics = calculate_all_metrics('vector')
    
    # Save metrics
    output_dir = Path('metrics/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving metrics...")
    with open(output_dir / 'original_metrics.json', 'w') as f:
        json.dump(original_metrics, f, indent=2)
    
    with open(output_dir / 'vector_metrics.json', 'w') as f:
        json.dump(vector_metrics, f, indent=2)
    
    print(f"✓ Saved metrics to {output_dir}/")
    
    # Print summary
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    
    print(f"\nEmbedding Spread:")
    print(f"  Original: mean distance = {original_metrics['embedding_spread']['mean_pairwise_distance']:.4f}")
    print(f"  Vector: mean distance = {vector_metrics['embedding_spread']['mean_pairwise_distance']:.4f}")
    
    print(f"\nResponse Entropy (Final Round):")
    for qid in SURVEY_QUESTIONS.keys():
        orig_entropy = original_metrics['response_entropy'][qid][-1]['entropy']
        vec_entropy = vector_metrics['response_entropy'][qid][-1]['entropy']
        print(f"  {qid}:")
        print(f"    Original: {orig_entropy:.4f}")
        print(f"    Vector: {vec_entropy:.4f}")
    
    print(f"\nOpinion Clustering (Homophily):")
    print(f"  Original: {original_metrics['opinion_clustering']['homophily_effect']:.4f}")
    print(f"  Vector: {vector_metrics['opinion_clustering']['homophily_effect']:.4f}")
    
    print(f"\nConvergence:")
    for qid in SURVEY_QUESTIONS.keys():
        orig_conv = original_metrics['convergence'][qid]['convergence_round']
        vec_conv = vector_metrics['convergence'][qid]['convergence_round']
        print(f"  {qid}:")
        print(f"    Original: Round {orig_conv if orig_conv else 'N/A'}")
        print(f"    Vector: Round {vec_conv if vec_conv else 'N/A'}")


if __name__ == '__main__':
    main()

