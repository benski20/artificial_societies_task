"""
Demo script for topology and embeddings system.

Demonstrates Phases 1-4:
1. Embedding generation
2. Social exposure graph
3. Cognitive affinity graph
4. Dual-graph model

Includes visualizations.
"""

import json
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from embeddings import (
    load_personas_and_generate_embeddings,
    cosine_similarity_matrix,
    similarity_statistics
)
from networks import (
    create_watts_strogatz_graph,
    create_cognitive_graph,
    create_dual_graph_model,
    visualize_social_graph,
    visualize_cognitive_graph,
    visualize_dual_graph,
    visualize_similarity_heatmap,
    visualize_graph_comparison
)


def main():
    """Run the demo."""
    print("=" * 80)
    print("TOPOLOGY + EMBEDDINGS DEMO")
    print("=" * 80)
    
    # Load personas and generate embeddings
    print("\nPhase 1: Generating Embeddings...")
    print("-" * 80)
    personas_file = Path('outputs/constraint_personas.json')
    
    if not personas_file.exists():
        print(f"Error: {personas_file} not found!")
        print("Please run the persona generation first:")
        print("  python3 main.py --method constraint --n 100 --augment --surveys")
        return
    
    print(f"Loading personas from {personas_file}...")
    personas, embeddings = load_personas_and_generate_embeddings(
        str(personas_file),
        model_name='all-MiniLM-L6-v2'
    )
    print(f"✓ Generated embeddings for {len(personas)} personas")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    # Calculate similarity statistics
    similarity_matrix = cosine_similarity_matrix(embeddings)
    sim_stats = similarity_statistics(similarity_matrix)
    print(f"\nSimilarity Statistics:")
    print(f"  Mean: {sim_stats['mean']:.3f}")
    print(f"  Median: {sim_stats['median']:.3f}")
    print(f"  Std: {sim_stats['std']:.3f}")
    print(f"  Range: [{sim_stats['min']:.3f}, {sim_stats['max']:.3f}]")
    
    # Create social graph
    print("\n\nPhase 2: Creating Social Exposure Graph (Graph A)...")
    print("-" * 80)
    social_graph = create_watts_strogatz_graph(
        n=len(personas),
        k=4,  # Each node connected to 4 neighbors
        p=0.2,  # 20% rewiring probability
        seed=42
    )
    print(f"✓ Created Watts-Strogatz graph")
    print(f"  Nodes: {social_graph.number_of_nodes()}")
    print(f"  Edges: {social_graph.number_of_edges()}")
    print(f"  Average degree: {sum(dict(social_graph.degree()).values()) / social_graph.number_of_nodes():.2f}")
    print(f"  Clustering coefficient: {nx.average_clustering(social_graph):.3f}")
    
    # Create cognitive graph
    print("\n\nPhase 3: Creating Cognitive Affinity Graph (Graph B)...")
    print("-" * 80)
    cognitive_graph = create_cognitive_graph(
        embeddings,
        similarity_threshold=0.0  # Include all edges
    )
    print(f"✓ Created fully connected cognitive graph")
    print(f"  Nodes: {cognitive_graph.number_of_nodes()}")
    print(f"  Edges: {cognitive_graph.number_of_edges()}")
    
    # Get edge weights
    weights = [data.get('weight', 0) for _, _, data in cognitive_graph.edges(data=True)]
    print(f"  Average similarity: {np.mean(weights):.3f}")
    print(f"  Min similarity: {np.min(weights):.3f}")
    print(f"  Max similarity: {np.max(weights):.3f}")
    
    # Create dual-graph model
    print("\n\nPhase 4: Creating Dual-Graph Model...")
    print("-" * 80)
    dual_model = create_dual_graph_model(
        n_personas=len(personas),
        embeddings=embeddings,
        social_k=4,
        social_p=0.2,
        cognitive_threshold=0.0,
        seed=42
    )
    print(f"✓ Created dual-graph model")
    
    stats = dual_model.statistics()
    print(f"\nDual-Graph Statistics:")
    print(f"  Social edges: {stats['social_edges']}")
    print(f"  Cognitive edges: {stats['cognitive_edges']}")
    print(f"  Active influences: {stats['active_influences']}")
    print(f"  Average influence weight: {stats['avg_influence_weight']:.3f}")
    print(f"  Max influence weight: {stats['max_influence_weight']:.3f}")
    
    # Show example: who can influence persona 0?
    print(f"\nExample: Who can influence Persona 0?")
    influencers = dual_model.get_influence_neighbors(0)
    print(f"  {len(influencers)} personas can influence Persona 0")
    if influencers:
        print(f"  Top 5 influencers:")
        for i, (neighbor_idx, weight) in enumerate(influencers[:5], 1):
            print(f"    {i}. Persona {neighbor_idx}: influence weight = {weight:.3f}")
    
    # Create visualizations
    print("\n\nCreating Visualizations...")
    print("-" * 80)
    
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Similarity heatmap
    print("1. Generating similarity heatmap...")
    fig = visualize_similarity_heatmap(embeddings)
    fig.savefig(output_dir / 'similarity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved to {output_dir / 'similarity_heatmap.png'}")
    
    # 2. Social graph
    print("2. Generating social graph visualization...")
    fig = visualize_social_graph(social_graph)
    fig.savefig(output_dir / 'social_graph.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved to {output_dir / 'social_graph.png'}")
    
    # 3. Cognitive graph (simplified for readability)
    print("3. Generating cognitive graph visualization...")
    fig = visualize_cognitive_graph(cognitive_graph, edge_threshold=0.5)
    fig.savefig(output_dir / 'cognitive_graph.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved to {output_dir / 'cognitive_graph.png'}")
    
    # 4. Dual-graph model
    print("4. Generating dual-graph visualization...")
    fig = visualize_dual_graph(dual_model)
    fig.savefig(output_dir / 'dual_graph.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved to {output_dir / 'dual_graph.png'}")
    
    # 5. Comparison
    print("5. Generating graph comparison...")
    fig = visualize_graph_comparison(dual_model)
    fig.savefig(output_dir / 'graph_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✓ Saved to {output_dir / 'graph_comparison.png'}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nKey Insights:")
    print(f"  • {stats['active_influences']} active influence connections")
    print(f"    (out of {stats['social_edges']} possible social connections)")
    print(f"  • {stats['active_influences']} active influences from {stats['social_edges']} social edges")
    print(f"  • Each social edge creates 2 influence connections (bidirectional)")
    print(f"  • Only connections with sufficient cognitive similarity are active")
    print(f"    connections have sufficient cognitive similarity for influence")
    print(f"  • This creates natural echo chambers and selective influence")


if __name__ == '__main__':
    import networkx as nx
    main()

