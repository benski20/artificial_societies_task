"""
Network visualization functions.

Provides functions to visualize social graphs, cognitive graphs,
and the dual-graph model.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def visualize_social_graph(graph: nx.Graph,
                          title: str = "Social Exposure Graph (Graph A)",
                          node_size: int = 300,
                          figsize: Tuple[int, int] = (12, 8),
                          seed: Optional[int] = 42) -> plt.Figure:
    """
    Visualize the social exposure graph.
    
    Args:
        graph: Social exposure graph
        title: Plot title
        node_size: Size of nodes
        figsize: Figure size
        seed: Random seed for layout
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(graph, seed=seed)
    
    # Draw graph
    nx.draw_networkx_nodes(graph, pos, node_size=node_size,
                          node_color='lightblue', ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_cognitive_graph(graph: nx.Graph,
                             title: str = "Cognitive Affinity Graph (Graph B)",
                             node_size: int = 300,
                             figsize: Tuple[int, int] = (12, 8),
                             seed: Optional[int] = 42,
                             edge_threshold: float = 0.3) -> plt.Figure:
    """
    Visualize the cognitive affinity graph.
    
    Colors edges by similarity weight.
    
    Args:
        graph: Cognitive affinity graph
        title: Plot title
        node_size: Size of nodes
        figsize: Figure size
        seed: Random seed for layout
        edge_threshold: Only show edges above this similarity
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use spring layout
    pos = nx.spring_layout(graph, seed=seed)
    
    # Separate edges by weight
    high_weight_edges = [(u, v) for u, v, d in graph.edges(data=True)
                         if d.get('weight', 0) >= edge_threshold]
    low_weight_edges = [(u, v) for u, v, d in graph.edges(data=True)
                        if d.get('weight', 0) < edge_threshold]
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=node_size,
                          node_color='lightcoral', ax=ax)
    
    # Draw edges with different colors/widths
    if high_weight_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=high_weight_edges,
                              width=2, alpha=0.6, edge_color='darkred', ax=ax)
    if low_weight_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=low_weight_edges,
                              width=0.5, alpha=0.2, edge_color='lightgray', ax=ax)
    
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_dual_graph(dual_model,
                        title: str = "Dual-Graph Model: Active Influences",
                        node_size: int = 300,
                        figsize: Tuple[int, int] = (12, 8),
                        seed: Optional[int] = 42) -> plt.Figure:
    """
    Visualize the dual-graph model showing active influences.
    
    Only shows edges that exist in both graphs (active influences).
    
    Args:
        dual_model: DualGraphModel instance
        title: Plot title
        node_size: Size of nodes
        figsize: Figure size
        seed: Random seed for layout
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a combined graph with only active influences
    active_graph = nx.Graph()
    active_graph.add_nodes_from(range(dual_model.n_personas))
    
    # Add edges that exist in both graphs
    influence_matrix = dual_model.get_influence_matrix()
    for i in range(dual_model.n_personas):
        for j in range(i + 1, dual_model.n_personas):
            if influence_matrix[i, j] > 0:
                active_graph.add_edge(i, j, weight=influence_matrix[i, j])
    
    # Use spring layout
    pos = nx.spring_layout(active_graph, seed=seed)
    
    # Get edge weights for coloring
    edges = active_graph.edges()
    weights = [active_graph[u][v]['weight'] for u, v in edges]
    
    # Draw nodes
    nx.draw_networkx_nodes(active_graph, pos, node_size=node_size,
                          node_color='lightgreen', ax=ax)
    
    # Draw edges colored by weight
    nx.draw_networkx_edges(active_graph, pos, width=2,
                          edge_color=weights, edge_cmap=plt.cm.Reds,
                          alpha=0.6, ax=ax)
    
    nx.draw_networkx_labels(active_graph, pos, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_similarity_heatmap(embeddings: np.ndarray,
                                 title: str = "Cognitive Similarity Heatmap",
                                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualize similarity matrix as a heatmap.
    
    Args:
        embeddings: Embeddings array
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    from embeddings.similarity import cosine_similarity_matrix
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity_matrix(embeddings)
    
    # Create heatmap
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto',
                   vmin=0, vmax=1)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Persona Index', fontsize=12)
    ax.set_ylabel('Persona Index', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity', fontsize=10)
    
    plt.tight_layout()
    return fig


def visualize_graph_comparison(dual_model,
                              figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
    """
    Create side-by-side comparison of social and cognitive graphs.
    
    Args:
        dual_model: DualGraphModel instance
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Social graph
    pos = nx.spring_layout(dual_model.social_graph, seed=42)
    nx.draw_networkx_nodes(dual_model.social_graph, pos, node_size=200,
                          node_color='lightblue', ax=ax1)
    nx.draw_networkx_edges(dual_model.social_graph, pos, alpha=0.5, ax=ax1)
    ax1.set_title('Social Exposure Graph\n(Graph A)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Cognitive graph (simplified - only high similarity edges)
    cognitive_simple = nx.Graph()
    cognitive_simple.add_nodes_from(range(dual_model.n_personas))
    for i in range(dual_model.n_personas):
        for j in range(i + 1, dual_model.n_personas):
            sim = dual_model.cognitive_graph[i][j].get('weight', 0)
            if sim > 0.5:  # Only show high similarity
                cognitive_simple.add_edge(i, j, weight=sim)
    
    pos2 = nx.spring_layout(cognitive_simple, seed=42)
    nx.draw_networkx_nodes(cognitive_simple, pos2, node_size=200,
                          node_color='lightcoral', ax=ax2)
    edges = cognitive_simple.edges()
    weights = [cognitive_simple[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(cognitive_simple, pos2, width=2,
                          edge_color=weights, edge_cmap=plt.cm.Reds,
                          alpha=0.6, ax=ax2)
    ax2.set_title('Cognitive Affinity Graph\n(Graph B - High Similarity)', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

