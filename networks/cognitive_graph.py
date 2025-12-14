"""
Cognitive Affinity Graph (Graph B).

Implements a fully connected graph where edge weights represent
embedding similarity between personas.
"""

import numpy as np
import networkx as nx
from typing import Optional, List, Tuple
from embeddings.similarity import cosine_similarity_matrix


def create_cognitive_graph(embeddings: np.ndarray,
                          similarity_threshold: float = 0.0,
                          directed: bool = False) -> nx.Graph:
    """
    Create a fully connected cognitive affinity graph.
    
    Edge weights represent similarity between persona embeddings.
    This graph is dynamic and updates as beliefs change.
    
    Args:
        embeddings: numpy array of shape (n_personas, embedding_dim)
        similarity_threshold: Minimum similarity to include edge (default: 0.0 = all edges)
        directed: Whether graph is directed (default: False, undirected)
        
    Returns:
        NetworkX Graph object with similarity weights
    """
    n = len(embeddings)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity_matrix(embeddings)
    
    # Create graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(range(n))
    
    # Add edges with similarity weights
    for i in range(n):
        for j in range(i + 1 if not directed else n):
            if i != j:  # No self-loops
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    G.add_edge(i, j, weight=similarity, similarity=similarity)
    
    # Add metadata
    G.graph['type'] = 'cognitive_affinity'
    G.graph['similarity_threshold'] = similarity_threshold
    G.graph['n_personas'] = n
    
    return G


def update_cognitive_graph(graph: nx.Graph,
                         new_embeddings: np.ndarray) -> nx.Graph:
    """
    Update cognitive graph with new embeddings.
    
    Recalculates all edge weights based on new embeddings.
    
    Args:
        graph: Existing cognitive graph
        new_embeddings: New embeddings array
        
    Returns:
        Updated graph
    """
    # Recreate graph with new embeddings
    similarity_threshold = graph.graph.get('similarity_threshold', 0.0)
    directed = isinstance(graph, nx.DiGraph)
    
    updated_graph = create_cognitive_graph(
        new_embeddings,
        similarity_threshold=similarity_threshold,
        directed=directed
    )
    
    return updated_graph


def get_cognitive_similarity(graph: nx.Graph, node1: int, node2: int) -> float:
    """
    Get cognitive similarity between two nodes.
    
    Args:
        graph: Cognitive affinity graph
        node1: First node index
        node2: Second node index
        
    Returns:
        Similarity score (0.0 if not connected)
    """
    if graph.has_edge(node1, node2):
        return graph[node1][node2].get('similarity', graph[node1][node2].get('weight', 0.0))
    return 0.0


def get_cognitive_neighbors(graph: nx.Graph,
                           node: int,
                           min_similarity: float = 0.0) -> List[tuple]:
    """
    Get cognitive neighbors of a node with their similarities.
    
    Args:
        graph: Cognitive affinity graph
        node: Node index
        min_similarity: Minimum similarity to include
        
    Returns:
        List of (neighbor_index, similarity) tuples
    """
    neighbors = []
    for neighbor in graph.neighbors(node):
        similarity = get_cognitive_similarity(graph, node, neighbor)
        if similarity >= min_similarity:
            neighbors.append((neighbor, similarity))
    
    # Sort by similarity (descending)
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return neighbors


def cognitive_graph_statistics(graph: nx.Graph) -> dict:
    """
    Calculate statistics about the cognitive graph.
    
    Args:
        graph: Cognitive affinity graph
        
    Returns:
        Dictionary with graph statistics
    """
    # Get all edge weights
    weights = [data.get('weight', data.get('similarity', 0.0)) 
               for _, _, data in graph.edges(data=True)]
    
    return {
        'n_nodes': graph.number_of_nodes(),
        'n_edges': graph.number_of_edges(),
        'avg_similarity': float(np.mean(weights)) if weights else 0.0,
        'min_similarity': float(np.min(weights)) if weights else 0.0,
        'max_similarity': float(np.max(weights)) if weights else 0.0,
        'std_similarity': float(np.std(weights)) if weights else 0.0,
        'density': nx.density(graph)
    }

