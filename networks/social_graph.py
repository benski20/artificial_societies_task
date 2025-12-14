"""
Social Exposure Graph (Graph A).

Implements a Watts-Strogatz small-world network representing
who talks to whom in the social network.
"""

import numpy as np
import networkx as nx
from typing import Optional, Tuple, List


def create_watts_strogatz_graph(n: int,
                               k: int = 4,
                               p: float = 0.2,
                               seed: Optional[int] = None) -> nx.Graph:
    """
    Create a Watts-Strogatz small-world graph.
    
    This represents the social exposure network - who talks to whom.
    The graph is fixed and doesn't change during simulation.
    
    Args:
        n: Number of nodes (personas)
        k: Each node is connected to k nearest neighbors in ring topology
        p: Rewiring probability (0 = ring, 1 = random)
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX Graph object
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    
    # Add metadata
    G.graph['type'] = 'social_exposure'
    G.graph['model'] = 'watts_strogatz'
    G.graph['k'] = k
    G.graph['p'] = p
    
    return G


def get_social_neighbors(graph: nx.Graph, node: int) -> List[int]:
    """
    Get social neighbors of a node.
    
    Args:
        graph: Social exposure graph
        node: Node index
        
    Returns:
        List of neighbor node indices
    """
    return list(graph.neighbors(node))


def are_socially_connected(graph: nx.Graph, node1: int, node2: int) -> bool:
    """
    Check if two nodes are socially connected.
    
    Args:
        graph: Social exposure graph
        node1: First node index
        node2: Second node index
        
    Returns:
        True if connected, False otherwise
    """
    return graph.has_edge(node1, node2)


def get_social_connection_matrix(graph: nx.Graph) -> np.ndarray:
    """
    Get adjacency matrix of social graph.
    
    Args:
        graph: Social exposure graph
        
    Returns:
        Binary adjacency matrix (n x n)
    """
    return nx.adjacency_matrix(graph).toarray()


def social_graph_statistics(graph: nx.Graph) -> dict:
    """
    Calculate statistics about the social graph.
    
    Args:
        graph: Social exposure graph
        
    Returns:
        Dictionary with graph statistics
    """
    return {
        'n_nodes': graph.number_of_nodes(),
        'n_edges': graph.number_of_edges(),
        'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
        'clustering_coefficient': nx.average_clustering(graph),
        'avg_path_length': nx.average_shortest_path_length(graph),
        'density': nx.density(graph),
        'is_connected': nx.is_connected(graph)
    }

