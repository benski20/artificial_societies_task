"""
Dual-Graph Model: Combining Social Exposure and Cognitive Affinity.

Implements the core concept: Influence occurs only if both graphs align.
- Graph A (Social): Who talks to whom (Watts-Strogatz)
- Graph B (Cognitive): Similarity-based connections (fully connected)
"""

import numpy as np
import networkx as nx
from typing import Optional, Tuple, List
from .social_graph import create_watts_strogatz_graph, get_social_connection_matrix
from .cognitive_graph import create_cognitive_graph, get_cognitive_similarity


class DualGraphModel:
    """
    Dual-graph model combining social exposure and cognitive affinity.
    
    Influence weight between two personas is:
    influence(i→j) = social_connected(i,j) × cognitive_similarity(i,j) × reduction_factor
    
    This means influence only occurs when:
    1. They are socially connected (Graph A)
    2. They have cognitive similarity (Graph B)
    3. Influence is reduced by factor to simulate real-world resistance
    """
    
    def __init__(self,
                 social_graph: nx.Graph,
                 cognitive_graph: nx.Graph,
                 influence_reduction_factor: float = 0.15):
        """
        Initialize dual-graph model.
        
        Args:
            social_graph: Social exposure graph (Graph A)
            cognitive_graph: Cognitive affinity graph (Graph B)
            influence_reduction_factor: Factor to reduce influence weights (0.0-1.0)
                                      Lower values = less influence (more realistic)
        """
        self.social_graph = social_graph
        self.cognitive_graph = cognitive_graph
        self.influence_reduction_factor = influence_reduction_factor
        
        # Validate graphs have same number of nodes
        if social_graph.number_of_nodes() != cognitive_graph.number_of_nodes():
            raise ValueError(
                f"Graphs must have same number of nodes: "
                f"social={social_graph.number_of_nodes()}, "
                f"cognitive={cognitive_graph.number_of_nodes()}"
            )
        
        self.n_personas = social_graph.number_of_nodes()
        
        # Precompute combined influence matrix
        self._influence_matrix = None
        self._update_influence_matrix()
    
    def _update_influence_matrix(self):
        """Update the combined influence matrix."""
        # Social connections (binary)
        social_matrix = get_social_connection_matrix(self.social_graph)
        
        # Cognitive similarities
        cognitive_matrix = np.zeros((self.n_personas, self.n_personas))
        for i in range(self.n_personas):
            for j in range(self.n_personas):
                if i != j:
                    cognitive_matrix[i, j] = get_cognitive_similarity(
                        self.cognitive_graph, i, j
                    )
        
        # Combined: element-wise multiplication
        raw_influence = social_matrix * cognitive_matrix
        
        # Apply reduction factor to simulate real-world resistance to influence
        self._influence_matrix = raw_influence * self.influence_reduction_factor
    
    def get_influence_weight(self, source: int, target: int) -> float:
        """
        Get influence weight from source to target.
        
        Args:
            source: Source persona index
            target: Target persona index
            
        Returns:
            Influence weight (0.0 if not connected in both graphs)
        """
        return float(self._influence_matrix[source, target])
    
    def get_influence_neighbors(self, node: int) -> List[tuple]:
        """
        Get neighbors that can influence this node (connected in both graphs).
        
        Args:
            node: Node index
            
        Returns:
            List of (neighbor_index, influence_weight) tuples
        """
        neighbors = []
        for neighbor in range(self.n_personas):
            if neighbor != node:
                weight = self.get_influence_weight(neighbor, node)
                if weight > 0:
                    neighbors.append((neighbor, weight))
        
        # Sort by influence weight (descending)
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors
    
    def get_influence_matrix(self) -> np.ndarray:
        """
        Get the full influence matrix.
        
        Returns:
            numpy array of shape (n, n) with influence weights
        """
        return self._influence_matrix.copy()
    
    def update_cognitive_graph(self, new_embeddings: np.ndarray):
        """
        Update cognitive graph with new embeddings and recalculate influence.
        
        Args:
            new_embeddings: New embeddings array
        """
        from .cognitive_graph import update_cognitive_graph
        self.cognitive_graph = update_cognitive_graph(
            self.cognitive_graph, new_embeddings
        )
        self._update_influence_matrix()
    
    def statistics(self) -> dict:
        """
        Get statistics about the dual-graph model.
        
        Returns:
            Dictionary with statistics
        """
        influence_weights = self._influence_matrix[
            self._influence_matrix > 0
        ]
        
        return {
            'n_personas': self.n_personas,
            'social_edges': self.social_graph.number_of_edges(),
            'cognitive_edges': self.cognitive_graph.number_of_edges(),
            'active_influences': len(influence_weights),
            'avg_influence_weight': float(np.mean(influence_weights)) if len(influence_weights) > 0 else 0.0,
            'max_influence_weight': float(np.max(influence_weights)) if len(influence_weights) > 0 else 0.0,
            'social_density': nx.density(self.social_graph),
            'cognitive_density': nx.density(self.cognitive_graph)
        }


def create_dual_graph_model(n_personas: int,
                           embeddings: np.ndarray,
                           social_k: int = 4,
                           social_p: float = 0.2,
                           cognitive_threshold: float = 0.0,
                           influence_reduction_factor: float = 0.15,
                           seed: Optional[int] = None) -> DualGraphModel:
    """
    Create a dual-graph model from scratch.
    
    Args:
        n_personas: Number of personas
        embeddings: Embeddings array
        social_k: Watts-Strogatz k parameter
        social_p: Watts-Strogatz rewiring probability
        cognitive_threshold: Minimum similarity for cognitive edges
        seed: Random seed
        
    Returns:
        DualGraphModel instance
    """
    # Create social graph
    social_graph = create_watts_strogatz_graph(
        n_personas, k=social_k, p=social_p, seed=seed
    )
    
    # Create cognitive graph
    cognitive_graph = create_cognitive_graph(
        embeddings, similarity_threshold=cognitive_threshold
    )
    
    # Create dual-graph model
    model = DualGraphModel(social_graph, cognitive_graph, 
                          influence_reduction_factor=influence_reduction_factor)
    
    return model

