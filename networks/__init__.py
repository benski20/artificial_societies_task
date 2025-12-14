"""
Network models for social influence and dynamics.

Implements dual-graph model combining social exposure and cognitive affinity.
"""

from .social_graph import (
    create_watts_strogatz_graph,
    get_social_neighbors,
    are_socially_connected,
    get_social_connection_matrix,
    social_graph_statistics
)
from .cognitive_graph import (
    create_cognitive_graph,
    update_cognitive_graph,
    get_cognitive_similarity,
    get_cognitive_neighbors,
    cognitive_graph_statistics
)
from .dual_graph import (
    DualGraphModel,
    create_dual_graph_model
)
from .visualization import (
    visualize_social_graph,
    visualize_cognitive_graph,
    visualize_dual_graph,
    visualize_similarity_heatmap,
    visualize_graph_comparison
)
from .belief_updates import (
    update_persona_beliefs,
    update_all_beliefs,
    calculate_susceptibility
)
from .simulation import (
    run_simulation,
    SimulationResults,
    save_simulation_results,
    print_simulation_summary
)

__all__ = [
    'create_watts_strogatz_graph',
    'get_social_neighbors',
    'are_socially_connected',
    'get_social_connection_matrix',
    'social_graph_statistics',
    'create_cognitive_graph',
    'update_cognitive_graph',
    'get_cognitive_similarity',
    'get_cognitive_neighbors',
    'cognitive_graph_statistics',
    'DualGraphModel',
    'create_dual_graph_model',
    'visualize_social_graph',
    'visualize_cognitive_graph',
    'visualize_dual_graph',
    'visualize_similarity_heatmap',
    'visualize_graph_comparison',
    'update_persona_beliefs',
    'update_all_beliefs',
    'calculate_susceptibility',
    'run_simulation',
    'SimulationResults',
    'save_simulation_results',
    'print_simulation_summary'
]

