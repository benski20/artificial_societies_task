# Topology + Embeddings Implementation Guide

This document explains the topology and embeddings system (Phases 1-4) for modeling social influence networks.

## Overview

The system implements a **dual-graph model** that combines:
1. **Social Exposure Graph (Graph A)**: Who talks to whom (Watts-Strogatz small-world)
2. **Cognitive Affinity Graph (Graph B)**: Similarity-based connections (fully connected)

**Key Principle**: Influence occurs only when both graphs align.

```
Influence(i→j) = Social_Connected(i,j) × Cognitive_Similarity(i,j)
```

## Phase 1: Embedding Generation

### Purpose
Convert persona narratives and beliefs into numerical vectors (embeddings) that capture semantic meaning.

### Implementation
- **File**: `embeddings/generator.py`
- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Input**: Persona narrative + all beliefs
- **Output**: numpy array of shape (n_personas, 384)

### Usage
```python
from embeddings import load_personas_and_generate_embeddings

personas, embeddings = load_personas_and_generate_embeddings(
    'outputs/constraint_personas.json'
)
```

### Key Functions
- `generate_embeddings()`: Generate embeddings for persona list
- `generate_persona_text()`: Combine narrative + beliefs into text
- `load_personas_and_generate_embeddings()`: Load from JSON and generate

## Phase 2: Social Exposure Graph (Graph A)

### Purpose
Model who talks to whom in the social network using a small-world topology.

### Implementation
- **File**: `networks/social_graph.py`
- **Model**: Watts-Strogatz small-world network
- **Parameters**:
  - `k`: Each node connected to k nearest neighbors (default: 4)
  - `p`: Rewiring probability (default: 0.2)
- **Properties**: Fixed structure, undirected, binary edges

### Why Watts-Strogatz?
- Captures local clustering (friends of friends are friends)
- Includes long-range connections (realistic for high school networks)
- Small average path length (information spreads quickly)

### Usage
```python
from networks import create_watts_strogatz_graph

social_graph = create_watts_strogatz_graph(
    n=100,  # 100 personas
    k=4,    # 4 neighbors each
    p=0.2,  # 20% rewiring
    seed=42
)
```

### Key Functions
- `create_watts_strogatz_graph()`: Create social graph
- `are_socially_connected()`: Check if two personas are connected
- `get_social_neighbors()`: Get neighbors of a persona
- `social_graph_statistics()`: Get graph metrics

## Phase 3: Cognitive Affinity Graph (Graph B)

### Purpose
Model cognitive similarity between personas based on their beliefs and narratives.

### Implementation
- **File**: `networks/cognitive_graph.py`
- **Model**: Fully connected graph with similarity weights
- **Edge Weights**: Cosine similarity between embeddings (0 to 1)
- **Properties**: Dynamic (updates as beliefs change), weighted edges

### Why Fully Connected?
- Allows influence from any similar persona, not just neighbors
- Captures "echo chambers" where similar beliefs cluster
- Enables modeling of indirect influence

### Usage
```python
from networks import create_cognitive_graph

cognitive_graph = create_cognitive_graph(
    embeddings,
    similarity_threshold=0.0  # Include all edges
)
```

### Key Functions
- `create_cognitive_graph()`: Create cognitive graph from embeddings
- `get_cognitive_similarity()`: Get similarity between two personas
- `update_cognitive_graph()`: Update graph with new embeddings
- `cognitive_graph_statistics()`: Get graph metrics

## Phase 4: Dual-Graph Model

### Purpose
Combine social exposure and cognitive affinity to determine actual influence.

### Implementation
- **File**: `networks/dual_graph.py`
- **Class**: `DualGraphModel`
- **Influence Formula**: `influence(i→j) = social_connected(i,j) × cognitive_similarity(i,j)`

### Key Insight
- **Exposure without persuasion**: Socially connected but low similarity → no influence
- **Echo chambers**: High similarity but not directly connected → potential indirect influence
- **Active influence**: Both connected AND similar → strong influence

### Usage
```python
from networks import create_dual_graph_model

dual_model = create_dual_graph_model(
    n_personas=100,
    embeddings=embeddings,
    social_k=4,
    social_p=0.2,
    seed=42
)

# Get who can influence persona 0
influencers = dual_model.get_influence_neighbors(0)
```

### Key Methods
- `get_influence_weight(i, j)`: Get influence weight from i to j
- `get_influence_neighbors(node)`: Get all neighbors that can influence this node
- `update_cognitive_graph()`: Update with new embeddings
- `statistics()`: Get combined statistics

## Visualization

### Available Visualizations
1. **Similarity Heatmap**: Shows pairwise similarity matrix
2. **Social Graph**: Network visualization of social connections
3. **Cognitive Graph**: Network visualization of cognitive similarities
4. **Dual Graph**: Shows only active influences (both graphs aligned)
5. **Graph Comparison**: Side-by-side social vs cognitive graphs

### Usage
```python
from networks import (
    visualize_social_graph,
    visualize_cognitive_graph,
    visualize_dual_graph,
    visualize_similarity_heatmap
)

# Visualize social graph
fig = visualize_social_graph(social_graph)
fig.savefig('social_graph.png')

# Visualize dual-graph model
fig = visualize_dual_graph(dual_model)
fig.savefig('dual_graph.png')
```

## Running the Demo

```bash
python3 networks/demo.py
```

This will:
1. Load personas and generate embeddings
2. Create social and cognitive graphs
3. Build dual-graph model
4. Generate all visualizations
5. Save visualizations to `outputs/visualizations/`

## File Structure

```
embeddings/
├── generator.py      # Embedding generation
├── similarity.py     # Similarity calculations
└── __init__.py

networks/
├── social_graph.py   # Graph A: Social exposure
├── cognitive_graph.py # Graph B: Cognitive affinity
├── dual_graph.py     # Dual-graph model
├── visualization.py  # Visualization functions
├── demo.py           # Demo script
└── __init__.py
```

## Key Metrics

### Social Graph Metrics
- **Average Degree**: Average number of connections per persona
- **Clustering Coefficient**: How tightly connected neighbors are
- **Average Path Length**: Average shortest path between any two personas
- **Density**: Proportion of possible edges that exist

### Cognitive Graph Metrics
- **Average Similarity**: Mean similarity across all pairs
- **Similarity Distribution**: Range and variance of similarities
- **Edge Density**: Proportion of edges above threshold

### Dual-Graph Metrics
- **Active Influences**: Number of connections that exist in both graphs
- **Average Influence Weight**: Mean influence strength
- **Influence Efficiency**: Active influences / social edges

## Example Results

For 100 personas:
- **Social Graph**: ~200 edges (each persona connected to ~4 others)
- **Cognitive Graph**: ~4,950 edges (fully connected)
- **Active Influences**: ~50-100 edges (only 25-50% of social connections have sufficient similarity)

This means:
- Most social connections don't lead to influence (low cognitive similarity)
- Echo chambers form naturally (high similarity clusters)
- Influence is selective and targeted

## Next Steps (Future Phases)

- **Phase 5**: Belief update mechanism
- **Phase 6**: Temporal dynamics & rewiring
- **Phase 7**: Multi-round simulations

## Dependencies

- `sentence-transformers`: For embedding generation
- `networkx`: For graph operations
- `matplotlib`: For visualizations
- `numpy`: For numerical operations
- `scikit-learn`: For similarity calculations

Install with:
```bash
pip install sentence-transformers networkx matplotlib scikit-learn
```

