# Vector-Based Belief System Experiment

## Overview

This experiment implements a **continuous, probabilistic belief system** where:
- Each persona has a **latent belief vector** (internal representation)
- Survey questions are **embedded** and **projected** onto this belief space
- Responses are **probabilistic** based on the projection

## Key Differences from Original System

### Original System
- Beliefs: Text descriptions (e.g., "The persona strongly believes that college is very important")
- Responses: Deterministic based on attributes + text beliefs
- Updates: Text-to-numeric conversion → update → numeric-to-text

### Vector-Based System
- Beliefs: Continuous vector in belief space (e.g., `[0.7, -0.3, 0.5, ...]`)
- Responses: Probabilistic based on projection of question onto belief vector
- Updates: Direct vector operations (weighted averages, etc.)

## Architecture

### 1. Latent Belief Vector
- Each persona has a `belief_vector` of dimension `belief_dim` (default: 16)
- Represents internal belief state in continuous space
- Initialized from demographics/attributes or randomly

### 2. Survey Question Embeddings
- Each survey question + each option is embedded
- Creates a question embedding space
- Questions are projected onto belief space

### 3. Probabilistic Response Generation
- Project question embedding onto persona's belief vector
- Calculate similarity/dot product for each option
- Use softmax to get probability distribution
- Sample response probabilistically

### 4. Belief Updates
- Update belief vectors directly (not text)
- Weighted average of neighbor belief vectors
- Influence weights from dual-graph model

## Files Structure

```
experiments/vector_beliefs/
├── persona/
│   └── vector_persona.py      # Persona with belief vector
├── surveys/
│   ├── question_embeddings.py # Embed survey questions
│   └── vector_responses.py     # Probabilistic responses via projection
├── networks/
│   └── vector_belief_updates.py # Update belief vectors
├── embeddings/
│   └── belief_embeddings.py   # Generate embeddings from vectors
├── utils/
│   └── metrics.py              # Metrics for vector system
└── run_vector_simulation.py    # Main simulation script
```

## Usage

```bash
# Run vector-based simulation
python3 experiments/vector_beliefs/run_vector_simulation.py

# Compare with original system
python3 run_simulation.py  # Original
python3 experiments/vector_beliefs/run_vector_simulation.py  # Vector-based
```

## Metrics

- Belief vector statistics (mean, std, distribution)
- Projection magnitudes
- Response probability distributions
- Belief vector changes over rounds
- Cosine similarity between belief vectors

