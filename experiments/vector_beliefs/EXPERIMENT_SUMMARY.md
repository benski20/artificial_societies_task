# Vector-Based Belief System Experiment Summary

## Overview

This experiment implements a **continuous, probabilistic belief system** where personas have latent belief vectors instead of text-based beliefs. Survey questions are embedded and projected onto this belief space to generate probabilistic responses.

## Key Features

### 1. Latent Belief Vectors
- Each persona has a **16-dimensional belief vector**
- Represents internal belief state in continuous space
- Initialized from demographic/behavioral attributes
- Updated directly (no text conversion needed)

### 2. Survey Question Embeddings
- Questions and options are embedded using sentence transformers
- Projected onto belief space via learned projection matrix
- Creates semantic mapping between questions and beliefs

### 3. Probabilistic Response Generation
- Project question/option embeddings onto persona's belief vector
- Calculate cosine similarity (dot product)
- Use softmax to get probability distribution
- Sample response probabilistically

### 4. Vector-Based Belief Updates
- Update belief vectors directly via weighted averages
- Use dual-graph model for influence weights
- No text conversion overhead

## Architecture

```
Persona Attributes → Belief Vector (16D)
                          ↓
                    Survey Questions
                          ↓
              Question Embeddings → Belief Space
                          ↓
              Project onto Belief Vector
                          ↓
              Calculate Probabilities (Softmax)
                          ↓
              Sample Response
```

## Results

### Simulation Run
- **Personas**: 100
- **Rounds**: 5
- **Belief Dimension**: 16
- **Influence Reduction**: 15%

### Survey Response Changes

**College Importance:**
- "Very Important": 22 → 30 (+8 personas, +36%)
- Shift toward valuing college

**Social Media Stress:**
- "Strongly Agree": 28 → 19 (-9 personas, -32%)
- "Neither Agree nor Disagree": 14 → 23 (+9 personas, +64%)
- Shift toward neutrality

**School Start Times:**
- "Strongly Support": 25 → 12 (-13 personas, -52%)
- "Support": 14 → 23 (+9 personas, +64%)
- Shift from strong support to moderate support

### Vector Metrics

**Belief Vector Statistics:**
- Mean Vector Norm: 1.17 (stable across rounds)
- Pairwise Similarity: 0.07 (low, indicating diversity)
- Vector norms range: [0.73, 1.80]

## Comparison with Original System

| Feature | Original System | Vector-Based System |
|---------|----------------|---------------------|
| Belief Representation | Text descriptions | Continuous vectors (16D) |
| Response Generation | Deterministic (logit-based) | Probabilistic (projection + softmax) |
| Belief Updates | Text → numeric → update → text | Direct vector operations |
| Computational Overhead | Text processing | Vector math only |
| Interpretability | High (readable text) | Lower (requires analysis) |
| Flexibility | Limited by text format | High (continuous space) |

## Advantages of Vector-Based System

1. **Continuous Belief Space**: Smooth transitions, no discretization
2. **Probabilistic Responses**: More realistic uncertainty
3. **Efficient Updates**: Direct vector operations, no text conversion
4. **Semantic Projection**: Questions naturally map to belief space
5. **Scalability**: Easy to add dimensions or questions

## Files Created

```
experiments/vector_beliefs/
├── persona/
│   └── vector_persona.py          # VectorPersona class
├── surveys/
│   ├── question_embeddings.py     # Question embedding system
│   └── vector_responses.py        # Probabilistic response generation
├── networks/
│   └── vector_belief_updates.py   # Vector belief updates
├── embeddings/
│   └── belief_embeddings.py       # Generate embeddings from vectors
├── utils/
│   └── metrics.py                 # Metrics calculation
├── run_vector_simulation.py       # Main simulation script
├── visualize_vector_simulation.py # Visualization script
└── README.md                       # Documentation
```

## Usage

```bash
# Run simulation
python3 experiments/vector_beliefs/run_vector_simulation.py

# Visualize results
python3 experiments/vector_beliefs/visualize_vector_simulation.py
```

## Output Files

```
outputs/vector_beliefs_simulation/
├── survey_evolution.json          # Survey responses by round
├── influence_stats.json           # Influence statistics
├── vector_metrics.json             # Belief vector metrics
├── final_personas.json            # Final personas with belief vectors
└── visualizations/
    ├── survey_evolution_*.png     # Individual question plots
    ├── vector_metrics.png          # Vector statistics over time
    └── change_summary.png          # Change summary
```

## Key Metrics Reported

1. **Vector Statistics**:
   - Vector norms (mean, std, min, max)
   - Per-dimension statistics
   - Pairwise cosine similarities

2. **Response Metrics**:
   - Response distributions
   - Probability statistics
   - Entropy (uncertainty)

3. **Evolution Metrics**:
   - Changes over rounds
   - Convergence/divergence patterns
   - Influence propagation

## Future Enhancements

1. **Higher Dimensionality**: Experiment with different belief dimensions
2. **Interpretable Dimensions**: Learn meaningful belief dimensions
3. **Multi-Modal Beliefs**: Combine vectors with text for interpretability
4. **Adaptive Projections**: Learn projection matrix from data
5. **Belief Trajectories**: Track individual persona belief changes

## Conclusion

The vector-based system successfully implements continuous, probabilistic beliefs with:
- ✅ Smooth belief updates
- ✅ Probabilistic responses
- ✅ Efficient computation
- ✅ Realistic social influence dynamics

The system shows measurable changes in survey responses over rounds, demonstrating that belief vectors can effectively model opinion dynamics.

