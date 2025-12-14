# Metrics Comparison Guide

## Overview

This folder contains comprehensive metrics comparing the **Original Text-Based System** and the **Vector-Based System**.

## Metrics Calculated

### 1. Belief Space Diversity (Embedding Spread)

**What it measures:**
- How spread out beliefs are in the embedding space
- Diversity of belief representations

**Metrics:**
- Mean pairwise distance between embeddings
- Distance range (max - min)
- Dimension-wise variance
- PCA variance explained

**Key Finding:**
- Vector system has **more diverse** beliefs (mean distance: 1.34 vs 0.96)
- Belief vectors are more spread out in continuous space

### 2. Aggregate Response Distributions

**What it measures:**
- Final distribution of responses for each question
- How opinions are distributed across options

**Visualizations:**
- Side-by-side bar charts comparing final distributions
- Evolution plots showing changes over time

**Key Finding:**
- Original system: More polarized (e.g., 54% "Very Important")
- Vector system: More uniform (e.g., 30% "Very Important", 24% each for others)

### 3. Response Entropy (Uncertainty)

**What it measures:**
- Shannon entropy of response distributions
- Higher entropy = more uncertainty/diversity
- Lower entropy = more certainty/concentration

**Formula:** `H = -Σ(p * log2(p))`

**Key Finding:**
- Vector system shows **higher entropy** (more uncertainty)
- More realistic probabilistic responses
- Original system converges to lower entropy (more certain)

### 4. Opinion Clustering (Network Homophily)

**What it measures:**
- Do beliefs cluster along social connections?
- Homophily effect = Connected similarity - Unconnected similarity

**Metrics:**
- Connected pairs similarity (socially connected)
- Unconnected pairs similarity
- Homophily effect (positive = clustering)
- Clustering quality (silhouette score)

**Key Finding:**
- Vector system shows **stronger homophily** (0.06 vs 0.006)
- Beliefs cluster more along social connections
- More realistic echo chamber effects

### 5. Convergence Metrics

**What it measures:**
- How quickly opinions stabilize
- Convergence round (when 95% stability reached)
- Total change over simulation

**Key Finding:**
- Original system: Converges faster (Round 1-2)
- Vector system: Slower convergence (Round 3+ or no convergence)
- Vector system remains more dynamic

### 6. Influence Network Metrics

**What it measures:**
- Network structure and centrality
- Influence distribution
- Network density

**Metrics:**
- In-degree centrality (who influences you)
- Out-degree centrality (who you influence)
- Influence strength (weighted centrality)
- Network density

## Files Generated

### Reports
- `reports/original_metrics.json` - All metrics for original system
- `reports/vector_metrics.json` - All metrics for vector system
- `reports/system_comparison_report.txt` - Comprehensive comparison

### Visualizations
- `visualizations/embedding_spread_comparison.png` - Belief diversity
- `visualizations/response_distribution_*.png` - Final distributions
- `visualizations/entropy_over_time.png` - Entropy evolution
- `visualizations/opinion_clustering_comparison.png` - Clustering metrics
- `visualizations/network_metrics_comparison.png` - Network structure
- `visualizations/response_evolution_*.png` - Response changes over time

## Usage

```bash
# Calculate all metrics
python3 metrics/scripts/calculate_metrics.py

# Generate visualizations
python3 metrics/scripts/visualize_metrics.py

# Generate comparison report
python3 metrics/scripts/compare_systems.py
```

## Key Comparisons

| Metric | Original System | Vector System | Winner |
|--------|----------------|---------------|--------|
| **Belief Diversity** | Lower (0.96) | Higher (1.34) | Vector |
| **Response Entropy** | Lower (more certain) | Higher (more uncertain) | Vector (more realistic) |
| **Homophily** | Weak (0.006) | Strong (0.06) | Vector (more realistic) |
| **Convergence Speed** | Fast (Round 1-2) | Slow (Round 3+) | Original (faster) |
| **Polarization** | High (54% extreme) | Low (30% extreme) | Vector (more nuanced) |

## Interpretation

### Vector System Advantages:
1. **More Diverse Beliefs**: Beliefs spread out more in continuous space
2. **Higher Uncertainty**: More realistic probabilistic responses
3. **Stronger Homophily**: Beliefs cluster along social connections (realistic)
4. **Less Polarization**: More nuanced, less extreme responses
5. **More Dynamic**: Slower convergence, opinions continue evolving

### Original System Advantages:
1. **Faster Convergence**: Reaches stable state quickly
2. **More Interpretable**: Text beliefs are readable
3. **Clearer Patterns**: More obvious polarization trends

## Conclusion

The **vector system** better captures:
- Realistic uncertainty in responses
- Belief diversity in continuous space
- Network-based opinion clustering
- Nuanced, less polarized opinions

The **original system** better captures:
- Fast convergence to stable states
- Clear, interpretable belief patterns
- Strong polarization effects

Both systems have value depending on the research question!

