# Metrics Comparison: Original vs Vector-Based Systems

This folder contains comprehensive metrics and comparisons between the original text-based belief system and the vector-based belief system.

## Metrics Calculated

### 1. Belief Space Diversity
- **Embedding Spread**: Variance and distribution of embeddings in belief space
- **Belief Vector Diversity**: For vector system, measures spread of belief vectors
- **Dimension Analysis**: How beliefs vary across dimensions

### 2. Aggregate Response Distributions
- Response distributions for each question
- Changes over time (rounds)
- Comparison between systems

### 3. Opinion Clustering
- Network-based clustering: Do similar beliefs cluster in the social network?
- Cognitive similarity vs social connections
- Echo chamber detection

### 4. Response Entropy
- Uncertainty in responses (Shannon entropy)
- How certain/uncertain are personas?
- Entropy changes over time

### 5. Influence Network Metrics
- Centrality measures
- Influence distribution
- Network structure comparison

### 6. Convergence Metrics
- How quickly opinions converge
- Polarization measures
- Stability over time

## Files

- `scripts/calculate_metrics.py` - Main metrics calculation script
- `scripts/compare_systems.py` - Side-by-side comparison
- `visualizations/` - All comparison plots
- `reports/` - Detailed metric reports

## Usage

```bash
# Calculate all metrics
python3 metrics/scripts/calculate_metrics.py

# Generate comparison report
python3 metrics/scripts/compare_systems.py
```

