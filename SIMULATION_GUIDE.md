# Multi-Round Simulation Guide

## Overview

The simulation system implements belief updates and multi-round survey responses to track how opinions change over time through social influence.

## Key Features

### 1. Influence Weight Reduction ✅
- **Factor**: 0.15 (15% of raw influence weights)
- **Purpose**: Simulates real-world resistance to influence
- **Effect**: People don't always change beliefs even when exposed to similar others
- **Implementation**: Applied in `DualGraphModel` class

### 2. Belief Update Mechanism ✅
- **Susceptibility Factors**:
  - Mental health (Poor = +30% susceptibility)
  - Social media intensity (Almost Constant = +25% susceptibility)
  - GPA (Lower GPA = more susceptible)
- **Update Formula**: 
  ```
  new_belief = current_belief + (neighbor_avg - current_belief) × susceptibility × influence_weight × 0.5
  ```
- **Implementation**: `networks/belief_updates.py`

### 3. Multi-Round Simulation ✅
- **Rounds**: Default 5 rounds
- **Each Round**:
  1. Update beliefs based on neighbor influences
  2. Update embeddings (recalculate from new beliefs)
  3. Update cognitive graph (recalculate similarities)
  4. Generate survey responses (based on updated beliefs)
- **Implementation**: `networks/simulation.py`

### 4. Survey Response Integration ✅
- Survey responses now incorporate belief strength (70% weight)
- Base attributes still matter (30% weight)
- Beliefs override static attributes when updated
- **Implementation**: `surveys/responses.py`

## Usage

### Run Simulation
```bash
python3 run_simulation.py
```

This will:
1. Load 100 personas
2. Create dual-graph model with 15% influence reduction
3. Run 5 rounds of belief updates
4. Track survey response changes
5. Save results to `outputs/simulation/`

### Visualize Results
```bash
python3 visualize_simulation.py
```

This generates:
- Individual evolution plots for each survey question
- Combined evolution plot (all questions)
- Change summary (first → last round)

## Results Structure

```
outputs/simulation/
├── multi_round_survey_evolution.json    # Survey responses by round
├── multi_round_influence_stats.json     # Influence statistics by round
├── final_personas.json                  # Personas after all rounds
└── visualizations/
    ├── survey_evolution_college_importance.png
    ├── survey_evolution_social_media_stress.png
    ├── survey_evolution_school_start_times.png
    ├── all_surveys_evolution.png
    └── change_summary.png
```

## Example Results

From a 5-round simulation:

**College Importance:**
- Round 0: 43% "Very Important"
- Round 5: 54% "Very Important" (+11%)
- Change: More personas value college after social influence

**Social Media Stress:**
- Round 0: 56% "Strongly Agree" (increases stress)
- Round 5: 63% "Strongly Agree" (+7%)
- Change: Consensus building around social media stress

**School Start Times:**
- Round 0: 58% "Strongly Support" later times
- Round 5: 62% "Strongly Support" (+4%)
- Change: Slight increase in support

## Key Insights

1. **Opinion Drift**: Beliefs shift gradually over rounds
2. **Consensus Building**: Some topics show convergence
3. **Selective Influence**: Only ~15% of potential influence is realized
4. **Susceptibility Matters**: Personas with poor mental health change more
5. **Network Effects**: Changes propagate through social connections

## Parameters

### Influence Reduction Factor
- **Default**: 0.15 (15%)
- **Range**: 0.0-1.0
- **Lower** = More resistance to influence (more realistic)
- **Higher** = More influence (faster changes)

### Susceptibility Base
- **Default**: 0.5 (50%)
- **Adjustments**:
  - Poor mental health: +0.3
  - Almost constant social media: +0.25
  - Low GPA: +0.1

### Update Strength
- **Scale Factor**: 0.5 (in update formula)
- Controls how much beliefs change per round
- Lower = slower, more gradual changes

## Customization

### Change Number of Rounds
```python
final_personas, results = run_simulation(
    personas=personas,
    n_rounds=10,  # More rounds
    influence_reduction_factor=0.15
)
```

### Adjust Influence Reduction
```python
# More realistic (less influence)
influence_reduction_factor=0.10

# Less realistic (more influence)
influence_reduction_factor=0.30
```

### Disable Embedding Updates
```python
run_simulation(
    personas=personas,
    update_embeddings=False  # Keep embeddings fixed
)
```

## Files

- `networks/belief_updates.py` - Belief update logic
- `networks/simulation.py` - Multi-round simulation
- `run_simulation.py` - Main simulation script
- `visualize_simulation.py` - Visualization script
- `surveys/responses.py` - Updated to use beliefs

## Next Steps

Future enhancements:
- Add belief polarization metrics
- Track individual persona belief trajectories
- Implement network rewiring based on belief changes
- Add more sophisticated belief update models
- Compare different influence reduction factors

