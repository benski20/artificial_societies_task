# Belief Update Mechanism: Dual-Graph Model Integration

## How Belief Updates Work

The belief update system is **fully integrated with the dual-graph model**, ensuring that influence only occurs when both social connections AND cognitive similarity exist.

## Dual-Graph Influence Formula

```
Influence Weight = Social_Connected × Cognitive_Similarity × Reduction_Factor
```

Where:
- **Social_Connected**: 1 if connected in Graph A (Watts-Strogatz), 0 otherwise
- **Cognitive_Similarity**: 0.0-1.0 from embedding similarity (Graph B)
- **Reduction_Factor**: 0.15 (15% of raw influence for realism)

## Belief Update Process

### Step 1: Get Influencers
```python
influencers = dual_model.get_influence_neighbors(persona_idx)
```

This returns only neighbors that:
- ✅ Are socially connected (Graph A)
- ✅ Have cognitive similarity (Graph B)
- ✅ Have non-zero influence weight

**Example**: If Persona A is socially connected to Personas B, C, D:
- Persona B: Social=1, Cognitive=0.8 → Influence = 1 × 0.8 × 0.15 = **0.12** ✓
- Persona C: Social=1, Cognitive=0.3 → Influence = 1 × 0.3 × 0.15 = **0.045** ✓
- Persona D: Social=0 (not connected) → Influence = 0 × 0.7 × 0.15 = **0.0** ✗

Only B and C can influence A, even though D might be cognitively similar.

### Step 2: Weighted Belief Update
```python
# Calculate weighted average of neighbor beliefs
weighted_sum = sum(neighbor_belief × influence_weight for each neighbor)
total_weight = sum(influence_weights)

avg_neighbor_belief = weighted_sum / total_weight

# Update formula
belief_difference = avg_neighbor_belief - current_belief
update = belief_difference × susceptibility × avg_influence × 0.5
new_belief = current_belief + update
```

### Step 3: Susceptibility Modulation
Each persona has different susceptibility based on:
- Mental health (Poor = +30% susceptibility)
- Social media intensity (Almost Constant = +25% susceptibility)
- GPA (Lower = more susceptible)

## Key Properties

### 1. Selective Influence
- Not all social connections create influence
- Only those with sufficient cognitive similarity
- Reduction factor (15%) further limits influence

### 2. Weighted by Similarity
- Higher cognitive similarity = stronger influence
- Social connection is binary (on/off)
- Combined weight determines influence strength

### 3. Realistic Resistance
- 15% reduction factor simulates real-world resistance
- People don't always change beliefs even when exposed
- Susceptibility varies by persona traits

## Example Flow

**Persona 0 wants to update beliefs:**

1. **Get influencers from dual-graph:**
   - Persona 13: influence_weight = 0.12 (social=1, cognitive=0.8, reduction=0.15)
   - Persona 45: influence_weight = 0.09 (social=1, cognitive=0.6, reduction=0.15)
   - Persona 67: influence_weight = 0.0 (social=0, cognitive=0.9) → **No influence**

2. **Calculate weighted average:**
   - Persona 13 belief: 0.7, weight: 0.12
   - Persona 45 belief: 0.5, weight: 0.09
   - Weighted avg = (0.7×0.12 + 0.5×0.09) / (0.12+0.09) = 0.614

3. **Apply update:**
   - Current belief: 0.6
   - Difference: 0.614 - 0.6 = 0.014
   - Susceptibility: 0.65 (moderate)
   - Update: 0.014 × 0.65 × 0.105 × 0.5 = 0.0005
   - New belief: 0.6005 (small change)

## Verification

The code ensures:
- ✅ `get_influence_neighbors()` uses dual-graph model
- ✅ Influence weights include social × cognitive × reduction
- ✅ Only neighbors with non-zero weights are considered
- ✅ Weights are properly matched to beliefs
- ✅ Updates are weighted by influence strength

## Files

- `networks/belief_updates.py` - Belief update logic using dual-graph
- `networks/dual_graph.py` - Dual-graph model implementation
- `networks/simulation.py` - Multi-round simulation using updates

## Summary

**Belief updates ARE correctly using the dual-graph model:**
- Social connections (Graph A) determine who can influence
- Cognitive similarity (Graph B) determines influence strength
- Combined weight determines actual influence
- Reduction factor makes it realistic

This creates selective influence where only socially connected AND cognitively similar neighbors can change beliefs.

