# Implementation Notes

This document tracks the implementation progress and key decisions made during development.

## Project Structure

The project follows a modular architecture:

```
AS_task/
├── data/              # Research data and distribution functions
├── generation/        # Three generation methods
├── persona/           # Persona data structure
├── llm/               # LLM augmentation layer
├── surveys/           # Survey system
├── evaluation/        # Validation and metrics
├── outputs/          # Generated results
└── config/            # Configuration files
```

## Implementation Status

### ✅ Completed

1. **Project Setup**
   - Folder structure created
   - `.gitignore` configured
   - `requirements.txt` with all dependencies
   - Updated `README.md` with project documentation

2. **Data Layer** (`data/`)
   - `research_data.py`: All research constants and distributions
   - `distributions.py`: Sampling functions (lognormal, normal, categorical)

3. **Persona Representation** (`persona/`)
   - `persona.py`: Complete Persona dataclass with all attributes
   - Support for LLM-generated fields (narrative, beliefs, embeddings)

4. **Generation Methods** (`generation/`)
   - `marginal.py`: Method 1 - Marginal Distributions
   - `constraint_based.py`: Method 2 - Constraint-Based (PRIMARY)
   - `stratified.py`: Method 3 - Stratified Sampling
   - All methods implement the same interface

5. **LLM Augmentation** (`llm/`)
   - `prompts.py`: Prompt templates for narratives and beliefs
   - `augmentation.py`: Functions to augment personas with LLM
   - Supports both OpenAI and Anthropic APIs
   - Graceful fallback if API keys not configured

6. **Survey System** (`surveys/`)
   - `questions.py`: Survey question definitions
   - `responses.py`: Probabilistic response generation
   - Support for LLM-based responses (optional)

7. **Evaluation** (`evaluation/`)
   - `validation.py`: Distribution and correlation validation
   - `metrics.py`: Comparison metrics and analysis

8. **Orchestration**
   - `main.py`: Complete CLI pipeline
   - `example_usage.py`: Example scripts demonstrating usage

## Key Design Decisions

### 1. Hybrid Approach (Distributions + LLM)

We use a two-layer approach:
- **Layer 1**: Statistical generation ensures population-level accuracy
- **Layer 2**: LLM augmentation adds narrative richness

This combines the best of both worlds: statistical rigor + realistic narratives.

### 2. Constraint-Based as Primary Method

Method 2 (Constraint-Based) is recommended as the primary method because:
- Captures realistic correlations
- Better survey validity
- More believable personas
- Still computationally efficient

### 3. Modular LLM Integration

LLM augmentation is optional and can be:
- Skipped entirely (pure statistical generation)
- Applied selectively (e.g., only to a subset)
- Used for different purposes (narratives, beliefs, survey responses)

### 4. Probabilistic Survey Responses

Survey responses use probabilistic models based on persona attributes:
- Logistic functions map attributes to response probabilities
- More stable and reproducible than pure LLM
- Can be validated against expected distributions

## Usage Patterns

### Basic Usage (No LLM)

```python
from generation.constraint_based import generate_constraint_based_personas

personas = generate_constraint_based_personas(n=100, random_seed=42)
```

### With LLM Augmentation

```python
from generation.constraint_based import generate_constraint_based_personas
from llm.augmentation import augment_personas_with_llm

personas = generate_constraint_based_personas(n=100, random_seed=42)
augmented = augment_personas_with_llm(personas, model_name='gpt-4o-mini')
```

### Full Pipeline

```python
# Generate
personas = generate_constraint_based_personas(n=100)

# Augment
augmented = augment_personas_with_llm(personas)

# Survey
from surveys.responses import generate_survey_responses
responses = generate_survey_responses(augmented)

# Validate
from evaluation.validation import validate_distributions
validation = validate_distributions(augmented)
```

## Environment Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (optional, for LLM):
```bash
cp config/.env.example config/.env
# Edit config/.env with your API keys
```

## Testing

Run examples:
```bash
python example_usage.py
```

Run full pipeline:
```bash
python main.py --method constraint --n 100 --validate
```

With LLM augmentation:
```bash
python main.py --method constraint --n 100 --augment --surveys --validate
```

## Future Enhancements

### Planned (Not Yet Implemented)

1. **Network Models** (`networks/`)
   - Dual-graph model (social exposure + cognitive affinity)
   - Influence propagation
   - Network rewiring

2. **Embeddings** (`embeddings/`)
   - Text embedding generation
   - Similarity calculations
   - Clustering and diversity metrics

3. **Temporal Dynamics**
   - Multi-round simulations
   - Belief updates
   - Opinion drift tracking

4. **Advanced Features**
   - Export to various formats (CSV, Parquet)
   - Visualization tools
   - Interactive exploration

## Notes

- All random number generation uses `numpy.random.Generator` for reproducibility
- Persona objects are serializable to JSON (with numpy array conversion)
- LLM augmentation gracefully handles API failures
- Validation reports show expected vs observed distributions

## Dependencies

Key dependencies:
- `numpy`, `scipy`: Statistical distributions
- `pandas`: Data manipulation
- `openai` / `anthropic`: LLM APIs (optional)
- `sentence-transformers`: Embeddings (future)
- `networkx`: Network analysis (future)

See `requirements.txt` for complete list.

