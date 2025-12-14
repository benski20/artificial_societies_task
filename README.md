# Synthetic Persona Generation for U.S. High School Students

A comprehensive framework for generating 100 synthetic personas representing U.S. high school students (ages 16-18) using multiple methodologies, with LLM augmentation for narrative richness.

## Project Structure

```
AS_task/
├── data/                    # Data layer
│   ├── __init__.py
│   ├── distributions.py    # Distribution functions (lognormal, normal, etc.)
│   └── research_data.py     # Research constants and parameters
├── generation/              # Persona generation methods
│   ├── __init__.py
│   ├── marginal.py         # Method 1: Marginal Distributions
│   ├── constraint_based.py # Method 2: Constraint-Based (Primary)
│   └── stratified.py       # Method 3: Stratified Sampling
├── persona/                 # Persona representation
│   ├── __init__.py
│   ├── persona.py          # Persona class/data structure
│   └── traits.py           # Latent traits and belief vectors
├── llm/                     # LLM augmentation layer
│   ├── __init__.py
│   ├── augmentation.py     # Narrative and belief generation
│   └── prompts.py          # LLM prompt templates
├── surveys/                 # Survey system
│   ├── __init__.py
│   ├── questions.py        # Survey question definitions
│   └── responses.py        # Response generation logic
├── networks/                # Network models (future)
│   ├── __init__.py
│   ├── graph.py            # Dual-graph model
│   └── influence.py        # Influence propagation
├── embeddings/              # Embedding system (future)
│   ├── __init__.py
│   └── similarity.py       # Embedding & similarity calculations
├── evaluation/              # Validation and metrics
│   ├── __init__.py
│   ├── validation.py       # Distribution validation
│   └── metrics.py          # Comparison metrics
├── outputs/                 # Generated personas and results
├── config/                  # Configuration files
│   └── .env.example        # Example environment variables
├── main.py                  # Main orchestration script
├── requirements.txt         # Python dependencies
├── .gitignore
├── README.md                # This file
└── persona_methodology_guide.md  # Detailed methodology documentation
```

## Setup

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp config/.env.example config/.env
# Edit config/.env with your API keys
```

## Usage

```python
from generation.constraint_based import generate_constraint_based_personas
from llm.augmentation import augment_personas_with_llm

# Generate 100 personas using constraint-based method
personas = generate_constraint_based_personas(n=100)

# Augment with LLM narratives
augmented_personas = augment_personas_with_llm(personas)

# Generate survey responses
from surveys.responses import respond_to_survey
responses = [respond_to_survey(p, "How important is college to your future?") 
             for p in augmented_personas]
```

## Methods

Three generation methodologies are implemented:

1. **Marginal Distributions**: Fast, independent sampling
2. **Constraint-Based**: Most realistic, captures correlations (recommended)
3. **Stratified Sampling**: Most representative, guaranteed demographic coverage

See `persona_methodology_guide.md` for detailed methodology comparison.

## Features

- ✅ Statistical distribution-based generation
- ✅ LLM augmentation for narrative richness
- ✅ Survey response generation
- 🔄 Network models (in progress)
- 🔄 Temporal dynamics (planned)
- 🔄 Embedding-based similarity (planned)

## Research Foundation

All distributions and correlations are based on:
- CDC Adolescent Behaviors and Experiences Survey (ABES) 2021
- Pew Research Center Social Media & Technology surveys 2024-2025
- National Center for Education Statistics (NCES)
- U.S. Bureau of Labor Statistics (BLS)

## License

[Add your license here]
