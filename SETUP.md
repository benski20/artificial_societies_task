# Setup Guide

Quick start guide for the Synthetic Persona Generation project.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd AS_task
```

2. **Create a virtual environment:**
```bash
python -m venv venv
```

3. **Activate the virtual environment:**
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Configuration (Optional - for LLM features)

If you want to use LLM augmentation features:

1. **Create environment file:**
```bash
mkdir -p config
```

2. **Create `config/.env` file with your API keys:**
```bash
# OpenAI API Key (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (optional, for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Model preferences
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Generation parameters
RANDOM_SEED=42
NUM_PERSONAS=100
```

**Note:** The project works without LLM features. You can generate personas and survey responses using pure statistical methods.

## Quick Start

### Example 1: Basic Generation

```python
from generation.constraint_based import generate_constraint_based_personas

# Generate 100 personas
personas = generate_constraint_based_personas(n=100, random_seed=42)

# View first persona
print(personas[0].to_dict())
```

### Example 2: Run Examples

```bash
python example_usage.py
```

### Example 3: Full Pipeline

```bash
# Generate and validate (no LLM)
python main.py --method constraint --n 100 --validate

# With LLM augmentation (requires API key)
python main.py --method constraint --n 100 --augment --surveys --validate
```

## Available Methods

- `marginal`: Fast, independent sampling
- `constraint`: Most realistic, captures correlations (recommended)
- `stratified`: Most representative, guaranteed demographic coverage
- `all`: Run all three methods and compare

## Output

Results are saved to the `outputs/` directory:
- `{method}_personas.json`: Generated personas
- `{method}_survey_results.json`: Survey responses (if generated)

## Troubleshooting

### LLM API Errors

If you see errors about missing API keys:
- The project works fine without LLM features
- Set up `config/.env` with your API keys if you want LLM augmentation
- LLM features are optional - statistical generation works independently

### Import Errors

If you see import errors:
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check that you're in the project root directory

### Validation Warnings

If validation shows differences from expected values:
- This is normal - small variations are expected
- Differences < 5% are generally acceptable
- For more accurate results, increase `n` (number of personas)

## Next Steps

1. Read `README.md` for project overview
2. Check `persona_methodology_guide.md` for methodology details
3. Review `IMPLEMENTATION_NOTES.md` for implementation details
4. Run `example_usage.py` to see examples
5. Use `main.py` for full pipeline execution

## Support

For questions or issues, refer to:
- `README.md`: Project documentation
- `IMPLEMENTATION_NOTES.md`: Implementation details
- `persona_methodology_guide.md`: Methodology guide

