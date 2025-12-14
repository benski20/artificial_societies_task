# LLM Augmentation Setup Guide

To add rich narratives and beliefs to your personas using LLM augmentation, follow these steps:

## Step 1: Install OpenAI Package

```bash
pip install openai
```

Or if using a virtual environment:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install openai
```

## Step 2: Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Create a new API key
4. Copy the key (it starts with `sk-...`)

## Step 3: Configure Environment Variables

Create or edit `config/.env`:

```bash
mkdir -p config
```

Then create `config/.env` with:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
DEFAULT_LLM_MODEL=gpt-4o-mini
```

**Important**: Never commit your `.env` file to git! It's already in `.gitignore`.

## Step 4: Augment Existing Personas

If you already have personas generated, use the augmentation script:

```bash
python3 augment_existing_personas.py
```

This will:
- Load personas from `outputs/constraint_personas.json`
- Add LLM-generated narratives and beliefs
- Save to `outputs/constraint_personas_augmented.json`

## Step 5: Generate New Personas with LLM

Or generate new personas with LLM augmentation from the start:

```bash
python3 main.py --method constraint --n 100 --augment --surveys --validate
```

## What Gets Added

Each persona will get:
- **Narrative**: A 2-3 paragraph background story connecting all attributes
- **Beliefs**: Explanations of the persona's perspective on:
  - College importance and future plans
  - Social media impact on daily life
  - School start times and sleep
  - Mental health and stress
  - Academic pressure and expectations

## Cost Estimate

Using `gpt-4o-mini`:
- ~$0.15 per 100 personas (narratives + beliefs)
- Very affordable for research purposes

## Troubleshooting

### "OpenAI package not installed"
```bash
pip install openai
```

### "OPENAI_API_KEY not found"
- Check that `config/.env` exists
- Verify the key is correct (starts with `sk-`)
- Make sure you're running from the project root directory

### "Rate limit exceeded"
- OpenAI has rate limits on free tier
- Wait a few minutes and try again
- Consider upgrading your OpenAI plan

### API Errors
- Check your OpenAI account has credits
- Verify the API key is valid
- Check OpenAI status page: https://status.openai.com/

## Alternative: Use Without LLM

The system works perfectly fine without LLM augmentation! You can:
- Generate personas using pure statistical methods
- Generate survey responses
- Validate distributions
- Compare methods

LLM augmentation is optional and adds narrative richness, but isn't required for the core functionality.

