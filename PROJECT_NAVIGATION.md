# How to Navigate This Project: A Stream-of-Consciousness Guide

Welcome! You've stumbled into a fascinating project about simulating social dynamics among high school students. Let me walk you through this codebase as if we're exploring it together for the first time.

## What Is This Project, Really?

At its core, this is a **synthetic persona modeling system** that creates 100 realistic high school students (ages 16-18), gives them beliefs, connects them in social networks, and then watches how their opinions change as they influence each other over time. Think of it as a digital petri dish for studying how ideas spread through a social network.

But here's the twist: we built **two different systems** to represent beliefs:
1. **Original System**: Beliefs are text descriptions (like "The persona strongly believes that college is very important")
2. **Vector System**: Beliefs are continuous vectors in a 16-dimensional space (like `[0.7, -0.3, 0.5, ...]`)

We then compare them to see which better represents human belief systems. Spoiler: both have strengths!

---

## Where Do I Start? (The Entry Points)

### If you want to **run the original system**:
```bash
python3 run_simulation.py
```
This loads 100 personas, runs 5 rounds of belief updates, and saves results to `outputs/simulation/`

### If you want to **run the vector-based system**:
```bash
python3 experiments/vector_beliefs/run_vector_simulation.py
```
Same idea, but uses continuous belief vectors instead of text. Results go to `outputs/vector_beliefs_simulation/`

### If you want to **generate personas from scratch**:
```bash
python3 main.py --method constraint --n_personas 100
```
This creates personas using demographic distributions, then optionally augments them with LLM-generated narratives.

---

## The Project Structure: A Mental Map

Let me walk you through the folders as if we're exploring a house together...

### `/persona/` - The Foundation
**What's here:** The data structure that represents a single high school student.

**Key file:** `persona.py`
- This is the `Persona` class - think of it as a blueprint for what makes a student
- Contains demographics (age, gender, race, income, GPA)
- Contains behavioral traits (mental health, social media use, sports)
- Contains beliefs (text descriptions in original system)
- Contains survey responses

**Why it matters:** Everything else builds on this. When you see "persona" throughout the codebase, this is what they're talking about.

---

### `/data/` - The Research Foundation
**What's here:** The statistical distributions and correlations that make personas realistic.

**Key files:**
- `research_data.py`: Real demographic data (gender percentages, income distributions, GPA means)
- `distributions.py`: Functions to sample from distributions (lognormal for income, normal for GPA)

**Why it matters:** This is what makes the personas realistic. Instead of random numbers, we use actual U.S. high school demographics. When you see "constraint-based generation," this is the data it uses.

---

### `/generation/` - Creating the Personas
**What's here:** Three different methods to generate personas.

**Key files:**
- `constraint_based.py`: **The main method** - uses probabilistic constraints to reflect real correlations (e.g., higher income → higher GPA → more likely to plan for college)
- `marginal.py`: Baseline method - samples each attribute independently
- `stratified.py`: Ensures representation - guarantees certain demographics are present

**Why it matters:** This is where the 100 personas come from. The constraint-based method is the most realistic because it captures correlations (like how income affects college plans).

---

### `/llm/` - Adding Narrative Richness
**What's here:** Code to augment statistical personas with LLM-generated stories and beliefs.

**Key files:**
- `augmentation.py`: Calls OpenAI API to generate narratives and beliefs
- `prompts.py`: Templates for LLM prompts

**Why it matters:** Raw statistics create boring personas. The LLM adds personality, background stories, and nuanced beliefs. This is optional but makes personas feel more human.

**Note:** You need an OpenAI API key in `config/.env` for this to work.

---

### `/surveys/` - Asking Questions
**What's here:** The survey questions and response generation logic.

**Key files:**
- `questions.py`: Defines the three survey questions (college importance, social media stress, school start times)
- `responses.py`: Generates responses based on persona attributes and beliefs

**Why it matters:** This is how we measure what personas think. The response generation uses probabilistic models (logit functions) to simulate realistic answers.

---

### `/embeddings/` - Converting Text to Numbers
**What's here:** Code to generate embeddings (numerical representations) from persona text.

**Key files:**
- `generator.py`: Uses sentence-transformers to create embeddings from narratives/beliefs
- `similarity.py`: Calculates cosine similarity between embeddings

**Why it matters:** Embeddings let us measure how similar two personas' beliefs are. This is crucial for the cognitive affinity graph (Graph B).

---

### `/networks/` - The Social Dynamics Engine
**What's here:** The dual-graph model and belief update mechanisms. This is where the magic happens.

**Key files:**
- `social_graph.py`: Creates Graph A (Watts-Strogatz small-world network) - who talks to whom
- `cognitive_graph.py`: Creates Graph B (fully connected, similarity-weighted) - how similar beliefs are
- `dual_graph.py`: **The core model** - combines both graphs. Influence only happens when BOTH social connection AND cognitive similarity exist
- `belief_updates.py`: Updates beliefs based on neighbor influences
- `simulation.py`: Runs multi-round simulations

**Why it matters:** This is the heart of the project. The dual-graph model ensures realistic influence: you're not influenced by everyone, only by people you know AND who think similarly to you.

**The formula:** `influence = social_connected × cognitive_similarity × 0.15`

---

### `/evaluation/` - Checking Our Work
**What's here:** Validation and metrics to ensure personas are realistic.

**Key files:**
- `validation.py`: Checks if generated personas match expected demographics
- `metrics.py`: Calculates comparison metrics between generation methods

**Why it matters:** We need to verify that our synthetic population actually represents real high school students. This catches errors and validates the approach.

---

### `/experiments/vector_beliefs/` - The Alternative System
**What's here:** A complete parallel implementation using continuous belief vectors instead of text.

**Structure mirrors the main project:**
- `persona/vector_persona.py`: Persona with belief vector (16D) instead of text beliefs
- `surveys/question_embeddings.py`: Embeds questions into belief space
- `surveys/vector_responses.py`: Generates probabilistic responses via projection
- `networks/vector_belief_updates.py`: Updates belief vectors directly
- `run_vector_simulation.py`: Main simulation script

**Why it matters:** This is our experimental alternative. Instead of "The persona strongly believes...", we have `[0.7, -0.3, 0.5, ...]`. Responses are generated by projecting questions onto this vector space. It's more computationally efficient and allows for smoother belief transitions.

**Key difference:** Vector system is probabilistic and continuous; original system is deterministic and discrete.

---

### `/metrics/` - Comparing Everything
**What's here:** Comprehensive metrics comparing both systems.

**Key files:**
- `scripts/calculate_metrics.py`: Calculates all metrics (diversity, entropy, clustering, etc.)
- `scripts/visualize_metrics.py`: Generates comparison plots
- `scripts/compare_systems.py`: Creates detailed comparison report

**Metrics calculated:**
1. **Belief Space Diversity**: How spread out are beliefs? (Vector: more diverse)
2. **Response Entropy**: How uncertain are responses? (Vector: more uncertain/realistic)
3. **Opinion Clustering**: Do beliefs cluster along social connections? (Vector: stronger homophily)
4. **Convergence**: How quickly do opinions stabilize? (Original: faster)
5. **Network Metrics**: Centrality, influence distribution, etc.

**Why it matters:** This is how we scientifically compare the two approaches. The metrics tell us which system better captures real-world dynamics.

---

### `/outputs/` - Where Results Live
**What's here:** All generated data and results.

**Structure:**
- `constraint_personas.json`: The 100 generated personas
- `simulation/`: Original system results
  - `final_personas.json`: Personas after simulation
  - `multi_round_survey_evolution.json`: How responses changed over rounds
  - `llm_interpretation_report.txt`: LLM analysis of beliefs
  - `influence_network_analysis.txt`: Who influenced whom
- `vector_beliefs_simulation/`: Vector system results (same structure)
- `visualizations/`: Network graphs, similarity heatmaps, etc.

**Why it matters:** This is where you'll find all the outputs. Want to see how opinions changed? Check `survey_evolution.json`. Want to see who influenced whom? Check `influence_network_analysis.txt`.

---

## The Workflow: How Everything Connects

Let me trace through what happens when you run a simulation:

### Step 1: Generate Personas
```
main.py → generation/constraint_based.py → data/research_data.py
```
- Uses demographic distributions
- Applies probabilistic constraints
- Creates 100 Persona objects

### Step 2: Add LLM Narratives (Optional)
```
main.py → llm/augmentation.py → OpenAI API
```
- Generates background stories
- Creates belief explanations
- Makes personas feel more human

### Step 3: Generate Embeddings
```
embeddings/generator.py → sentence-transformers
```
- Converts text (narrative + beliefs) to vectors
- Used for similarity calculations

### Step 4: Create Networks
```
networks/social_graph.py → Watts-Strogatz graph
networks/cognitive_graph.py → Similarity-weighted graph
networks/dual_graph.py → Combined model
```
- Graph A: Social connections (who talks to whom)
- Graph B: Cognitive similarity (how similar beliefs are)
- Dual: Only influence when both exist

### Step 5: Run Simulation
```
networks/simulation.py → networks/belief_updates.py
```
- Each round:
  1. Update beliefs based on neighbor influences
  2. Update embeddings (beliefs changed)
  3. Update cognitive graph (similarities changed)
  4. Generate survey responses
  5. Track changes

### Step 6: Analyze Results
```
metrics/scripts/calculate_metrics.py
analyze_influence_network.py
llm_interpretation_report.py
```
- Calculate metrics
- Analyze influence patterns
- Interpret beliefs with LLM

---

## Key Concepts You Need to Understand

### 1. Dual-Graph Model
This is the core innovation. Instead of just "who talks to whom," we have:
- **Graph A (Social)**: Binary connections (you're either friends or not)
- **Graph B (Cognitive)**: Continuous similarity (0.0-1.0 based on belief similarity)
- **Combined**: Influence weight = social × cognitive × 0.15

**Why this matters:** In real life, you're not influenced by everyone you know - only by people who think similarly. This model captures that.

### 2. Belief Updates
When Persona A influences Persona B:
1. Check if they're socially connected (Graph A)
2. Check if they're cognitively similar (Graph B)
3. If both: Calculate influence weight
4. Update B's beliefs toward A's beliefs (weighted by susceptibility)

**Susceptibility factors:**
- Poor mental health → more susceptible
- High social media → more susceptible
- Lower GPA → more susceptible

### 3. Survey Response Generation

**Original System:**
- Uses logit functions based on attributes + beliefs
- Deterministic probabilities
- Beliefs extracted from text via keyword matching

**Vector System:**
- Projects question embeddings onto belief vector
- Uses cosine similarity
- Softmax for probabilities
- Fully probabilistic

---

## Common Tasks & Where to Find Them

### "I want to see how beliefs changed over time"
→ Check `outputs/simulation/multi_round_survey_evolution.json` or `outputs/vector_beliefs_simulation/survey_evolution.json`

### "I want to understand the influence network"
→ Run `python3 analyze_influence_network.py` → Check `outputs/simulation/influence_network_analysis.txt`

### "I want to see what the LLM thinks personas believe"
→ Check `outputs/simulation/llm_interpretation_report.txt` or `outputs/vector_beliefs_simulation/llm_interpretation_report.txt`

### "I want to compare the two systems"
→ Run `python3 metrics/scripts/compare_systems.py` → Check `metrics/reports/system_comparison_report.txt`

### "I want to generate new personas"
→ Run `python3 main.py --method constraint --n_personas 100`

### "I want to visualize the networks"
→ Run `python3 networks/demo.py` → Check `outputs/visualizations/`

### "I want to see metrics comparisons"
→ Run `python3 metrics/scripts/visualize_metrics.py` → Check `metrics/visualizations/`

---

## The Experiment: Original vs Vector

We built two parallel systems to answer: **How should we represent human beliefs?**

### Original System (Text-Based)
- **Beliefs**: Text descriptions
- **Responses**: Deterministic logit-based
- **Updates**: Text → numeric → update → text
- **Strengths**: Interpretable, fast convergence, clear patterns
- **Weaknesses**: Lossy conversion, less flexible, can over-polarize

### Vector System (Continuous)
- **Beliefs**: 16D continuous vectors
- **Responses**: Probabilistic projection-based
- **Updates**: Direct vector operations
- **Strengths**: Efficient, flexible, realistic uncertainty, smoother transitions
- **Weaknesses**: Less interpretable, slower convergence, more variation

**The metrics show:** Vector system has more diverse beliefs, higher uncertainty (more realistic), stronger homophily, but slower convergence.

---

## File Naming Conventions

- `*_personas.json`: Generated personas
- `*_survey_evolution.json`: How responses changed over rounds
- `*_influence_stats.json`: Network statistics
- `*_metrics.json`: Calculated metrics
- `*_report.txt`: Human-readable reports
- `*_report.json`: Machine-readable reports

---

## Dependencies & Setup

**Core dependencies:**
- `numpy`, `scipy`, `pandas`: Data manipulation
- `networkx`: Graph operations
- `sentence-transformers`: Embedding generation
- `matplotlib`, `seaborn`: Visualizations
- `openai`: LLM augmentation (optional)
- `scikit-learn`: Machine learning utilities

**Setup:**
1. Install: `pip install -r requirements.txt`
2. (Optional) Add OpenAI API key to `config/.env` for LLM features
3. Run: `python3 main.py` or `python3 run_simulation.py`

---

## Reading Order for Newcomers

If you're completely new, here's a suggested reading order:

1. **Start here:** `README.md` - High-level project overview
2. **Understand methodology:** `persona_methodology_guide.md` - How we generate personas
3. **See the pipeline:** `IMPLEMENTATION_STATUS.md` - What's implemented
4. **Understand networks:** `TOPOLOGY_EMBEDDINGS_GUIDE.md` - How the dual-graph works
5. **See simulations:** `SIMULATION_GUIDE.md` - How simulations run
6. **Compare systems:** `metrics/METRICS_GUIDE.md` - Metrics comparison
7. **Explore code:** Start with `main.py` or `run_simulation.py`

---

## Quick Reference: What Each Script Does

- `main.py`: Generate personas from scratch
- `run_simulation.py`: Run original system simulation
- `experiments/vector_beliefs/run_vector_simulation.py`: Run vector system simulation
- `analyze_influence_network.py`: Analyze who influenced whom
- `llm_interpretation_report_original.py`: LLM interpretation of text beliefs
- `experiments/vector_beliefs/llm_interpretation_report.py`: LLM interpretation of vectors
- `metrics/scripts/calculate_metrics.py`: Calculate all metrics
- `metrics/scripts/visualize_metrics.py`: Generate comparison plots
- `metrics/scripts/compare_systems.py`: Generate comparison report
- `networks/demo.py`: Visualize network structures
- `visualize_simulation.py`: Visualize simulation results
- `experiments/vector_beliefs/visualize_vector_simulation.py`: Visualize vector results

---

## Common Questions

**Q: Why two systems?**
A: To compare different belief representations. Text is interpretable; vectors are efficient and probabilistic.

**Q: What's the "influence reduction factor" (0.15)?**
A: It simulates real-world resistance. Even if you're connected and similar, only 15% of potential influence actually happens. People resist change!

**Q: Why Watts-Strogatz for social graph?**
A: It creates small-world networks (like real social networks) - high clustering but short path lengths.

**Q: What's "cognitive similarity"?**
A: Cosine similarity between embeddings. Measures how similar two personas' beliefs are based on their narratives and beliefs.

**Q: How do I know which system is better?**
A: Check `metrics/reports/system_comparison_report.txt`. It depends on your goals: interpretability vs. realism vs. efficiency.

**Q: Can I add more survey questions?**
A: Yes! Add to `surveys/questions.py` and update response generation in `surveys/responses.py` (original) or `experiments/vector_beliefs/surveys/vector_responses.py` (vector).

**Q: Can I change the influence model?**
A: Yes! Modify `networks/belief_updates.py` (original) or `experiments/vector_beliefs/networks/vector_belief_updates.py` (vector).

---

## The Mental Model

Think of this project as having three layers:

1. **Data Layer**: Personas, their attributes, beliefs, responses
2. **Network Layer**: Social connections, cognitive similarities, influence weights
3. **Dynamics Layer**: Belief updates, opinion changes, convergence

Everything flows: Data → Networks → Dynamics → Updated Data → Updated Networks → ...

---

## Where to Go Next

**If you want to understand the methodology:**
→ Read `persona_methodology_guide.md` and `TOPOLOGY_EMBEDDINGS_GUIDE.md`

**If you want to see results:**
→ Check `outputs/simulation/` and `outputs/vector_beliefs_simulation/`

**If you want to compare systems:**
→ Read `metrics/reports/system_comparison_report.txt` and check `metrics/visualizations/`

**If you want to modify the system:**
→ Start with `networks/belief_updates.py` (original) or `experiments/vector_beliefs/networks/vector_belief_updates.py` (vector)

**If you want to add features:**
→ Look at `IMPLEMENTATION_STATUS.md` to see what's done and what could be added

---

## Final Thoughts

This project is essentially asking: **How do we computationally represent human beliefs and social influence?**

The original system says: "Use text - it's interpretable."
The vector system says: "Use vectors - it's more realistic."

Both are valid. The metrics help us understand when to use which.

The codebase is modular by design - you can swap out components, try different influence models, add new questions, or experiment with different belief representations.

Welcome to the project! Explore, experiment, and let me know what you discover.

---

*Last updated: December 2024*
*For questions or issues, check the individual guide files in each folder.*

