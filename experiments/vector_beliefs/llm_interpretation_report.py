"""
Use LLM to interpret belief vectors for sampled personas and report their responses.

This script:
1. Loads vector personas and survey evolution data
2. Samples 5 random personas
3. Uses LLM to interpret their belief vectors
4. Reports their responses at each timestep
5. Saves results for human verification
"""

import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.vector_beliefs.persona.vector_persona import VectorPersona
from surveys.questions import SURVEY_QUESTIONS

# LLM client setup
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / 'config' / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def get_llm_client():
    """Get OpenAI LLM client."""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


def load_vector_personas(filepath: str) -> List[VectorPersona]:
    """Load vector personas from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    personas = []
    for p_dict in data:
        persona = VectorPersona.from_dict(p_dict)
        personas.append(persona)
    
    return personas


def load_survey_evolution(filepath: str) -> Dict:
    """Load survey evolution data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_persona_responses_over_time(persona_idx: int,
                                   survey_evolution: Dict) -> Dict:
    """
    Get responses for a specific persona across all rounds.
    
    Note: This reconstructs individual responses from aggregate data.
    Since we only have aggregate distributions, we'll note this limitation.
    """
    responses_over_time = {}
    
    for question_id, evolution in survey_evolution.items():
        responses_over_time[question_id] = []
        for round_data in evolution:
            responses_over_time[question_id].append({
                'round': round_data['round'],
                'distribution': round_data['distribution'],
                'total': round_data['total']
            })
    
    return responses_over_time


def interpret_belief_vector_llm(persona: VectorPersona,
                                llm_client,
                                round_num: Optional[int] = None) -> Dict[str, str]:
    """
    Use LLM to interpret belief vector and generate natural language descriptions.
    
    Args:
        persona: VectorPersona object
        llm_client: LLM client
        round_num: Optional round number for context
        
    Returns:
        Dictionary mapping question_id -> belief description
    """
    belief_vector = persona.get_belief_vector()
    
    # Get response probabilities for context
    response_context = {}
    for qid in SURVEY_QUESTIONS.keys():
        if qid in persona.response_probabilities:
            probs = persona.response_probabilities[qid]
            options = SURVEY_QUESTIONS[qid]['options']
            max_idx = np.argmax(probs)
            response_context[qid] = {
                'most_likely_response': options[max_idx],
                'confidence': float(probs[max_idx]),
                'all_probabilities': {
                    opt: float(p) for opt, p in zip(options, probs)
                }
            }
        elif qid in persona.survey_responses:
            response_context[qid] = {
                'response': persona.survey_responses[qid]
            }
    
    # Create prompt for LLM
    prompt = f"""You are analyzing a high school student's belief system represented as a 16-dimensional vector.

PERSONA DEMOGRAPHICS:
- Age: {persona.age} years old
- Gender: {persona.gender}
- Race/Ethnicity: {persona.race}
- Sexual Orientation: {persona.sexual_orientation}
- Family Income: ${persona.family_income:,.0f}
- GPA: {persona.gpa:.2f}
- Mental Health: {persona.mental_health}
- Social Media Intensity: {persona.social_media_intensity}
- College Intention: {persona.college_intention}
- Sports Participation: {persona.sports_participation}
- First Generation College: {persona.first_gen_college}

BELIEF VECTOR (16 dimensions):
{belief_vector.tolist()}

SURVEY RESPONSES AND PROBABILITIES:
{json.dumps(response_context, indent=2)}

SURVEY QUESTIONS:
{json.dumps(SURVEY_QUESTIONS, indent=2)}

TASK:
Based on the belief vector, demographics, and survey responses, generate natural language descriptions 
of this student's beliefs about each of the three topics. Be specific and interpret what the vector 
dimensions might represent in the context of a high school student's worldview.

For each question, provide:
1. A natural language description of their belief (2-3 sentences)
2. What their response probabilities suggest about their certainty/uncertainty
3. How their demographics might influence these beliefs

Format your response as JSON with the following structure:
{{
    "college_importance": {{
        "belief_description": "Natural language description of their belief about college importance",
        "certainty": "high/medium/low",
        "demographic_influence": "Brief note on how demographics might shape this belief"
    }},
    "social_media_stress": {{
        "belief_description": "Natural language description of their belief about social media stress",
        "certainty": "high/medium/low",
        "demographic_influence": "Brief note on how demographics might shape this belief"
    }},
    "school_start_times": {{
        "belief_description": "Natural language description of their belief about school start times",
        "certainty": "high/medium/low",
        "demographic_influence": "Brief note on how demographics might shape this belief"
    }}
}}

Be concise but informative. Focus on what the belief vector and responses reveal about this student's 
internal belief state."""

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in psychology and education, analyzing student belief systems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        
        # Try to extract JSON from response
        # Sometimes LLM wraps JSON in markdown code blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        
        beliefs = json.loads(content)
        return beliefs
        
    except Exception as e:
        print(f"Error interpreting belief vector with LLM: {e}")
        return {
            qid: {
                "belief_description": f"Error: Could not interpret belief vector ({str(e)})",
                "certainty": "unknown",
                "demographic_influence": "unknown"
            }
            for qid in SURVEY_QUESTIONS.keys()
        }


def generate_interpretation_report(personas: List[VectorPersona],
                                  survey_evolution: Dict,
                                  sample_size: int = 5,
                                  random_seed: int = 42) -> Dict:
    """
    Generate LLM interpretation report for sampled personas.
    
    Args:
        personas: List of all personas
        survey_evolution: Survey evolution data
        sample_size: Number of personas to sample
        random_seed: Random seed for sampling
        
    Returns:
        Report dictionary
    """
    # Sample random personas
    np.random.seed(random_seed)
    sample_indices = np.random.choice(len(personas), min(sample_size, len(personas)), replace=False)
    sample_personas = [personas[i] for i in sample_indices]
    
    print(f"Sampling {len(sample_personas)} personas (indices: {sample_indices.tolist()})")
    
    # Get LLM client
    try:
        llm_client = get_llm_client()
        print("✓ LLM client initialized")
    except Exception as e:
        print(f"Error initializing LLM client: {e}")
        print("Please ensure OPENAI_API_KEY is set in config/.env")
        return None
    
    # Process each sampled persona
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'total_personas': len(personas),
        'sample_size': len(sample_personas),
        'random_seed': random_seed,
        'personas': []
    }
    
    for idx, (persona_idx, persona) in enumerate(zip(sample_indices, sample_personas)):
        print(f"\nProcessing persona {idx + 1}/{len(sample_personas)} (index {persona_idx})...")
        
        # Get responses over time (from aggregate data)
        # Note: We can't get individual responses, but we can show aggregate distributions
        responses_over_time = get_persona_responses_over_time(persona_idx, survey_evolution)
        
        # Interpret belief vector with LLM
        print(f"  Interpreting belief vector with LLM...")
        llm_interpretations = interpret_belief_vector_llm(persona, llm_client)
        
        # Compile persona data
        persona_data = {
            'persona_index': int(persona_idx),
            'demographics': {
                'age': persona.age,
                'gender': persona.gender,
                'race': persona.race,
                'sexual_orientation': persona.sexual_orientation,
                'family_income': float(persona.family_income),
                'gpa': float(persona.gpa),
                'mental_health': persona.mental_health,
                'social_media_intensity': persona.social_media_intensity,
                'college_intention': persona.college_intention,
                'sports_participation': persona.sports_participation,
                'first_gen_college': persona.first_gen_college
            },
            'belief_vector': persona.get_belief_vector().tolist(),
            'llm_interpretations': llm_interpretations,
            'final_responses': persona.survey_responses,
            'final_response_probabilities': {
                qid: {
                    opt: float(p) for opt, p in zip(SURVEY_QUESTIONS[qid]['options'], probs)
                }
                for qid, probs in persona.response_probabilities.items()
            },
            'responses_over_time': responses_over_time
        }
        
        report_data['personas'].append(persona_data)
        print(f"  ✓ Completed persona {idx + 1}")
    
    return report_data


def format_report_for_display(report: Dict) -> str:
    """Format report as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("LLM INTERPRETATION REPORT: BELIEF VECTORS TO TEXT")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {report['generated_at']}")
    lines.append(f"Total Personas: {report['total_personas']}")
    lines.append(f"Sample Size: {report['sample_size']}")
    lines.append(f"Random Seed: {report['random_seed']}")
    
    for idx, persona_data in enumerate(report['personas'], 1):
        lines.append("\n" + "=" * 80)
        lines.append(f"PERSONA {idx} (Index {persona_data['persona_index']})")
        lines.append("=" * 80)
        
        # Demographics
        demo = persona_data['demographics']
        lines.append("\nDEMOGRAPHICS:")
        lines.append(f"  Age: {demo['age']}, Gender: {demo['gender']}, Race: {demo['race']}")
        lines.append(f"  GPA: {demo['gpa']:.2f}, Mental Health: {demo['mental_health']}")
        lines.append(f"  Social Media: {demo['social_media_intensity']}")
        lines.append(f"  College Intention: {demo['college_intention']}")
        lines.append(f"  Family Income: ${demo['family_income']:,.0f}")
        
        # LLM Interpretations
        lines.append("\nLLM INTERPRETATIONS OF BELIEF VECTOR:")
        interpretations = persona_data['llm_interpretations']
        for qid, interp in interpretations.items():
            question_text = SURVEY_QUESTIONS[qid]['question']
            lines.append(f"\n  {question_text}:")
            lines.append(f"    Belief: {interp.get('belief_description', 'N/A')}")
            lines.append(f"    Certainty: {interp.get('certainty', 'N/A')}")
            lines.append(f"    Demographic Influence: {interp.get('demographic_influence', 'N/A')}")
        
        # Final Responses
        lines.append("\nFINAL SURVEY RESPONSES (Round 5):")
        for qid, response in persona_data['final_responses'].items():
            question_text = SURVEY_QUESTIONS[qid]['question']
            lines.append(f"  {question_text}: {response}")
        
        # Response Probabilities
        lines.append("\nFINAL RESPONSE PROBABILITIES:")
        for qid, probs in persona_data['final_response_probabilities'].items():
            question_text = SURVEY_QUESTIONS[qid]['question']
            lines.append(f"  {question_text}:")
            for option, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"    {option}: {prob:.1%}")
        
        # Responses Over Time
        lines.append("\nRESPONSES OVER TIME (Aggregate Distributions):")
        for qid, rounds_data in persona_data['responses_over_time'].items():
            question_text = SURVEY_QUESTIONS[qid]['question']
            lines.append(f"\n  {question_text}:")
            for round_data in rounds_data:
                lines.append(f"    Round {round_data['round']}:")
                for option, count in sorted(round_data['distribution'].items(), key=lambda x: x[1], reverse=True):
                    pct = count / round_data['total'] * 100
                    lines.append(f"      {option}: {count} ({pct:.1f}%)")
    
    return "\n".join(lines)


def main():
    """Main function."""
    print("=" * 80)
    print("LLM INTERPRETATION REPORT GENERATOR")
    print("=" * 80)
    
    # File paths
    personas_file = Path('outputs/vector_beliefs_simulation/final_personas.json')
    survey_evolution_file = Path('outputs/vector_beliefs_simulation/survey_evolution.json')
    output_json = Path('outputs/vector_beliefs_simulation/llm_interpretation_report.json')
    output_txt = Path('outputs/vector_beliefs_simulation/llm_interpretation_report.txt')
    
    # Load data
    print(f"\nLoading personas from {personas_file}...")
    personas = load_vector_personas(str(personas_file))
    print(f"✓ Loaded {len(personas)} personas")
    
    print(f"\nLoading survey evolution from {survey_evolution_file}...")
    survey_evolution = load_survey_evolution(str(survey_evolution_file))
    print(f"✓ Loaded survey evolution data")
    
    # Generate report
    print(f"\nGenerating LLM interpretation report...")
    report = generate_interpretation_report(
        personas,
        survey_evolution,
        sample_size=5,
        random_seed=42
    )
    
    if report is None:
        print("\nError: Could not generate report")
        return
    
    # Save JSON report
    print(f"\nSaving JSON report to {output_json}...")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved JSON report")
    
    # Save human-readable report
    print(f"\nSaving human-readable report to {output_txt}...")
    formatted_report = format_report_for_display(report)
    with open(output_txt, 'w') as f:
        f.write(formatted_report)
    print(f"✓ Saved text report")
    
    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nReports saved to:")
    print(f"  - {output_json}")
    print(f"  - {output_txt}")
    print(f"\nYou can now compare the LLM interpretations with the actual responses!")


if __name__ == '__main__':
    main()

