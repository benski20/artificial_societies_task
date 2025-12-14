"""
Use LLM to interpret text-based beliefs for sampled personas and report their responses.

This script:
1. Loads original personas (with text beliefs) and survey evolution data
2. Samples 5 random personas
3. Uses LLM to interpret their text-based beliefs
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
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from persona.persona import Persona
from surveys.questions import SURVEY_QUESTIONS

# LLM client setup
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / 'config' / '.env'
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


def load_personas_from_json(filepath: str) -> List[Persona]:
    """Load personas from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    personas = []
    for p_dict in data:
        persona = Persona(
            age=p_dict['age'],
            gender=p_dict['gender'],
            race=p_dict['race'],
            sexual_orientation=p_dict['sexual_orientation'],
            family_income=p_dict['family_income'],
            income_percentile=p_dict.get('income_percentile'),
            income_quintile=p_dict.get('income_quintile'),
            first_gen_college=p_dict.get('first_gen_college', False),
            gpa=p_dict.get('gpa', 0.0),
            sports_participation=p_dict.get('sports_participation', False),
            mental_health=p_dict.get('mental_health', 'Fair'),
            social_media_intensity=p_dict.get('social_media_intensity', 'Moderate'),
            college_intention=p_dict.get('college_intention', 'Unsure'),
            narrative=p_dict.get('narrative'),
            beliefs=p_dict.get('beliefs', {}),
            survey_responses=p_dict.get('survey_responses', {})
        )
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


def interpret_text_beliefs_llm(persona: Persona,
                               llm_client) -> Dict[str, str]:
    """
    Use LLM to interpret text-based beliefs and generate natural language descriptions.
    
    Args:
        persona: Persona object with text beliefs
        llm_client: LLM client
        
    Returns:
        Dictionary mapping question_id -> belief description
    """
    # Get text beliefs
    beliefs_text = persona.beliefs if persona.beliefs else {}
    
    # Map question IDs to belief topics
    question_to_topic = {
        'college_importance': 'college importance and future plans',
        'social_media_stress': 'social media impact on daily life',
        'school_start_times': 'school start times and sleep'
    }
    
    # Create prompt for LLM
    prompt = f"""You are analyzing a high school student's belief system based on their stated beliefs.

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

STATED BELIEFS (Text Descriptions):
{json.dumps(beliefs_text, indent=2) if beliefs_text else "No explicit beliefs stated"}

NARRATIVE/BACKGROUND:
{persona.narrative[:500] if persona.narrative else "No narrative available"}

SURVEY RESPONSES:
{json.dumps(persona.survey_responses, indent=2) if persona.survey_responses else "No responses recorded"}

SURVEY QUESTIONS:
{json.dumps(SURVEY_QUESTIONS, indent=2)}

TASK:
Based on the stated beliefs, demographics, narrative, and survey responses, generate natural language 
descriptions of this student's beliefs about each of the three topics. Be specific and interpret what 
the text beliefs reveal about their internal belief state.

For each question, provide:
1. A natural language description of their belief (2-3 sentences) based on the stated beliefs
2. How certain/strong their belief appears to be (high/medium/low)
3. How their demographics might influence these beliefs
4. Whether the stated beliefs align with their survey responses

Format your response as JSON with the following structure:
{{
    "college_importance": {{
        "belief_description": "Natural language description of their belief about college importance",
        "certainty": "high/medium/low",
        "demographic_influence": "Brief note on how demographics might shape this belief",
        "alignment_with_response": "Whether stated belief aligns with survey response"
    }},
    "social_media_stress": {{
        "belief_description": "Natural language description of their belief about social media stress",
        "certainty": "high/medium/low",
        "demographic_influence": "Brief note on how demographics might shape this belief",
        "alignment_with_response": "Whether stated belief aligns with survey response"
    }},
    "school_start_times": {{
        "belief_description": "Natural language description of their belief about school start times",
        "certainty": "high/medium/low",
        "demographic_influence": "Brief note on how demographics might shape this belief",
        "alignment_with_response": "Whether stated belief aligns with survey response"
    }}
}}

Be concise but informative. Focus on what the text beliefs reveal about this student's internal belief state."""

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
        print(f"Error interpreting text beliefs with LLM: {e}")
        return {
            qid: {
                "belief_description": f"Error: Could not interpret beliefs ({str(e)})",
                "certainty": "unknown",
                "demographic_influence": "unknown",
                "alignment_with_response": "unknown"
            }
            for qid in SURVEY_QUESTIONS.keys()
        }


def generate_interpretation_report(personas: List[Persona],
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
        'system_type': 'original_text_based',
        'personas': []
    }
    
    for idx, (persona_idx, persona) in enumerate(zip(sample_indices, sample_personas)):
        print(f"\nProcessing persona {idx + 1}/{len(sample_personas)} (index {persona_idx})...")
        
        # Get responses over time (from aggregate data)
        responses_over_time = get_persona_responses_over_time(persona_idx, survey_evolution)
        
        # Interpret text beliefs with LLM
        print(f"  Interpreting text beliefs with LLM...")
        llm_interpretations = interpret_text_beliefs_llm(persona, llm_client)
        
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
            'text_beliefs': persona.beliefs if persona.beliefs else {},
            'narrative': persona.narrative[:500] if persona.narrative else None,
            'llm_interpretations': llm_interpretations,
            'final_responses': persona.survey_responses if persona.survey_responses else {},
            'responses_over_time': responses_over_time
        }
        
        report_data['personas'].append(persona_data)
        print(f"  ✓ Completed persona {idx + 1}")
    
    return report_data


def format_report_for_display(report: Dict) -> str:
    """Format report as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("LLM INTERPRETATION REPORT: TEXT-BASED BELIEFS")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {report['generated_at']}")
    lines.append(f"System Type: {report.get('system_type', 'original_text_based')}")
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
        
        # Text Beliefs
        if persona_data.get('text_beliefs'):
            lines.append("\nSTATED TEXT BELIEFS:")
            for topic, belief_text in persona_data['text_beliefs'].items():
                lines.append(f"  {topic}: {belief_text}")
        else:
            lines.append("\nSTATED TEXT BELIEFS: None")
        
        # Narrative
        if persona_data.get('narrative'):
            lines.append("\nNARRATIVE/BACKGROUND:")
            lines.append(f"  {persona_data['narrative'][:300]}...")
        
        # LLM Interpretations
        lines.append("\nLLM INTERPRETATIONS OF TEXT BELIEFS:")
        interpretations = persona_data['llm_interpretations']
        for qid, interp in interpretations.items():
            question_text = SURVEY_QUESTIONS[qid]['question']
            lines.append(f"\n  {question_text}:")
            lines.append(f"    Belief: {interp.get('belief_description', 'N/A')}")
            lines.append(f"    Certainty: {interp.get('certainty', 'N/A')}")
            lines.append(f"    Demographic Influence: {interp.get('demographic_influence', 'N/A')}")
            if 'alignment_with_response' in interp:
                lines.append(f"    Alignment with Response: {interp.get('alignment_with_response', 'N/A')}")
        
        # Final Responses
        if persona_data.get('final_responses'):
            lines.append("\nFINAL SURVEY RESPONSES (Round 5):")
            for qid, response in persona_data['final_responses'].items():
                question_text = SURVEY_QUESTIONS[qid]['question']
                lines.append(f"  {question_text}: {response}")
        else:
            lines.append("\nFINAL SURVEY RESPONSES: Not available")
        
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
    print("LLM INTERPRETATION REPORT GENERATOR (ORIGINAL TEXT-BASED SYSTEM)")
    print("=" * 80)
    
    # File paths
    personas_file = Path('outputs/simulation/final_personas.json')
    survey_evolution_file = Path('outputs/simulation/multi_round_survey_evolution.json')
    output_json = Path('outputs/simulation/llm_interpretation_report.json')
    output_txt = Path('outputs/simulation/llm_interpretation_report.txt')
    
    # Load data
    print(f"\nLoading personas from {personas_file}...")
    personas = load_personas_from_json(str(personas_file))
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
        random_seed=42  # Same seed for comparison
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
    print(f"\nNote: This uses the same random seed (42) as the vector-based report")
    print(f"for direct comparison of the same personas across both systems.")


if __name__ == '__main__':
    main()

