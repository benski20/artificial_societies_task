"""
Analyze influence network to show how certain personas influenced others.

This script:
1. Loads personas and reconstructs dual-graph model
2. Samples 5-10 personas to analyze
3. Tracks their influence on others over time
4. Uses LLM to interpret influence patterns
5. Generates report showing influence relationships
"""

import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from persona.persona import Persona
from experiments.vector_beliefs.persona.vector_persona import VectorPersona
from surveys.questions import SURVEY_QUESTIONS

# LLM client setup
import os
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
        raise ImportError("OpenAI package not installed")
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    return OpenAI(api_key=api_key)


def load_personas(filepath: str, system_type: str = 'original') -> List:
    """Load personas from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if system_type == 'vector':
        personas = []
        for p_dict in data:
            persona = VectorPersona.from_dict(p_dict)
            personas.append(persona)
    else:
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


def reconstruct_dual_graph(personas: List, system_type: str = 'original'):
    """
    Reconstruct dual-graph model from final personas.
    
    Returns dual_graph model and embeddings.
    """
    from networks.dual_graph import create_dual_graph_model
    from experiments.vector_beliefs.embeddings.belief_embeddings import generate_embeddings_from_vectors
    
    if system_type == 'vector':
        # Generate embeddings from belief vectors
        embeddings = generate_embeddings_from_vectors(personas, embedding_dim=384)
    else:
        # Generate embeddings from text (narrative + beliefs)
        from embeddings.generator import generate_embeddings
        embeddings = generate_embeddings(personas, show_progress=False)
    
    # Create dual-graph model
    dual_model = create_dual_graph_model(
        len(personas),
        embeddings,
        social_k=4,
        social_p=0.2,
        influence_reduction_factor=0.15,
        seed=42
    )
    
    return dual_model, embeddings


def analyze_influence_network(dual_model, 
                              personas: List,
                              sample_indices: List[int],
                              survey_evolution: Dict) -> Dict:
    """
    Analyze influence network for sampled personas.
    
    For each sampled persona, find:
    - Who they influence (outgoing)
    - Who influences them (incoming)
    - Influence strength
    - How beliefs changed
    """
    influence_analysis = {}
    
    for persona_idx in sample_indices:
        persona = personas[persona_idx]
        
        # Get outgoing influences (who this persona influences)
        outgoing = dual_model.get_influence_neighbors(persona_idx)
        # Note: get_influence_neighbors returns (neighbor_idx, weight) where neighbor influences persona_idx
        # We need the reverse: who does persona_idx influence?
        
        # Get incoming influences (who influences this persona)
        incoming = []
        for other_idx in range(len(personas)):
            if other_idx != persona_idx:
                weight = dual_model.get_influence_weight(other_idx, persona_idx)
                if weight > 0:
                    incoming.append((other_idx, weight))
        
        # Get outgoing influences (who persona_idx influences)
        outgoing = []
        for other_idx in range(len(personas)):
            if other_idx != persona_idx:
                weight = dual_model.get_influence_weight(persona_idx, other_idx)
                if weight > 0:
                    outgoing.append((other_idx, weight))
        
        # Sort by influence strength
        incoming.sort(key=lambda x: x[1], reverse=True)
        outgoing.sort(key=lambda x: x[1], reverse=True)
        
        # Get persona demographics
        demo = {
            'age': persona.age,
            'gender': persona.gender,
            'race': persona.race,
            'gpa': float(persona.gpa),
            'mental_health': persona.mental_health,
            'social_media_intensity': persona.social_media_intensity,
            'college_intention': persona.college_intention
        }
        
        # Get responses over time
        responses_over_time = {}
        for qid in SURVEY_QUESTIONS.keys():
            responses_over_time[qid] = []
            if qid in survey_evolution:
                for round_data in survey_evolution[qid]:
                    responses_over_time[qid].append({
                        'round': round_data['round'],
                        'distribution': round_data['distribution']
                    })
        
        influence_analysis[persona_idx] = {
            'persona_index': persona_idx,
            'demographics': demo,
            'incoming_influences': [
                {
                    'influencer_idx': idx,
                    'weight': float(weight),
                    'influencer_demo': {
                        'age': personas[idx].age,
                        'gender': personas[idx].gender,
                        'race': personas[idx].race,
                        'gpa': float(personas[idx].gpa)
                    }
                }
                for idx, weight in incoming[:10]  # Top 10 influencers
            ],
            'outgoing_influences': [
                {
                    'influenced_idx': idx,
                    'weight': float(weight),
                    'influenced_demo': {
                        'age': personas[idx].age,
                        'gender': personas[idx].gender,
                        'race': personas[idx].race,
                        'gpa': float(personas[idx].gpa)
                    }
                }
                for idx, weight in outgoing[:10]  # Top 10 influenced
            ],
            'total_incoming': len(incoming),
            'total_outgoing': len(outgoing),
            'avg_incoming_weight': float(np.mean([w for _, w in incoming])) if incoming else 0.0,
            'avg_outgoing_weight': float(np.mean([w for _, w in outgoing])) if outgoing else 0.0,
            'responses_over_time': responses_over_time
        }
    
    return influence_analysis


def interpret_influence_patterns_llm(influence_analysis: Dict,
                                    personas: List,
                                    llm_client) -> Dict:
    """
    Use LLM to interpret influence patterns.
    
    For each sampled persona, analyze:
    - Why they might be influential (or influenced)
    - What makes their connections strong
    - Patterns in their influence network
    """
    interpretations = {}
    
    for persona_idx, analysis in influence_analysis.items():
        persona = personas[persona_idx]
        
        # Prepare influence summary
        top_influencers = analysis['incoming_influences'][:5]
        top_influenced = analysis['outgoing_influences'][:5]
        
        prompt = f"""Analyze the influence network position of a high school student persona.

PERSONA DEMOGRAPHICS:
- Age: {persona.age}, Gender: {persona.gender}, Race: {persona.race}
- GPA: {persona.gpa:.2f}, Mental Health: {persona.mental_health}
- Social Media: {persona.social_media_intensity}
- College Intention: {persona.college_intention}

INFLUENCE NETWORK POSITION:
- Total incoming influences: {analysis['total_incoming']} (people who influence this persona)
- Total outgoing influences: {analysis['total_outgoing']} (people this persona influences)
- Average incoming influence weight: {analysis['avg_incoming_weight']:.4f}
- Average outgoing influence weight: {analysis['avg_outgoing_weight']:.4f}

TOP 5 INFLUENCERS (who influence this persona):
{json.dumps(top_influencers, indent=2)}

TOP 5 INFLUENCED (who this persona influences):
{json.dumps(top_influenced, indent=2)}

TASK:
Analyze this persona's position in the influence network. Provide insights on:
1. Why this persona might be influential (or not) - what traits make them influential?
2. What patterns do you see in who influences them and who they influence?
3. Are there demographic similarities in their influence connections?
4. What does their influence network position suggest about their role in the social system?

Format as JSON:
{{
    "influence_role": "Description of their role (e.g., 'influencer', 'influenced', 'bridge', 'isolated')",
    "influence_strength": "strong/medium/weak",
    "key_traits": ["trait1", "trait2", "trait3"],
    "influence_patterns": "Description of patterns in their connections",
    "demographic_clustering": "Whether they connect with similar demographics",
    "social_position": "Description of their position in the network"
}}"""

        try:
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in social network analysis and adolescent psychology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            interpretation = json.loads(content)
            interpretations[persona_idx] = interpretation
            
        except Exception as e:
            print(f"Error interpreting influence for persona {persona_idx}: {e}")
            interpretations[persona_idx] = {
                "influence_role": "Error",
                "influence_strength": "unknown",
                "key_traits": [],
                "influence_patterns": f"Error: {str(e)}",
                "demographic_clustering": "unknown",
                "social_position": "unknown"
            }
    
    return interpretations


def generate_influence_report(personas: List,
                             survey_evolution: Dict,
                             system_type: str = 'original',
                             sample_size: int = 8,
                             random_seed: int = 42) -> Dict:
    """
    Generate influence network analysis report.
    
    Args:
        personas: List of personas
        survey_evolution: Survey evolution data
        system_type: 'original' or 'vector'
        sample_size: Number of personas to analyze
        random_seed: Random seed
        
    Returns:
        Report dictionary
    """
    print(f"Reconstructing dual-graph model ({system_type} system)...")
    dual_model, embeddings = reconstruct_dual_graph(personas, system_type)
    print(f"✓ Dual-graph model created")
    
    # Sample personas
    np.random.seed(random_seed)
    sample_indices = np.random.choice(len(personas), min(sample_size, len(personas)), replace=False).tolist()
    print(f"Sampling {len(sample_indices)} personas: {sample_indices}")
    
    # Analyze influence network
    print("Analyzing influence network...")
    influence_analysis = analyze_influence_network(dual_model, personas, sample_indices, survey_evolution)
    print(f"✓ Influence analysis complete")
    
    # Get LLM interpretations
    print("Getting LLM interpretations...")
    try:
        llm_client = get_llm_client()
        llm_interpretations = interpret_influence_patterns_llm(influence_analysis, personas, llm_client)
        print(f"✓ LLM interpretations complete")
    except Exception as e:
        print(f"Warning: Could not get LLM interpretations: {e}")
        llm_interpretations = {}
    
    # Compile report
    report = {
        'generated_at': datetime.now().isoformat(),
        'system_type': system_type,
        'total_personas': len(personas),
        'sample_size': len(sample_indices),
        'random_seed': random_seed,
        'network_statistics': {
            'total_edges': dual_model.social_graph.number_of_edges(),
            'social_density': float(dual_model.statistics()['social_density']),
            'avg_influence_weight': float(dual_model.statistics()['avg_influence_weight']),
            'active_influences': dual_model.statistics()['active_influences']
        },
        'personas': []
    }
    
    for persona_idx in sample_indices:
        analysis = influence_analysis[persona_idx]
        interpretation = llm_interpretations.get(persona_idx, {})
        
        report['personas'].append({
            **analysis,
            'llm_interpretation': interpretation
        })
    
    return report


def format_influence_report(report: Dict) -> str:
    """Format influence report as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("INFLUENCE NETWORK ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {report['generated_at']}")
    lines.append(f"System Type: {report['system_type']}")
    lines.append(f"Total Personas: {report['total_personas']}")
    lines.append(f"Sample Size: {report['sample_size']}")
    
    stats = report['network_statistics']
    lines.append(f"\nNETWORK STATISTICS:")
    lines.append(f"  Social Edges: {stats['total_edges']}")
    lines.append(f"  Social Density: {stats['social_density']:.4f}")
    lines.append(f"  Average Influence Weight: {stats['avg_influence_weight']:.4f}")
    lines.append(f"  Active Influences: {stats['active_influences']}")
    
    for idx, persona_data in enumerate(report['personas'], 1):
        lines.append("\n" + "=" * 80)
        lines.append(f"PERSONA {idx} (Index {persona_data['persona_index']})")
        lines.append("=" * 80)
        
        demo = persona_data['demographics']
        lines.append("\nDEMOGRAPHICS:")
        lines.append(f"  Age: {demo['age']}, Gender: {demo['gender']}, Race: {demo['race']}")
        lines.append(f"  GPA: {demo['gpa']:.2f}, Mental Health: {demo['mental_health']}")
        lines.append(f"  Social Media: {demo['social_media_intensity']}")
        
        lines.append("\nINFLUENCE NETWORK POSITION:")
        lines.append(f"  Incoming Influences: {persona_data['total_incoming']} (people who influence this persona)")
        lines.append(f"  Outgoing Influences: {persona_data['total_outgoing']} (people this persona influences)")
        lines.append(f"  Avg Incoming Weight: {persona_data['avg_incoming_weight']:.4f}")
        lines.append(f"  Avg Outgoing Weight: {persona_data['avg_outgoing_weight']:.4f}")
        
        # LLM Interpretation
        if persona_data.get('llm_interpretation'):
            interp = persona_data['llm_interpretation']
            lines.append("\nLLM INTERPRETATION:")
            lines.append(f"  Role: {interp.get('influence_role', 'N/A')}")
            lines.append(f"  Strength: {interp.get('influence_strength', 'N/A')}")
            lines.append(f"  Key Traits: {', '.join(interp.get('key_traits', []))}")
            lines.append(f"  Patterns: {interp.get('influence_patterns', 'N/A')}")
            lines.append(f"  Social Position: {interp.get('social_position', 'N/A')}")
        
        # Top Influencers
        lines.append("\nTOP 5 INFLUENCERS (who influence this persona):")
        for i, infl in enumerate(persona_data['incoming_influences'][:5], 1):
            demo_infl = infl['influencer_demo']
            lines.append(f"  {i}. Persona {infl['influencer_idx']}: weight={infl['weight']:.4f}")
            lines.append(f"     ({demo_infl['age']}, {demo_infl['gender']}, {demo_infl['race']}, GPA: {demo_infl['gpa']:.2f})")
        
        # Top Influenced
        lines.append("\nTOP 5 INFLUENCED (who this persona influences):")
        for i, infl in enumerate(persona_data['outgoing_influences'][:5], 1):
            demo_infl = infl['influenced_demo']
            lines.append(f"  {i}. Persona {infl['influenced_idx']}: weight={infl['weight']:.4f}")
            lines.append(f"     ({demo_infl['age']}, {demo_infl['gender']}, {demo_infl['race']}, GPA: {demo_infl['gpa']:.2f})")
        
        # Responses over time
        lines.append("\nRESPONSES OVER TIME (Aggregate):")
        for qid, rounds_data in persona_data['responses_over_time'].items():
            question_text = SURVEY_QUESTIONS[qid]['question']
            lines.append(f"\n  {question_text}:")
            for round_data in rounds_data[:3]:  # Show first 3 rounds
                lines.append(f"    Round {round_data['round']}:")
                top_responses = sorted(round_data['distribution'].items(), 
                                     key=lambda x: x[1], reverse=True)[:2]
                for option, count in top_responses:
                    pct = count / sum(round_data['distribution'].values()) * 100
                    lines.append(f"      {option}: {count} ({pct:.1f}%)")
    
    return "\n".join(lines)


def main():
    """Main function."""
    print("=" * 80)
    print("INFLUENCE NETWORK ANALYSIS")
    print("=" * 80)
    
    # Choose system
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'vector':
        system_type = 'vector'
        personas_file = Path('outputs/vector_beliefs_simulation/final_personas.json')
        survey_file = Path('outputs/vector_beliefs_simulation/survey_evolution.json')
        output_json = Path('outputs/vector_beliefs_simulation/influence_network_analysis.json')
        output_txt = Path('outputs/vector_beliefs_simulation/influence_network_analysis.txt')
    else:
        system_type = 'original'
        personas_file = Path('outputs/simulation/final_personas.json')
        survey_file = Path('outputs/simulation/multi_round_survey_evolution.json')
        output_json = Path('outputs/simulation/influence_network_analysis.json')
        output_txt = Path('outputs/simulation/influence_network_analysis.txt')
    
    print(f"\nSystem: {system_type}")
    
    # Load data
    print(f"\nLoading personas from {personas_file}...")
    personas = load_personas(str(personas_file), system_type)
    print(f"✓ Loaded {len(personas)} personas")
    
    print(f"\nLoading survey evolution from {survey_file}...")
    with open(survey_file) as f:
        survey_evolution = json.load(f)
    print(f"✓ Loaded survey evolution data")
    
    # Generate report
    print(f"\nGenerating influence network analysis...")
    report = generate_influence_report(
        personas,
        survey_evolution,
        system_type=system_type,
        sample_size=8,
        random_seed=42
    )
    
    # Save reports
    print(f"\nSaving JSON report to {output_json}...")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved JSON report")
    
    print(f"\nSaving text report to {output_txt}...")
    formatted_report = format_influence_report(report)
    with open(output_txt, 'w') as f:
        f.write(formatted_report)
    print(f"✓ Saved text report")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nReports saved to:")
    print(f"  - {output_json}")
    print(f"  - {output_txt}")


if __name__ == '__main__':
    main()

