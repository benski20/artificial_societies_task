"""
Script to augment existing personas with LLM narratives and beliefs.

This script loads existing personas from JSON and adds LLM-generated
narratives and beliefs, then saves them back.
"""

import json
import sys
from pathlib import Path
from persona.persona import Persona
from llm.augmentation import augment_personas_with_llm


def load_personas_from_json(filepath: str) -> list:
    """Load personas from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert dicts back to Persona objects
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
            first_gen_college=p_dict['first_gen_college'],
            gpa=p_dict['gpa'],
            sports_participation=p_dict['sports_participation'],
            mental_health=p_dict['mental_health'],
            social_media_intensity=p_dict['social_media_intensity'],
            college_intention=p_dict['college_intention'],
            narrative=p_dict.get('narrative'),
            beliefs=p_dict.get('beliefs', {}),
            survey_responses=p_dict.get('survey_responses', {})
        )
        personas.append(persona)
    
    return personas


def main():
    """Main function."""
    input_file = Path('outputs/constraint_personas.json')
    output_file = Path('outputs/constraint_personas_augmented.json')
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        print("Please run the generation script first:")
        print("  python3 main.py --method constraint --n 100 --surveys --validate")
        sys.exit(1)
    
    print("=" * 80)
    print("LLM AUGMENTATION SCRIPT")
    print("=" * 80)
    print(f"\nLoading personas from: {input_file}")
    
    # Load personas
    personas = load_personas_from_json(str(input_file))
    print(f"✓ Loaded {len(personas)} personas")
    
    # Check if already augmented
    if personas[0].narrative:
        response = input(f"\nPersonas already have narratives. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Augment with LLM
    print("\nAugmenting personas with LLM...")
    print("(This requires OpenAI API key in config/.env)")
    print("-" * 80)
    
    try:
        augmented_personas = augment_personas_with_llm(
            personas,
            model_name='gpt-4o-mini',
            verbose=True
        )
        
        # Save augmented personas
        print(f"\nSaving augmented personas to: {output_file}")
        personas_dict = [p.to_dict() for p in augmented_personas]
        
        with open(output_file, 'w') as f:
            json.dump(personas_dict, f, indent=2)
        
        print(f"✓ Saved {len(augmented_personas)} augmented personas")
        print(f"\nAugmentation complete!")
        print(f"  Original: {input_file}")
        print(f"  Augmented: {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error during augmentation: {e}")
        print("\nTo use LLM augmentation:")
        print("1. Install OpenAI package: pip install openai")
        print("2. Create config/.env with your API key:")
        print("   OPENAI_API_KEY=your_key_here")
        print("3. Run this script again")
        sys.exit(1)


if __name__ == '__main__':
    main()

