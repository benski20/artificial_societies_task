"""
Vector-based Persona with latent belief vector.

Each persona has a continuous belief vector representing their internal
belief state in a multi-dimensional belief space.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np


@dataclass
class VectorPersona:
    """
    Persona with continuous latent belief vector.
    
    Attributes:
        # Demographics (same as original)
        age, gender, race, sexual_orientation
        family_income, gpa, etc.
        
        # Vector-based beliefs
        belief_vector: Continuous vector in belief space (shape: [belief_dim])
        belief_dim: Dimension of belief vector (default: 16)
        
        # Survey responses (probabilistic)
        survey_responses: Dictionary of question_id -> response
        response_probabilities: Dictionary of question_id -> probability_distribution
    """
    
    # Demographics
    age: int
    gender: str
    race: str
    sexual_orientation: str
    
    # Socioeconomic
    family_income: float
    income_percentile: Optional[str] = None
    income_quintile: Optional[str] = None
    first_gen_college: bool = False
    
    # Academic
    gpa: float = 0.0
    
    # Behavioral
    sports_participation: bool = False
    mental_health: str = 'Fair'
    social_media_intensity: str = 'Moderate'
    
    # College intentions
    college_intention: str = 'Unsure'
    
    # LLM-generated (optional, for compatibility)
    narrative: Optional[str] = None
    
    # Vector-based beliefs
    belief_vector: Optional[np.ndarray] = None
    belief_dim: int = 16
    
    # Survey responses
    survey_responses: Dict[str, Any] = field(default_factory=dict)
    response_probabilities: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize belief vector if not provided."""
        if self.belief_vector is None:
            self.belief_vector = self._initialize_belief_vector()
    
    def _initialize_belief_vector(self) -> np.ndarray:
        """
        Initialize belief vector from persona attributes.
        
        Uses demographic and behavioral attributes to seed the belief vector.
        """
        vector = np.zeros(self.belief_dim)
        
        # Seed based on attributes (deterministic initialization)
        rng = np.random.default_rng(hash(f"{self.age}{self.gender}{self.race}{self.family_income}") % 2**32)
        
        # Initialize with small random values
        vector = rng.normal(0, 0.3, self.belief_dim)
        
        # Adjust based on attributes
        # Higher income → more positive beliefs about college
        income_factor = np.clip((self.family_income - 50000) / 100000, -1, 1)
        vector[0] += income_factor * 0.5
        
        # Higher GPA → more positive beliefs
        gpa_factor = (self.gpa - 2.0) / 2.0  # Normalize to [-1, 1]
        vector[1] += gpa_factor * 0.4
        
        # Mental health affects beliefs
        mental_health_map = {'Good': 0.3, 'Fair': 0.0, 'Poor': -0.3}
        vector[2] += mental_health_map.get(self.mental_health, 0.0)
        
        # Social media intensity
        social_media_map = {'Low': -0.2, 'Moderate': 0.0, 'High': 0.2, 'Almost Constant': 0.4}
        vector[3] += social_media_map.get(self.social_media_intensity, 0.0)
        
        # College intention
        if 'college' in self.college_intention.lower():
            vector[4] += 0.5
        elif 'unsure' in self.college_intention.lower():
            vector[4] += 0.0
        else:
            vector[4] -= 0.3
        
        # Normalize to reasonable range
        vector = np.clip(vector, -2.0, 2.0)
        
        return vector.astype(np.float32)
    
    def get_belief_vector(self) -> np.ndarray:
        """Get belief vector."""
        if self.belief_vector is None:
            self.belief_vector = self._initialize_belief_vector()
        return self.belief_vector
    
    def update_belief_vector(self, new_vector: np.ndarray):
        """Update belief vector."""
        if new_vector.shape != (self.belief_dim,):
            raise ValueError(f"Belief vector shape mismatch: {new_vector.shape} != ({self.belief_dim},)")
        self.belief_vector = new_vector.astype(np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        result = {
            'age': int(self.age),
            'gender': self.gender,
            'race': self.race,
            'sexual_orientation': self.sexual_orientation,
            'family_income': float(self.family_income),
            'income_percentile': self.income_percentile,
            'income_quintile': self.income_quintile,
            'first_gen_college': bool(self.first_gen_college),
            'gpa': float(self.gpa),
            'sports_participation': bool(self.sports_participation),
            'mental_health': self.mental_health,
            'social_media_intensity': self.social_media_intensity,
            'college_intention': self.college_intention,
            'narrative': self.narrative,
            'belief_vector': convert_to_native(self.get_belief_vector()),
            'belief_dim': self.belief_dim,
            'survey_responses': convert_to_native(self.survey_responses),
            'response_probabilities': convert_to_native(self.response_probabilities)
        }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorPersona':
        """Create VectorPersona from dictionary."""
        belief_vector = None
        if 'belief_vector' in data and data['belief_vector'] is not None:
            belief_vector = np.array(data['belief_vector'], dtype=np.float32)
        
        return cls(
            age=data['age'],
            gender=data['gender'],
            race=data['race'],
            sexual_orientation=data['sexual_orientation'],
            family_income=data['family_income'],
            income_percentile=data.get('income_percentile'),
            income_quintile=data.get('income_quintile'),
            first_gen_college=data.get('first_gen_college', False),
            gpa=data.get('gpa', 0.0),
            sports_participation=data.get('sports_participation', False),
            mental_health=data.get('mental_health', 'Fair'),
            social_media_intensity=data.get('social_media_intensity', 'Moderate'),
            college_intention=data.get('college_intention', 'Unsure'),
            narrative=data.get('narrative'),
            belief_vector=belief_vector,
            belief_dim=data.get('belief_dim', 16),
            survey_responses=data.get('survey_responses', {}),
            response_probabilities={
                k: np.array(v) for k, v in data.get('response_probabilities', {}).items()
            } if data.get('response_probabilities') else {}
        )

