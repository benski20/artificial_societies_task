"""
Persona data structure representing a synthetic high school student.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np


@dataclass
class Persona:
    """
    Represents a synthetic U.S. high school student (ages 16-18).
    
    Attributes:
        # Demographics
        age: Age (16-18)
        gender: 'Female' or 'Male'
        race: Race/ethnicity category
        sexual_orientation: 'Heterosexual' or 'LGBTQ+'
        
        # Socioeconomic
        family_income: Annual household income in dollars
        income_percentile: 'Top 30%', 'Middle 40%', or 'Bottom 30%'
        income_quintile: For stratified sampling ('Bottom 20%', '20-40%', etc.)
        first_gen_college: Whether first-generation college student
        
        # Academic
        gpa: Grade point average (0.0-4.0)
        
        # Behavioral
        sports_participation: Whether participates in sports
        mental_health: 'Good', 'Fair', or 'Poor'
        social_media_intensity: 'Low', 'Moderate', 'High', or 'Almost Constant'
        
        # College intentions
        college_intention: College plan category
        
        # LLM-generated (optional)
        narrative: Background story/narrative
        beliefs: Dictionary of belief explanations by topic
        embedding: Text embedding vector (for similarity calculations)
        
        # Survey responses (optional)
        survey_responses: Dictionary of survey question -> response
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
    
    # LLM-generated (optional)
    narrative: Optional[str] = None
    beliefs: Dict[str, str] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    # Survey responses (optional)
    survey_responses: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary for serialization."""
        import numpy as np
        
        # Convert numpy types to native Python types for JSON serialization
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
            'beliefs': convert_to_native(self.beliefs),
            'survey_responses': convert_to_native(self.survey_responses)
        }
        
        # Handle embedding if present
        if self.embedding is not None:
            result['embedding'] = convert_to_native(self.embedding)
        
        return result
    
    def get_summary(self) -> str:
        """Get a text summary of the persona for embedding generation."""
        parts = [
            f"Age {self.age}, {self.gender}, {self.race}",
            f"Family income: ${self.family_income:,.0f}",
            f"GPA: {self.gpa:.2f}",
            f"First-generation college: {self.first_gen_college}",
            f"Mental health: {self.mental_health}",
            f"Social media: {self.social_media_intensity}",
            f"College plan: {self.college_intention}"
        ]
        if self.narrative:
            parts.append(f"Background: {self.narrative[:200]}...")
        return " | ".join(parts)

