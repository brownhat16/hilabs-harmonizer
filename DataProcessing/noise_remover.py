"""
Medical Text Noise Removal Module

Handles removal of various types of noise commonly found in medical text:
- Measurement units and values
- Special characters and formatting
- Common medical text artifacts
- Non-medical content
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class NoisePattern:
    """Represents a noise pattern with its regex and replacement."""
    name: str
    pattern: re.Pattern
    replacement: str
    description: str

class MedicalNoiseRemover:
    """Removes various types of noise from medical text."""
    
    def __init__(self):
        self.noise_patterns = self._initialize_noise_patterns()
        self.medical_units = self._initialize_medical_units()
        self.common_artifacts = self._initialize_artifacts()
    
    def _initialize_noise_patterns(self) -> List[NoisePattern]:
        """Initialize common noise patterns in medical text."""
        patterns = [
            # Measurement patterns
            NoisePattern(
                name="measurements",
                pattern=re.compile(r'\b\d+(?:\.\d+)?\s*(?:mg|g|ml|l|mcg|iu|units?|tablets?|capsules?|drops?|times?|x)\b', re.IGNORECASE),
                replacement="",
                description="Remove dosage measurements"
            ),
            
            # Time patterns
            NoisePattern(
                name="time_patterns",
                pattern=re.compile(r'\b(?:daily|twice|three times|qid|tid|bid|qd|qhs|prn|as needed)\b', re.IGNORECASE),
                replacement="",
                description="Remove frequency indicators"
            ),
            
            # Special characters
            NoisePattern(
                name="special_chars",
                pattern=re.compile(r'[^\w\s\-\.\/\(\)\[\]%]'),
                replacement=" ",
                description="Remove special characters"
            ),
            
            # Multiple spaces
            NoisePattern(
                name="multiple_spaces",
                pattern=re.compile(r'\s+'),
                replacement=" ",
                description="Normalize whitespace"
            ),
            
            # Common medical artifacts
            NoisePattern(
                name="medical_artifacts",
                pattern=re.compile(r'\b(?:oral|iv|im|sc|po|pr|sl|buccal|topical|inhalation)\b', re.IGNORECASE),
                replacement="",
                description="Remove route of administration"
            ),
            
            # Brand name indicators
            NoisePattern(
                name="brand_indicators",
                pattern=re.compile(r'\b(?:brand|generic|trade name|proprietary)\b', re.IGNORECASE),
                replacement="",
                description="Remove brand name indicators"
            ),
        ]
        return patterns
    
    def _initialize_medical_units(self) -> List[str]:
        """Initialize common medical units to remove."""
        return [
            'mg', 'g', 'kg', 'ml', 'l', 'mcg', 'iu', 'units', 'unit',
            'tablets', 'tablet', 'capsules', 'capsule', 'drops', 'drop',
            'times', 'x', 'daily', 'bid', 'tid', 'qid', 'qd', 'qhs',
            'oral', 'iv', 'im', 'sc', 'po', 'pr', 'sl', 'buccal'
        ]
    
    def _initialize_artifacts(self) -> List[str]:
        """Initialize common medical text artifacts."""
        return [
            'brand', 'generic', 'trade name', 'proprietary',
            'as needed', 'prn', 'twice', 'three times',
            'topical', 'inhalation', 'sublingual'
        ]
    
    def remove_measurements(self, text: str) -> str:
        """Keep dosage measurements and units in text (updated per specifications)."""
        # We now keep measurements as they are clinically meaningful
        return text
    
    def remove_frequency_indicators(self, text: str) -> str:
        """Remove frequency and timing indicators."""
        if not text:
            return text
        
        frequency_patterns = [
            r'\b(?:daily|twice|three times|qid|tid|bid|qd|qhs|prn|as needed)\b',
            r'\b\d+\s*(?:times?|x)\s*(?:daily|per day|a day)\b',
            r'\b(?:every|once|twice)\s*(?:daily|day|week|month)\b'
        ]
        
        cleaned = text
        for pattern in frequency_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return self._clean_whitespace(cleaned)
    
    def remove_route_administration(self, text: str) -> str:
        """Remove route of administration indicators."""
        if not text:
            return text
        
        route_pattern = re.compile(
            r'\b(?:oral|iv|im|sc|po|pr|sl|buccal|topical|inhalation|sublingual)\b',
            re.IGNORECASE
        )
        cleaned = route_pattern.sub('', text)
        return self._clean_whitespace(cleaned)
    
    def remove_brand_indicators(self, text: str) -> str:
        """Remove brand name and generic indicators."""
        if not text:
            return text
        
        brand_pattern = re.compile(
            r'\b(?:brand|generic|trade name|proprietary|manufacturer)\b',
            re.IGNORECASE
        )
        cleaned = brand_pattern.sub('', text)
        return self._clean_whitespace(cleaned)
    
    def remove_special_characters(self, text: str, keep_medical: bool = True) -> str:
        """Remove special characters while optionally keeping medical-relevant ones."""
        if not text:
            return text
        
        if keep_medical:
            # Keep medical-relevant punctuation: hyphens, dots, slashes, parentheses, brackets
            pattern = r'[^\w\s\-\.\/\(\)\[\]%]'
        else:
            # Remove all non-alphanumeric except spaces
            pattern = r'[^\w\s]'
        
        cleaned = re.sub(pattern, ' ', text)
        return self._clean_whitespace(cleaned)
    
    def remove_common_artifacts(self, text: str) -> str:
        """Remove common medical text artifacts."""
        if not text:
            return text
        
        cleaned = text
        for artifact in self.common_artifacts:
            pattern = re.compile(rf'\b{re.escape(artifact)}\b', re.IGNORECASE)
            cleaned = pattern.sub('', cleaned)
        
        return self._clean_whitespace(cleaned)
    
    def clean_text(self, text: str, 
                   remove_measurements: bool = True,
                   remove_frequency: bool = True,
                   remove_routes: bool = True,
                   remove_brands: bool = True,
                   remove_special_chars: bool = True,
                   remove_artifacts: bool = True) -> str:
        """
        Comprehensive text cleaning with configurable options.
        
        Args:
            text: Input text to clean
            remove_measurements: Remove dosage measurements
            remove_frequency: Remove frequency indicators
            remove_routes: Remove route of administration
            remove_brands: Remove brand indicators
            remove_special_chars: Remove special characters
            remove_artifacts: Remove common artifacts
        
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        cleaned = text
        
        if remove_measurements:
            cleaned = self.remove_measurements(cleaned)
        
        if remove_frequency:
            cleaned = self.remove_frequency_indicators(cleaned)
        
        if remove_routes:
            cleaned = self.remove_route_administration(cleaned)
        
        if remove_brands:
            cleaned = self.remove_brand_indicators(cleaned)
        
        if remove_special_chars:
            cleaned = self.remove_special_characters(cleaned)
        
        if remove_artifacts:
            cleaned = self.remove_common_artifacts(cleaned)
        
        return cleaned.strip()
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace in text."""
        if not text:
            return text
        return re.sub(r'\s+', ' ', text).strip()
    
    def get_removed_components(self, text: str) -> Dict[str, List[str]]:
        """
        Extract components that would be removed for analysis.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with removed components by category
        """
        if not text:
            return {}
        
        components = {
            'measurements': [],
            'frequency': [],
            'routes': [],
            'brands': [],
            'artifacts': []
        }
        
        # Extract measurements
        measurement_matches = re.findall(
            r'\b\d+(?:\.\d+)?\s*(?:mg|g|ml|l|mcg|iu|units?|tablets?|capsules?|drops?)\b',
            text, re.IGNORECASE
        )
        components['measurements'] = measurement_matches
        
        # Extract frequency indicators
        frequency_matches = re.findall(
            r'\b(?:daily|twice|three times|qid|tid|bid|qd|qhs|prn|as needed)\b',
            text, re.IGNORECASE
        )
        components['frequency'] = frequency_matches
        
        # Extract routes
        route_matches = re.findall(
            r'\b(?:oral|iv|im|sc|po|pr|sl|buccal|topical|inhalation)\b',
            text, re.IGNORECASE
        )
        components['routes'] = route_matches
        
        # Extract brand indicators
        brand_matches = re.findall(
            r'\b(?:brand|generic|trade name|proprietary)\b',
            text, re.IGNORECASE
        )
        components['brands'] = brand_matches
        
        return components

# Convenience functions for easy usage
def remove_medical_noise(text: str, aggressive: bool = False) -> str:
    """
    Remove medical noise from text with configurable aggressiveness.
    
    Args:
        text: Input medical text
        aggressive: Use aggressive noise removal
    
    Returns:
        Cleaned text
    """
    remover = MedicalNoiseRemover()
    
    if aggressive:
        return remover.clean_text(text, remove_measurements=True, 
                                remove_frequency=True, remove_routes=True,
                                remove_brands=True, remove_special_chars=True,
                                remove_artifacts=True)
    else:
        return remover.clean_text(text, remove_measurements=True,
                                remove_frequency=False, remove_routes=False,
                                remove_brands=False, remove_special_chars=True,
                                remove_artifacts=False)

def analyze_medical_text(text: str) -> Dict[str, any]:
    """
    Analyze medical text and return components that would be removed.
    
    Args:
        text: Input medical text
    
    Returns:
        Analysis results
    """
    remover = MedicalNoiseRemover()
    
    return {
        'original': text,
        'cleaned': remover.clean_text(text),
        'aggressive_cleaned': remover.clean_text(text, aggressive=True),
        'removed_components': remover.get_removed_components(text)
    }

if __name__ == "__main__":
    # Test the noise remover
    test_texts = [
        "Metformin 500 mg oral tablet twice daily",
        "Aspirin 81 mg [Bayer] (generic available)",
        "Insulin 10 units subcutaneous injection",
        "Lisinopril 5mg PO qd for hypertension"
    ]
    
    remover = MedicalNoiseRemover()
    
    for text in test_texts:
        print(f"Original: {text}")
        print(f"Cleaned: {remover.clean_text(text)}")
        print(f"Aggressive: {remover.clean_text(text, aggressive=True)}")
        print("---")

