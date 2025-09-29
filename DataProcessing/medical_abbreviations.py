"""
Medical Abbreviations and Short Forms Module

Handles expansion of medical abbreviations and short forms commonly found in clinical text.
Supports both standard medical abbreviations and context-aware expansion.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class AbbreviationEntry:
    """Represents an abbreviation with its expansions and context."""
    abbreviation: str
    expansions: List[str]
    context: Optional[str] = None
    confidence: float = 1.0
    category: str = "general"

class MedicalAbbreviationExpander:
    """Expands medical abbreviations and short forms."""
    
    def __init__(self):
        self.abbreviations = self._initialize_abbreviations()
        self.context_patterns = self._initialize_context_patterns()
        self.ambiguous_abbreviations = self._initialize_ambiguous_abbreviations()
    
    def _initialize_abbreviations(self) -> Dict[str, AbbreviationEntry]:
        """Initialize comprehensive medical abbreviation dictionary."""
        abbreviations = {
            # Common medical procedures
            'mri': AbbreviationEntry('mri', ['magnetic resonance imaging'], category='procedure'),
            'ct': AbbreviationEntry('ct', ['computed tomography', 'cat scan'], category='procedure'),
            'ecg': AbbreviationEntry('ecg', ['electrocardiogram', 'electrocardiography'], category='procedure'),
            'ekg': AbbreviationEntry('ekg', ['electrocardiogram', 'electrocardiography'], category='procedure'),
            'eeg': AbbreviationEntry('eeg', ['electroencephalogram', 'electroencephalography'], category='procedure'),
            'emg': AbbreviationEntry('emg', ['electromyography', 'electromyogram'], category='procedure'),
            'us': AbbreviationEntry('us', ['ultrasound', 'ultrasonography'], category='procedure'),
            'xr': AbbreviationEntry('xr', ['x-ray', 'radiograph'], category='procedure'),
            'pet': AbbreviationEntry('pet', ['positron emission tomography'], category='procedure'),
            'spect': AbbreviationEntry('spect', ['single photon emission computed tomography'], category='procedure'),
            
            # Laboratory tests
            'cbc': AbbreviationEntry('cbc', ['complete blood count'], category='lab'),
            'bmp': AbbreviationEntry('bmp', ['basic metabolic panel'], category='lab'),
            'cmp': AbbreviationEntry('cmp', ['comprehensive metabolic panel'], category='lab'),
            'lft': AbbreviationEntry('lft', ['liver function test'], category='lab'),
            'rft': AbbreviationEntry('rft', ['renal function test'], category='lab'),
            'pt': AbbreviationEntry('pt', ['prothrombin time'], category='lab'),
            'ptt': AbbreviationEntry('ptt', ['partial thromboplastin time'], category='lab'),
            'inr': AbbreviationEntry('inr', ['international normalized ratio'], category='lab'),
            'hgb': AbbreviationEntry('hgb', ['hemoglobin'], category='lab'),
            'hct': AbbreviationEntry('hct', ['hematocrit'], category='lab'),
            'wbc': AbbreviationEntry('wbc', ['white blood cell'], category='lab'),
            'rbc': AbbreviationEntry('rbc', ['red blood cell'], category='lab'),
            'plt': AbbreviationEntry('plt', ['platelet'], category='lab'),
            'esr': AbbreviationEntry('esr', ['erythrocyte sedimentation rate'], category='lab'),
            'crp': AbbreviationEntry('crp', ['c-reactive protein'], category='lab'),
            'tsh': AbbreviationEntry('tsh', ['thyroid stimulating hormone'], category='lab'),
            't3': AbbreviationEntry('t3', ['triiodothyronine'], category='lab'),
            't4': AbbreviationEntry('t4', ['thyroxine'], category='lab'),
            'psa': AbbreviationEntry('psa', ['prostate specific antigen'], category='lab'),
            'hba1c': AbbreviationEntry('hba1c', ['hemoglobin a1c', 'glycated hemoglobin'], category='lab'),
            
            # Medical conditions
            'mi': AbbreviationEntry('mi', ['myocardial infarction', 'heart attack'], category='condition'),
            'cva': AbbreviationEntry('cva', ['cerebrovascular accident', 'stroke'], category='condition'),
            'tia': AbbreviationEntry('tia', ['transient ischemic attack'], category='condition'),
            'copd': AbbreviationEntry('copd', ['chronic obstructive pulmonary disease'], category='condition'),
            'chf': AbbreviationEntry('chf', ['congestive heart failure'], category='condition'),
            'dm': AbbreviationEntry('dm', ['diabetes mellitus'], category='condition'),
            'htn': AbbreviationEntry('htn', ['hypertension', 'high blood pressure'], category='condition'),
            'cad': AbbreviationEntry('cad', ['coronary artery disease'], category='condition'),
            'pe': AbbreviationEntry('pe', ['pulmonary embolism'], category='condition'),
            'dvt': AbbreviationEntry('dvt', ['deep vein thrombosis'], category='condition'),
            'uti': AbbreviationEntry('uti', ['urinary tract infection'], category='condition'),
            'pneumonia': AbbreviationEntry('pneumonia', ['pneumonia'], category='condition'),
            'sepsis': AbbreviationEntry('sepsis', ['sepsis'], category='condition'),
            'shock': AbbreviationEntry('shock', ['shock'], category='condition'),
            
            # Medications
            'asa': AbbreviationEntry('asa', ['aspirin', 'acetylsalicylic acid'], category='medication'),
            'ace': AbbreviationEntry('ace', ['angiotensin converting enzyme'], category='medication'),
            'arb': AbbreviationEntry('arb', ['angiotensin receptor blocker'], category='medication'),
            'ccb': AbbreviationEntry('ccb', ['calcium channel blocker'], category='medication'),
            'diuretic': AbbreviationEntry('diuretic', ['diuretic'], category='medication'),
            'statin': AbbreviationEntry('statin', ['statin'], category='medication'),
            'beta': AbbreviationEntry('beta', ['beta blocker'], category='medication'),
            'ppi': AbbreviationEntry('ppi', ['proton pump inhibitor'], category='medication'),
            'h2': AbbreviationEntry('h2', ['h2 blocker'], category='medication'),
            'nsaid': AbbreviationEntry('nsaid', ['nonsteroidal anti-inflammatory drug'], category='medication'),
            
            # Body parts and anatomy
            'abd': AbbreviationEntry('abd', ['abdomen', 'abdominal'], category='anatomy'),
            'chest': AbbreviationEntry('chest', ['chest'], category='anatomy'),
            'pelvis': AbbreviationEntry('pelvis', ['pelvis', 'pelvic'], category='anatomy'),
            'head': AbbreviationEntry('head', ['head'], category='anatomy'),
            'neck': AbbreviationEntry('neck', ['neck', 'cervical'], category='anatomy'),
            'back': AbbreviationEntry('back', ['back', 'spine', 'spinal'], category='anatomy'),
            'extremity': AbbreviationEntry('extremity', ['extremity', 'limb'], category='anatomy'),
            
            # Medical specialties
            'cardio': AbbreviationEntry('cardio', ['cardiology', 'cardiovascular'], category='specialty'),
            'neuro': AbbreviationEntry('neuro', ['neurology', 'neurological'], category='specialty'),
            'ortho': AbbreviationEntry('ortho', ['orthopedics', 'orthopedic'], category='specialty'),
            'derm': AbbreviationEntry('derm', ['dermatology', 'dermatological'], category='specialty'),
            'gyn': AbbreviationEntry('gyn', ['gynecology', 'gynecological'], category='specialty'),
            'uro': AbbreviationEntry('uro', ['urology', 'urological'], category='specialty'),
            'ent': AbbreviationEntry('ent', ['otolaryngology', 'ear nose throat'], category='specialty'),
            'psych': AbbreviationEntry('psych', ['psychiatry', 'psychiatric'], category='specialty'),
            
            # Common short forms
            'proc': AbbreviationEntry('proc', ['procedure'], category='general'),
            'dx': AbbreviationEntry('dx', ['diagnosis'], category='general'),
            'tx': AbbreviationEntry('tx', ['treatment', 'therapy'], category='general'),
            'rx': AbbreviationEntry('rx', ['prescription', 'medication'], category='general'),
            'hx': AbbreviationEntry('hx', ['history'], category='general'),
            'sx': AbbreviationEntry('sx', ['symptoms', 'symptom'], category='general'),
            'fx': AbbreviationEntry('fx', ['fracture'], category='general'),
            'dis': AbbreviationEntry('dis', ['discharge'], category='general'),
            'admit': AbbreviationEntry('admit', ['admission'], category='general'),
            'eval': AbbreviationEntry('eval', ['evaluation'], category='general'),
            'assess': AbbreviationEntry('assess', ['assessment'], category='general'),
            'monitor': AbbreviationEntry('monitor', ['monitoring'], category='general'),
            'follow': AbbreviationEntry('follow', ['follow-up'], category='general'),
        }
        return abbreviations
    
    def _initialize_context_patterns(self) -> Dict[str, List[str]]:
        """Initialize context patterns for disambiguating abbreviations."""
        return {
            'cardiac': ['heart', 'cardiac', 'coronary', 'myocardial', 'cardiovascular'],
            'neurological': ['brain', 'neurological', 'cerebral', 'spinal', 'nerve'],
            'respiratory': ['lung', 'pulmonary', 'respiratory', 'breathing', 'airway'],
            'gastrointestinal': ['stomach', 'intestinal', 'gastro', 'digestive', 'bowel'],
            'musculoskeletal': ['bone', 'muscle', 'joint', 'skeletal', 'orthopedic'],
            'endocrine': ['hormone', 'thyroid', 'diabetes', 'insulin', 'metabolic'],
            'renal': ['kidney', 'renal', 'urinary', 'nephrology'],
            'hematology': ['blood', 'hematology', 'anemia', 'bleeding', 'clotting']
        }
    
    def _initialize_ambiguous_abbreviations(self) -> Dict[str, List[AbbreviationEntry]]:
        """Initialize abbreviations that have multiple meanings."""
        return {
            'pt': [
                AbbreviationEntry('pt', ['prothrombin time'], context='lab', category='lab'),
                AbbreviationEntry('pt', ['physical therapy'], context='therapy', category='therapy'),
                AbbreviationEntry('pt', ['patient'], context='general', category='general')
            ],
            'ct': [
                AbbreviationEntry('ct', ['computed tomography'], context='imaging', category='procedure'),
                AbbreviationEntry('ct', ['chest tube'], context='procedure', category='procedure')
            ],
            'us': [
                AbbreviationEntry('us', ['ultrasound'], context='imaging', category='procedure'),
                AbbreviationEntry('us', ['united states'], context='geography', category='general')
            ]
        }
    
    def expand_abbreviation(self, text: str, context: Optional[str] = None) -> List[str]:
        """
        Expand a single abbreviation with optional context.
        
        Args:
            text: Abbreviation to expand
            context: Optional context for disambiguation
        
        Returns:
            List of possible expansions
        """
        if not text:
            return []
        
        text_lower = text.lower().strip()
        
        # Check for exact match
        if text_lower in self.abbreviations:
            entry = self.abbreviations[text_lower]
            return entry.expansions
        
        # Check for ambiguous abbreviations
        if text_lower in self.ambiguous_abbreviations:
            entries = self.ambiguous_abbreviations[text_lower]
            if context:
                # Try to match context
                for entry in entries:
                    if entry.context and context.lower() in entry.context.lower():
                        return entry.expansions
            # Return all possible expansions if no context match
            all_expansions = []
            for entry in entries:
                all_expansions.extend(entry.expansions)
            return all_expansions
        
        return []
    
    def expand_text_abbreviations(self, text: str, context: Optional[str] = None) -> str:
        """
        Expand all abbreviations in text.
        
        Args:
            text: Input text containing abbreviations
            context: Optional context for disambiguation
        
        Returns:
            Text with abbreviations expanded
        """
        if not text:
            return text
        
        # Find potential abbreviations (2-5 character words, possibly with periods)
        abbreviation_pattern = re.compile(r'\b[A-Za-z]{2,5}(?:\.)?\b')
        
        def replace_abbreviation(match):
            abbrev = match.group(0).rstrip('.').lower()
            expansions = self.expand_abbreviation(abbrev, context)
            
            if expansions:
                # Use the first (most common) expansion
                return expansions[0]
            else:
                # Return original if no expansion found
                return match.group(0)
        
        expanded_text = abbreviation_pattern.sub(replace_abbreviation, text)
        return expanded_text
    
    def get_abbreviation_candidates(self, text: str) -> List[Tuple[str, List[str]]]:
        """
        Find all abbreviation candidates in text and their expansions.
        
        Args:
            text: Input text
        
        Returns:
            List of tuples (abbreviation, expansions)
        """
        if not text:
            return []
        
        candidates = []
        abbreviation_pattern = re.compile(r'\b[A-Za-z]{2,5}(?:\.)?\b')
        
        for match in abbreviation_pattern.finditer(text):
            abbrev = match.group(0).rstrip('.').lower()
            expansions = self.expand_abbreviation(abbrev)
            
            if expansions:
                candidates.append((abbrev, expansions))
        
        return candidates
    
    def expand_with_confidence(self, text: str, context: Optional[str] = None) -> Dict[str, any]:
        """
        Expand abbreviations with confidence scoring.
        
        Args:
            text: Input text
            context: Optional context
        
        Returns:
            Dictionary with expansion results and confidence scores
        """
        if not text:
            return {'original': text, 'expanded': text, 'confidence': 0.0, 'expansions': []}
        
        candidates = self.get_abbreviation_candidates(text)
        expanded_text = text
        total_confidence = 0.0
        expansion_count = 0
        
        for abbrev, expansions in candidates:
            if expansions:
                # Use first expansion (most common)
                expansion = expansions[0]
                expanded_text = expanded_text.replace(abbrev, expansion)
                total_confidence += 1.0  # High confidence for known abbreviations
                expansion_count += 1
        
        avg_confidence = total_confidence / max(expansion_count, 1)
        
        return {
            'original': text,
            'expanded': expanded_text,
            'confidence': avg_confidence,
            'expansions': candidates,
            'expansion_count': expansion_count
        }
    
    def add_custom_abbreviation(self, abbreviation: str, expansions: List[str], 
                              context: Optional[str] = None, category: str = "custom"):
        """Add a custom abbreviation to the dictionary."""
        entry = AbbreviationEntry(abbreviation.lower(), expansions, context, 1.0, category)
        self.abbreviations[abbreviation.lower()] = entry
    
    def get_abbreviation_stats(self) -> Dict[str, int]:
        """Get statistics about the abbreviation dictionary."""
        stats = defaultdict(int)
        for entry in self.abbreviations.values():
            stats[entry.category] += 1
        return dict(stats)

# Convenience functions
def expand_medical_abbreviations(text: str, context: Optional[str] = None) -> str:
    """
    Expand medical abbreviations in text.
    
    Args:
        text: Input text with abbreviations
        context: Optional context for disambiguation
    
    Returns:
        Text with abbreviations expanded
    """
    expander = MedicalAbbreviationExpander()
    return expander.expand_text_abbreviations(text, context)

def analyze_abbreviations(text: str) -> Dict[str, any]:
    """
    Analyze abbreviations in text and return detailed results.
    
    Args:
        text: Input text
    
    Returns:
        Analysis results
    """
    expander = MedicalAbbreviationExpander()
    return expander.expand_with_confidence(text)

if __name__ == "__main__":
    # Test the abbreviation expander
    test_texts = [
        "Patient needs MRI of chest and CT scan",
        "CBC and BMP results are normal",
        "History of MI and CHF",
        "ASA 81mg daily for CAD",
        "PT/INR monitoring required"
    ]
    
    expander = MedicalAbbreviationExpander()
    
    for text in test_texts:
        print(f"Original: {text}")
        print(f"Expanded: {expander.expand_text_abbreviations(text)}")
        print(f"Candidates: {expander.get_abbreviation_candidates(text)}")
        print("---")

