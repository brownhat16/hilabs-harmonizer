"""
Layman to Clinical Terminology Mapper

Maps layman/patient-friendly terms to clinical/medical terminology for consistency.
Handles the challenge of "Consistency in Terminology: Layman v/s Clinical terms"
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TermMapping:
    """Represents a mapping between layman and clinical terms."""
    layman_term: str
    clinical_term: str
    confidence: float
    category: str
    context: Optional[str] = None
    synonyms: List[str] = None

class LaymanClinicalMapper:
    """Maps layman terms to clinical terminology."""
    
    def __init__(self):
        self.term_mappings = self._initialize_term_mappings()
        self.category_patterns = self._initialize_category_patterns()
        self.context_rules = self._initialize_context_rules()
    
    def _initialize_term_mappings(self) -> Dict[str, TermMapping]:
        """Initialize comprehensive layman to clinical term mappings."""
        mappings = {}
        
        # Body parts and anatomy
        anatomy_mappings = [
            ("heart", "cardiac", "cardiovascular"),
            ("lung", "pulmonary", "respiratory"),
            ("liver", "hepatic"),
            ("kidney", "renal"),
            ("brain", "cerebral", "neurological"),
            ("spine", "spinal", "vertebral"),
            ("joint", "articular"),
            ("muscle", "muscular"),
            ("bone", "osseous", "skeletal"),
            ("blood vessel", "vascular"),
            ("artery", "arterial"),
            ("vein", "venous"),
            ("nerve", "neural"),
            ("stomach", "gastric"),
            ("intestine", "intestinal"),
            ("bladder", "vesical"),
            ("chest", "thoracic"),
            ("belly", "abdominal"),
            ("back", "dorsal", "posterior"),
            ("front", "anterior", "ventral"),
            ("side", "lateral"),
            ("middle", "medial"),
            ("upper", "superior"),
            ("lower", "inferior")
        ]
        
        for layman, *clinical_terms in anatomy_mappings:
            for clinical in clinical_terms:
                key = f"{layman}_{clinical}"
                mappings[key] = TermMapping(
                    layman_term=layman,
                    clinical_term=clinical,
                    confidence=0.9,
                    category="anatomy",
                    synonyms=[layman, clinical]
                )
        
        # Medical conditions
        condition_mappings = [
            ("heart attack", "myocardial infarction", "MI"),
            ("stroke", "cerebrovascular accident", "CVA"),
            ("high blood pressure", "hypertension", "HTN"),
            ("diabetes", "diabetes mellitus", "DM"),
            ("cancer", "neoplasm", "malignancy"),
            ("tumor", "neoplasm", "mass"),
            ("lump", "mass", "nodule"),
            ("swelling", "edema", "inflammation"),
            ("bruise", "contusion", "ecchymosis"),
            ("cut", "laceration", "incision"),
            ("burn", "thermal injury"),
            ("fever", "pyrexia", "hyperthermia"),
            ("cold", "upper respiratory infection", "URI"),
            ("flu", "influenza"),
            ("pneumonia", "pneumonia"),
            ("asthma", "asthma"),
            ("arthritis", "arthritis"),
            ("depression", "major depressive disorder"),
            ("anxiety", "anxiety disorder"),
            ("seizure", "seizure disorder", "epilepsy"),
            ("headache", "cephalgia", "head pain"),
            ("chest pain", "thoracic pain", "chest discomfort"),
            ("stomach ache", "abdominal pain", "gastric pain"),
            ("back pain", "dorsal pain", "spinal pain"),
            ("joint pain", "arthralgia"),
            ("muscle pain", "myalgia"),
            ("nausea", "nausea"),
            ("vomiting", "emesis"),
            ("diarrhea", "diarrhea"),
            ("constipation", "constipation"),
            ("dizziness", "vertigo", "disequilibrium"),
            ("fatigue", "fatigue", "asthenia"),
            ("weakness", "weakness", "asthenia"),
            ("numbness", "paresthesia"),
            ("tingling", "paresthesia"),
            ("shortness of breath", "dyspnea"),
            ("difficulty breathing", "dyspnea"),
            ("rapid heartbeat", "tachycardia"),
            ("slow heartbeat", "bradycardia"),
            ("irregular heartbeat", "arrhythmia"),
            ("chest tightness", "chest discomfort"),
            ("wheezing", "wheezing"),
            ("coughing", "cough"),
            ("sneezing", "sneezing"),
            ("runny nose", "rhinorrhea"),
            ("stuffy nose", "nasal congestion"),
            ("sore throat", "pharyngitis"),
            ("ear pain", "otalgia"),
            ("eye pain", "ophthalmalgia"),
            ("blurred vision", "blurred vision"),
            ("double vision", "diplopia"),
            ("hearing loss", "hearing impairment"),
            ("ringing in ears", "tinnitus"),
            ("memory loss", "memory impairment"),
            ("confusion", "confusion", "disorientation"),
            ("sleep problems", "sleep disorder", "insomnia"),
            ("weight loss", "weight loss"),
            ("weight gain", "weight gain"),
            ("loss of appetite", "anorexia"),
            ("excessive thirst", "polydipsia"),
            ("excessive urination", "polyuria"),
            ("frequent urination", "frequency"),
            ("painful urination", "dysuria"),
            ("blood in urine", "hematuria"),
            ("blood in stool", "hematochezia"),
            ("black stool", "melena"),
            ("yellow skin", "jaundice"),
            ("pale skin", "pallor"),
            ("blue skin", "cyanosis"),
            ("rash", "dermatitis", "skin eruption"),
            ("itchy skin", "pruritus"),
            ("dry skin", "xerosis"),
            ("hair loss", "alopecia"),
            ("nail changes", "nail dystrophy")
        ]
        
        for layman, *clinical_terms in condition_mappings:
            for clinical in clinical_terms:
                key = f"{layman}_{clinical}"
                mappings[key] = TermMapping(
                    layman_term=layman,
                    clinical_term=clinical,
                    confidence=0.9,
                    category="condition",
                    synonyms=[layman, clinical]
                )
        
        # Medical procedures
        procedure_mappings = [
            ("x-ray", "radiograph", "radiography"),
            ("scan", "imaging study"),
            ("mri", "magnetic resonance imaging"),
            ("cat scan", "computed tomography", "CT"),
            ("ultrasound", "ultrasonography"),
            ("blood test", "laboratory test", "blood work"),
            ("urine test", "urinalysis"),
            ("biopsy", "biopsy"),
            ("surgery", "surgical procedure", "operation"),
            ("operation", "surgical procedure"),
            ("injection", "injection"),
            ("shot", "injection"),
            ("iv", "intravenous"),
            ("drip", "intravenous infusion"),
            ("breathing treatment", "respiratory therapy"),
            ("physical therapy", "physical therapy", "PT"),
            ("occupational therapy", "occupational therapy", "OT"),
            ("speech therapy", "speech therapy"),
            ("chemotherapy", "chemotherapy"),
            ("radiation", "radiation therapy"),
            ("dialysis", "dialysis"),
            ("transplant", "transplantation"),
            ("pacemaker", "cardiac pacemaker"),
            ("stent", "vascular stent"),
            ("bypass", "bypass surgery"),
            ("angioplasty", "angioplasty"),
            ("endoscopy", "endoscopy"),
            ("colonoscopy", "colonoscopy"),
            ("mammogram", "mammography"),
            ("pap smear", "pap smear", "cervical cytology"),
            ("ekg", "electrocardiogram", "ECG"),
            ("stress test", "stress test", "exercise tolerance test"),
            ("echo", "echocardiogram"),
            ("sleep study", "polysomnography"),
            ("allergy test", "allergy testing"),
            ("skin test", "skin testing"),
            ("pregnancy test", "pregnancy test"),
            ("glucose test", "glucose testing"),
            ("cholesterol test", "lipid panel"),
            ("thyroid test", "thyroid function test")
        ]
        
        for layman, *clinical_terms in procedure_mappings:
            for clinical in clinical_terms:
                key = f"{layman}_{clinical}"
                mappings[key] = TermMapping(
                    layman_term=layman,
                    clinical_term=clinical,
                    confidence=0.9,
                    category="procedure",
                    synonyms=[layman, clinical]
                )
        
        # Medications
        medication_mappings = [
            ("pain medicine", "analgesic", "pain medication"),
            ("pain killer", "analgesic"),
            ("antibiotic", "antibiotic"),
            ("anti-inflammatory", "anti-inflammatory medication"),
            ("blood thinner", "anticoagulant"),
            ("water pill", "diuretic"),
            ("heart medicine", "cardiac medication"),
            ("blood pressure medicine", "antihypertensive"),
            ("diabetes medicine", "antidiabetic"),
            ("cholesterol medicine", "statin", "lipid-lowering agent"),
            ("stomach medicine", "gastrointestinal medication"),
            ("sleeping pill", "sedative", "hypnotic"),
            ("anxiety medicine", "anxiolytic"),
            ("depression medicine", "antidepressant"),
            ("vitamin", "vitamin supplement"),
            ("supplement", "dietary supplement"),
            ("inhaler", "inhaler"),
            ("nasal spray", "nasal spray"),
            ("eye drops", "ophthalmic drops"),
            ("ear drops", "otic drops"),
            ("cream", "topical cream"),
            ("ointment", "topical ointment"),
            ("patch", "transdermal patch"),
            ("tablet", "tablet"),
            ("pill", "tablet", "capsule"),
            ("capsule", "capsule"),
            ("liquid", "liquid medication"),
            ("syrup", "syrup"),
            ("injection", "injectable medication")
        ]
        
        for layman, *clinical_terms in medication_mappings:
            for clinical in clinical_terms:
                key = f"{layman}_{clinical}"
                mappings[key] = TermMapping(
                    layman_term=layman,
                    clinical_term=clinical,
                    confidence=0.9,
                    category="medication",
                    synonyms=[layman, clinical]
                )
        
        return mappings
    
    def _initialize_category_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for different medical categories."""
        return {
            'anatomy': ['pain', 'ache', 'hurt', 'sore', 'tender'],
            'condition': ['problem', 'issue', 'trouble', 'disorder', 'disease'],
            'procedure': ['test', 'exam', 'check', 'study', 'scan'],
            'medication': ['medicine', 'drug', 'pill', 'treatment', 'therapy']
        }
    
    def _initialize_context_rules(self) -> Dict[str, str]:
        """Initialize context-based mapping rules."""
        return {
            'chest pain': 'thoracic pain',
            'heart pain': 'cardiac pain',
            'lung problem': 'pulmonary condition',
            'liver problem': 'hepatic condition',
            'kidney problem': 'renal condition',
            'brain problem': 'neurological condition',
            'stomach problem': 'gastric condition',
            'blood problem': 'hematological condition'
        }
    
    def map_layman_to_clinical(self, text: str, context: Optional[str] = None) -> str:
        """
        Map layman terms to clinical terminology in text.
        
        Args:
            text: Input text with layman terms
            context: Optional context for disambiguation
        
        Returns:
            Text with clinical terminology
        """
        if not text:
            return text
        
        mapped_text = text.lower()
        
        # Apply context rules first
        if context:
            for layman, clinical in self.context_rules.items():
                if layman in mapped_text:
                    mapped_text = mapped_text.replace(layman, clinical)
        
        # Apply term mappings
        for key, mapping in self.term_mappings.items():
            if mapping.layman_term in mapped_text:
                mapped_text = mapped_text.replace(mapping.layman_term, mapping.clinical_term)
        
        return mapped_text
    
    def get_clinical_terms(self, layman_term: str) -> List[str]:
        """
        Get clinical terms for a layman term.
        
        Args:
            layman_term: Layman term to map
        
        Returns:
            List of clinical terms
        """
        clinical_terms = []
        layman_lower = layman_term.lower()
        
        for mapping in self.term_mappings.values():
            if mapping.layman_term == layman_lower:
                clinical_terms.append(mapping.clinical_term)
        
        return list(set(clinical_terms))  # Remove duplicates
    
    def get_layman_terms(self, clinical_term: str) -> List[str]:
        """
        Get layman terms for a clinical term.
        
        Args:
            clinical_term: Clinical term to map
        
        Returns:
            List of layman terms
        """
        layman_terms = []
        clinical_lower = clinical_term.lower()
        
        for mapping in self.term_mappings.values():
            if mapping.clinical_term == clinical_lower:
                layman_terms.append(mapping.layman_term)
        
        return list(set(layman_terms))  # Remove duplicates
    
    def analyze_terminology_consistency(self, text: str) -> Dict[str, any]:
        """
        Analyze text for terminology consistency opportunities.
        
        Args:
            text: Input text
        
        Returns:
            Analysis results with mapping suggestions
        """
        if not text:
            return {'original': text, 'mappings': [], 'mapping_count': 0}
        
        text_lower = text.lower()
        mappings_found = []
        
        for key, mapping in self.term_mappings.items():
            if mapping.layman_term in text_lower:
                mappings_found.append({
                    'layman_term': mapping.layman_term,
                    'clinical_term': mapping.clinical_term,
                    'confidence': mapping.confidence,
                    'category': mapping.category,
                    'context': mapping.context
                })
        
        return {
            'original': text,
            'mappings': mappings_found,
            'mapping_count': len(mappings_found)
        }
    
    def get_terminology_suggestions(self, text: str) -> Dict[str, any]:
        """
        Get terminology suggestions for text.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with terminology suggestions
        """
        analysis = self.analyze_terminology_consistency(text)
        
        # Create suggested clinical version
        clinical_text = self.map_layman_to_clinical(text)
        
        return {
            'original': text,
            'clinical_version': clinical_text,
            'suggestions': analysis['mappings'],
            'improvement_score': len(analysis['mappings']) / max(len(text.split()), 1)
        }
    
    def add_custom_mapping(self, layman_term: str, clinical_term: str, 
                          category: str = "custom", confidence: float = 0.8):
        """Add a custom layman to clinical mapping."""
        key = f"{layman_term}_{clinical_term}"
        self.term_mappings[key] = TermMapping(
            layman_term=layman_term.lower(),
            clinical_term=clinical_term.lower(),
            confidence=confidence,
            category=category,
            synonyms=[layman_term, clinical_term]
        )
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about the mapping dictionary."""
        stats = defaultdict(int)
        for mapping in self.term_mappings.values():
            stats[mapping.category] += 1
        return dict(stats)

# Convenience functions
def map_to_clinical_terminology(text: str, context: Optional[str] = None) -> str:
    """
    Map layman terms to clinical terminology.
    
    Args:
        text: Input text with layman terms
        context: Optional context for disambiguation
    
    Returns:
        Text with clinical terminology
    """
    mapper = LaymanClinicalMapper()
    return mapper.map_layman_to_clinical(text, context)

def analyze_terminology_consistency(text: str) -> Dict[str, any]:
    """
    Analyze text for terminology consistency.
    
    Args:
        text: Input text
    
    Returns:
        Analysis results
    """
    mapper = LaymanClinicalMapper()
    return mapper.analyze_terminology_consistency(text)

def get_terminology_suggestions(text: str) -> Dict[str, any]:
    """
    Get terminology suggestions for text.
    
    Args:
        text: Input text
    
    Returns:
        Terminology suggestions
    """
    mapper = LaymanClinicalMapper()
    return mapper.get_terminology_suggestions(text)

if __name__ == "__main__":
    # Test the layman-clinical mapper
    test_texts = [
        "Patient has chest pain and heart problems",
        "Need to do blood test and x-ray",
        "Patient taking pain medicine for back ache",
        "High blood pressure and diabetes issues",
        "Lung problem with breathing difficulty"
    ]
    
    mapper = LaymanClinicalMapper()
    
    for text in test_texts:
        print(f"Original: {text}")
        print(f"Clinical: {mapper.map_layman_to_clinical(text)}")
        suggestions = mapper.get_terminology_suggestions(text)
        print(f"Suggestions: {suggestions['suggestions']}")
        print("---")

