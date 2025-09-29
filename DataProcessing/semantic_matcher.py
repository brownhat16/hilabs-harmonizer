"""
Semantic Matcher for Medical Text

Provides advanced semantic matching capabilities for medical terms including:
- Context-aware matching
- Semantic similarity scoring
- Multi-word phrase matching
- Medical concept relationship matching
"""

import re
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from rapidfuzz import fuzz, process

@dataclass
class SemanticMatch:
    """Represents a semantic match with confidence and context."""
    query: str
    match: str
    confidence: float
    method: str
    context: Optional[str] = None
    semantic_type: Optional[str] = None
    concept_id: Optional[str] = None

class MedicalSemanticMatcher:
    """Advanced semantic matching for medical terms."""
    
    def __init__(self, vocabulary_data: Optional[pd.DataFrame] = None):
        self.vocabulary_data = vocabulary_data
        self.semantic_types = self._initialize_semantic_types()
        self.medical_relationships = self._initialize_medical_relationships()
        self.context_patterns = self._initialize_context_patterns()
        self.phrase_patterns = self._initialize_phrase_patterns()
    
    def _initialize_semantic_types(self) -> Dict[str, List[str]]:
        """Initialize semantic type categories and their characteristics."""
        return {
            'procedure': [
                'surgery', 'operation', 'procedure', 'treatment', 'therapy',
                'examination', 'test', 'scan', 'imaging', 'biopsy', 'injection',
                'removal', 'repair', 'replacement', 'transplant', 'drainage'
            ],
            'condition': [
                'disease', 'disorder', 'syndrome', 'condition', 'illness',
                'infection', 'inflammation', 'injury', 'trauma', 'fracture',
                'cancer', 'tumor', 'mass', 'lesion', 'abnormality'
            ],
            'anatomy': [
                'organ', 'tissue', 'structure', 'part', 'region', 'area',
                'muscle', 'bone', 'joint', 'vessel', 'nerve', 'gland',
                'cavity', 'space', 'surface', 'layer', 'component'
            ],
            'medication': [
                'drug', 'medicine', 'medication', 'therapy', 'treatment',
                'injection', 'tablet', 'capsule', 'cream', 'ointment',
                'solution', 'suspension', 'powder', 'patch', 'inhaler'
            ],
            'lab': [
                'test', 'analysis', 'measurement', 'level', 'count',
                'concentration', 'value', 'result', 'finding', 'marker',
                'indicator', 'screening', 'monitoring', 'assessment'
            ]
        }
    
    def _initialize_medical_relationships(self) -> Dict[str, List[str]]:
        """Initialize medical concept relationships."""
        return {
            'anatomy_condition': {
                'heart': ['cardiac', 'cardiovascular', 'myocardial'],
                'lung': ['pulmonary', 'respiratory', 'bronchial'],
                'liver': ['hepatic', 'liver'],
                'kidney': ['renal', 'kidney'],
                'brain': ['cerebral', 'neurological', 'cranial'],
                'spine': ['spinal', 'vertebral', 'dorsal'],
                'joint': ['articular', 'joint'],
                'muscle': ['muscular', 'myo'],
                'bone': ['osseous', 'skeletal', 'bony']
            },
            'procedure_anatomy': {
                'surgery': ['surgical', 'operative'],
                'imaging': ['radiographic', 'scan'],
                'biopsy': ['tissue sampling'],
                'injection': ['injectable', 'parenteral'],
                'examination': ['clinical exam', 'physical exam']
            },
            'medication_condition': {
                'antibiotic': ['infection', 'bacterial'],
                'analgesic': ['pain', 'analgesia'],
                'antihypertensive': ['hypertension', 'blood pressure'],
                'antidiabetic': ['diabetes', 'glucose'],
                'anticoagulant': ['clotting', 'thrombosis'],
                'anti-inflammatory': ['inflammation', 'inflammatory']
            }
        }
    
    def _initialize_context_patterns(self) -> Dict[str, List[str]]:
        """Initialize context patterns for semantic matching."""
        return {
            'diagnostic': ['diagnosis', 'diagnosed', 'finding', 'result', 'test'],
            'therapeutic': ['treatment', 'therapy', 'medication', 'surgery'],
            'anatomical': ['location', 'site', 'region', 'area', 'part'],
            'temporal': ['acute', 'chronic', 'recent', 'previous', 'history'],
            'severity': ['mild', 'moderate', 'severe', 'critical', 'emergency']
        }
    
    def _initialize_phrase_patterns(self) -> List[Tuple[str, str]]:
        """Initialize common medical phrase patterns."""
        return [
            (r'\b(\w+)\s+(surgery|operation|procedure)\b', 'procedure'),
            (r'\b(\w+)\s+(disease|disorder|syndrome)\b', 'condition'),
            (r'\b(\w+)\s+(pain|ache|discomfort)\b', 'symptom'),
            (r'\b(\w+)\s+(test|exam|study)\b', 'procedure'),
            (r'\b(\w+)\s+(injection|injection)\b', 'procedure'),
            (r'\b(\w+)\s+(therapy|treatment)\b', 'treatment'),
            (r'\b(\w+)\s+(scan|imaging)\b', 'procedure'),
            (r'\b(\w+)\s+(biopsy|sampling)\b', 'procedure')
        ]
    
    def extract_semantic_context(self, text: str) -> Dict[str, List[str]]:
        """
        Extract semantic context from text.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with extracted context
        """
        if not text:
            return {}
        
        context = defaultdict(list)
        text_lower = text.lower()
        
        # Extract semantic types
        for sem_type, keywords in self.semantic_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    context['semantic_types'].append(sem_type)
        
        # Extract context patterns
        for context_type, patterns in self.context_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    context['context_types'].append(context_type)
        
        # Extract phrase patterns
        for pattern, category in self.phrase_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                context['phrase_matches'].append((match, category))
        
        return dict(context)
    
    def calculate_semantic_similarity(self, term1: str, term2: str, 
                                    context: Optional[Dict] = None) -> float:
        """
        Calculate semantic similarity between two terms.
        
        Args:
            term1: First term
            term2: Second term
            context: Optional context for similarity calculation
        
        Returns:
            Similarity score between 0 and 1
        """
        if not term1 or not term2:
            return 0.0
        
        # Basic string similarity
        basic_similarity = fuzz.ratio(term1.lower(), term2.lower()) / 100.0
        
        # Semantic type similarity
        semantic_similarity = 0.0
        if context:
            context1 = self.extract_semantic_context(term1)
            context2 = self.extract_semantic_context(term2)
            
            # Check for semantic type overlap
            types1 = set(context1.get('semantic_types', []))
            types2 = set(context2.get('semantic_types', []))
            
            if types1 and types2:
                semantic_similarity = len(types1.intersection(types2)) / len(types1.union(types2))
        
        # Medical relationship similarity
        relationship_similarity = 0.0
        for relationship_type, relationships in self.medical_relationships.items():
            for key, values in relationships.items():
                if key in term1.lower() and any(val in term2.lower() for val in values):
                    relationship_similarity = 0.8
                    break
                if key in term2.lower() and any(val in term1.lower() for val in values):
                    relationship_similarity = 0.8
                    break
        
        # Combine similarities with weights
        final_similarity = (
            basic_similarity * 0.4 +
            semantic_similarity * 0.3 +
            relationship_similarity * 0.3
        )
        
        return min(final_similarity, 1.0)
    
    def find_semantic_matches(self, query: str, candidates: List[str], 
                            context: Optional[Dict] = None,
                            threshold: float = 0.6) -> List[SemanticMatch]:
        """
        Find semantic matches for a query term.
        
        Args:
            query: Query term to match
            candidates: List of candidate terms
            context: Optional context for matching
            threshold: Minimum similarity threshold
        
        Returns:
            List of semantic matches
        """
        if not query or not candidates:
            return []
        
        matches = []
        query_context = self.extract_semantic_context(query)
        
        for candidate in candidates:
            # Calculate semantic similarity
            similarity = self.calculate_semantic_similarity(query, candidate, context)
            
            if similarity >= threshold:
                # Determine semantic type
                candidate_context = self.extract_semantic_context(candidate)
                semantic_type = None
                
                if candidate_context.get('semantic_types'):
                    semantic_type = candidate_context['semantic_types'][0]
                
                # Determine method
                method = "semantic_similarity"
                if similarity > 0.9:
                    method = "high_similarity"
                elif similarity > 0.8:
                    method = "good_similarity"
                else:
                    method = "moderate_similarity"
                
                matches.append(SemanticMatch(
                    query=query,
                    match=candidate,
                    confidence=similarity,
                    method=method,
                    context=context,
                    semantic_type=semantic_type
                ))
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches
    
    def match_medical_phrases(self, query: str, vocabulary_data: Optional[pd.DataFrame] = None) -> List[SemanticMatch]:
        """
        Match medical phrases using semantic understanding.
        
        Args:
            query: Query phrase to match
            vocabulary_data: Optional vocabulary data
        
        Returns:
            List of semantic matches
        """
        if not query:
            return []
        
        if vocabulary_data is None:
            vocabulary_data = self.vocabulary_data
        
        if vocabulary_data is None:
            return []
        
        # Extract context from query
        query_context = self.extract_semantic_context(query)
        
        # Get candidate terms from vocabulary
        candidates = vocabulary_data['STR'].astype(str).tolist()
        
        # Find semantic matches
        matches = self.find_semantic_matches(query, candidates, query_context)
        
        # Add concept information if available
        for match in matches:
            if 'CUI' in vocabulary_data.columns:
                match_row = vocabulary_data[vocabulary_data['STR'] == match.match]
                if not match_row.empty:
                    match.concept_id = match_row.iloc[0]['CUI']
            
            if 'STY' in vocabulary_data.columns:
                match_row = vocabulary_data[vocabulary_data['STR'] == match.match]
                if not match_row.empty:
                    match.semantic_type = match_row.iloc[0]['STY']
        
        return matches
    
    def resolve_token_order_variants(self, query: str, candidates: List[str]) -> List[SemanticMatch]:
        """
        Resolve token order variants (e.g., "coronary angioplasty" vs "percutaneous transluminal coronary angioplasty").
        
        Args:
            query: Query with potential token order issues
            candidates: List of candidate terms
        
        Returns:
            List of semantic matches
        """
        if not query or not candidates:
            return []
        
        query_tokens = set(query.lower().split())
        matches = []
        
        for candidate in candidates:
            candidate_tokens = set(candidate.lower().split())
            
            # Calculate token overlap
            overlap = len(query_tokens.intersection(candidate_tokens))
            total_tokens = len(query_tokens.union(candidate_tokens))
            
            if total_tokens > 0:
                token_similarity = overlap / total_tokens
                
                # Check for subset relationship
                if query_tokens.issubset(candidate_tokens):
                    token_similarity += 0.2  # Bonus for subset
                elif candidate_tokens.issubset(query_tokens):
                    token_similarity += 0.1  # Bonus for superset
                
                if token_similarity >= 0.6:
                    matches.append(SemanticMatch(
                        query=query,
                        match=candidate,
                        confidence=min(token_similarity, 1.0),
                        method="token_order_variant",
                        context="token_overlap"
                    ))
        
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches
    
    def get_semantic_analysis(self, text: str) -> Dict[str, Any]:
        """
        Get comprehensive semantic analysis of text.
        
        Args:
            text: Input text
        
        Returns:
            Semantic analysis results
        """
        if not text:
            return {'text': text, 'context': {}, 'matches': []}
        
        context = self.extract_semantic_context(text)
        
        # Find semantic matches if vocabulary data is available
        matches = []
        if self.vocabulary_data is not None:
            matches = self.match_medical_phrases(text)
        
        return {
            'text': text,
            'context': context,
            'matches': matches,
            'semantic_types': context.get('semantic_types', []),
            'context_types': context.get('context_types', []),
            'phrase_matches': context.get('phrase_matches', [])
        }
    
    def load_vocabulary_data(self, parquet_path: str):
        """Load vocabulary data from parquet file."""
        try:
            self.vocabulary_data = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"Error loading vocabulary data from {parquet_path}: {e}")

# Convenience functions
def find_semantic_matches(query: str, candidates: List[str], 
                         threshold: float = 0.6) -> List[SemanticMatch]:
    """
    Find semantic matches for a query term.
    
    Args:
        query: Query term to match
        candidates: List of candidate terms
        threshold: Minimum similarity threshold
    
    Returns:
        List of semantic matches
    """
    matcher = MedicalSemanticMatcher()
    return matcher.find_semantic_matches(query, candidates, threshold=threshold)

def resolve_token_variants(query: str, candidates: List[str]) -> List[SemanticMatch]:
    """
    Resolve token order variants.
    
    Args:
        query: Query with potential token order issues
        candidates: List of candidate terms
    
    Returns:
        List of semantic matches
    """
    matcher = MedicalSemanticMatcher()
    return matcher.resolve_token_order_variants(query, candidates)

def analyze_semantic_context(text: str) -> Dict[str, Any]:
    """
    Analyze semantic context of text.
    
    Args:
        text: Input text
    
    Returns:
        Semantic analysis results
    """
    matcher = MedicalSemanticMatcher()
    return matcher.get_semantic_analysis(text)

if __name__ == "__main__":
    # Test the semantic matcher
    test_queries = [
        "coronary angioplasty",
        "percutaneous transluminal coronary angioplasty",
        "heart surgery",
        "cardiac procedure",
        "chest pain",
        "thoracic discomfort"
    ]
    
    test_candidates = [
        "percutaneous transluminal coronary angioplasty",
        "coronary angioplasty",
        "cardiac surgery",
        "heart procedure",
        "chest pain",
        "thoracic pain",
        "abdominal pain",
        "headache"
    ]
    
    matcher = MedicalSemanticMatcher()
    
    for query in test_queries:
        print(f"Query: {query}")
        
        # Test semantic matching
        matches = matcher.find_semantic_matches(query, test_candidates)
        print(f"Semantic matches: {[(m.match, f'{m.confidence:.2f}') for m in matches[:3]]}")
        
        # Test token order variants
        variants = matcher.resolve_token_order_variants(query, test_candidates)
        print(f"Token variants: {[(v.match, f'{v.confidence:.2f}') for v in variants[:3]]}")
        
        # Test semantic analysis
        analysis = matcher.get_semantic_analysis(query)
        print(f"Context: {analysis['context']}")
        print("---")

