"""
Enhanced Medical Text Preprocessor

Comprehensive preprocessing pipeline that implements all specifications:
1. Input sanitize & normalization (Unicode NFKC, trim, to lowercase)
2. Punctuation rules (remove/keep specific punctuation)
3. Whitespace collapse & tokenization
4. Word-level abbreviation expansion & British→US mapping
5. Numeric format normalization
6. Dose/unit/concentration/form parsing
7. Combination product detection
8. Stop-word filtering
9. Optional fuzzy correction
10. Structured output schema

Provides a comprehensive preprocessing pipeline for medical text harmonization.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from rapidfuzz import fuzz, process as rf_process

# Import our custom modules
from handle_case_space import normalize_medical_text, extract_medical_components
from noise_remover import MedicalNoiseRemover, remove_medical_noise
from medical_matcher import MedicalMatcher, MatchingConfig

class ProcessingStage(Enum):
    """Enumeration of processing stages."""
    NORMALIZATION = "normalization"
    NOISE_REMOVAL = "noise_removal"
    ABBREVIATION_EXPANSION = "abbreviation_expansion"
    TOKEN_COMPLETION = "token_completion"
    TERMINOLOGY_MAPPING = "terminology_mapping"
    SEMANTIC_ANALYSIS = "semantic_analysis"

@dataclass
class ParsedDose:
    """Represents parsed dose information."""
    dose_value: Optional[float] = None
    dose_unit: Optional[str] = None
    concentration: Optional[Dict[str, Any]] = None
    form: Optional[str] = None

@dataclass
class Component:
    """Represents a component in a combination product."""
    component_raw: str
    normalized: str
    tokens: List[str]
    parsed: ParsedDose

@dataclass
class ComprehensivePreprocessingResult:
    """Comprehensive preprocessing result following exact specifications."""
    original_raw: str
    normalized: str
    tokens: List[str]
    tokens_no_stop: List[str]
    is_combination: bool
    components: List[Component]
    parsed: ParsedDose
    abbrev_expansions: Dict[str, str]
    locale_normalizations: Dict[str, str]
    preprocess_confidence: float
    entity: Optional[str] = None
    entity_confidence: float = 0.0

@dataclass
class ProcessingResult:
    """Represents the result of text processing."""
    original_text: str
    processed_text: str
    stage: ProcessingStage
    confidence: float
    metadata: Dict[str, Any]
    transformations: List[str]

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    enable_normalization: bool = True
    enable_noise_removal: bool = True
    enable_abbreviation_expansion: bool = True
    enable_token_completion: bool = True
    enable_terminology_mapping: bool = True
    enable_semantic_analysis: bool = True
    
    # Normalization settings
    aggressive_normalization: bool = False
    preserve_structure: bool = True
    
    # Noise removal settings
    remove_measurements: bool = True
    remove_frequency: bool = True
    remove_routes: bool = True
    remove_brands: bool = True
    remove_special_chars: bool = True
    remove_artifacts: bool = True
    
    # Abbreviation settings
    abbreviation_threshold: float = 0.7
    
    # Token completion settings
    completion_threshold: float = 0.7
    min_token_length: int = 3
    max_token_length: int = 15
    
    # Terminology mapping settings
    terminology_threshold: float = 0.8
    
    # Semantic analysis settings
    semantic_threshold: float = 0.6

class EnhancedMedicalPreprocessor:
    """Enhanced medical text preprocessor with comprehensive pipeline."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None,
                 vocabulary_data: Optional[pd.DataFrame] = None):
        self.config = config or PreprocessingConfig()
        self.vocabulary_data = vocabulary_data
        
        # Initialize processing modules
        self.normalizer = None  # Uses functions from handle_case_space
        self.noise_remover = MedicalNoiseRemover()
        
        # Initialize comprehensive preprocessing components
        self.abbrev_map = self._initialize_abbrev_map()
        self.british_us_map = self._initialize_british_us_map()
        self.stop_words = self._initialize_stop_words()
        self._compile_patterns()
        
        # Processing history
        self.processing_history: List[ProcessingResult] = []
    
    def _initialize_abbrev_map(self) -> Dict[str, str]:
        """Initialize abbreviation mapping dictionary."""
        return {
            'hb': 'hemoglobin',
            'hgb': 'hemoglobin', 
            'apap': 'acetaminophen',
            'xr': 'x-ray',
            'ct': 'ct scan',
            'iv': 'intravenous',
            'po': 'oral',
            'bp': 'blood pressure',
            'bs': 'blood sugar',
            'mri': 'magnetic resonance imaging',
            'ecg': 'electrocardiogram',
            'ekg': 'electrocardiogram',
            'cbc': 'complete blood count',
            'bmp': 'basic metabolic panel',
            'mi': 'myocardial infarction',
            'cva': 'cerebrovascular accident',
            'copd': 'chronic obstructive pulmonary disease',
            'chf': 'congestive heart failure',
            'dm': 'diabetes mellitus',
            'htn': 'hypertension',
            'cad': 'coronary artery disease',
            'pe': 'pulmonary embolism',
            'dvt': 'deep vein thrombosis',
            'uti': 'urinary tract infection',
            'asa': 'aspirin',
            'ace': 'angiotensin converting enzyme',
            'arb': 'angiotensin receptor blocker',
            'ccb': 'calcium channel blocker',
            'ppi': 'proton pump inhibitor',
            'nsaid': 'nonsteroidal anti-inflammatory drug'
        }
    
    def _initialize_british_us_map(self) -> Dict[str, str]:
        """Initialize British to US term mapping."""
        return {
            'paracetamol': 'acetaminophen',
            'adrenaline': 'epinephrine',
            'noradrenaline': 'norepinephrine',
            'haemoglobin': 'hemoglobin',
            'haematology': 'hematology',
            'oesophagus': 'esophagus',
            'paediatric': 'pediatric',
            'anaesthesia': 'anesthesia',
            'tumour': 'tumor',
            'colour': 'color',
            'favour': 'favor',
            'behaviour': 'behavior'
        }
    
    def _initialize_stop_words(self) -> set:
        """Initialize stop words for filtering."""
        return {
            'the', 'of', 'in', 'for', 'and', 'or', 'but', 'with', 'by', 'at',
            'to', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'under', 'over',
            'a', 'an', 'as', 'are', 'was', 'were', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'shall', 'is', 'am'
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        # Punctuation patterns
        self.punctuation_remove = re.compile(r'[`()\[\]{}:,;?!@"#$&<>\|^~`]')
        self.punctuation_keep = re.compile(r'(?<=\w)[+\-/%\.](?=\w)')
        self.punctuation_replace = re.compile(r'[^\w\s+\-/%\.]')
        
        # Dose patterns
        self.dose_pattern = re.compile(
            r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|µg|g|ml|mL|units?|iu|mcg\/kg|mg\/kg|mg\/mL|%|mcg\/mL)\b',
            re.IGNORECASE
        )
        
        # Concentration patterns
        self.concentration_pattern = re.compile(
            r'(?P<val1>\d+(?:\.\d+)?)\s*(?P<unit1>mg|mcg|g|ml|mL)\s*\/\s*(?P<val2>\d+(?:\.\d+)?)\s*(?P<unit2>mg|mcg|g|ml|mL)',
            re.IGNORECASE
        )
        
        # Combination split patterns
        self.combination_pattern = re.compile(r'\s*(\+|/| and | & )\s*')
        
        # Form detection patterns
        self.form_pattern = re.compile(
            r'\b(tablet|tab|capsule|cap|oral|injection|syrup|patch|ointment|cream|suppository|iv|po|im|sc|x-ray|xr)\b',
            re.IGNORECASE
        )
        
        # Numeric normalization patterns
        self.numeric_normalize = re.compile(r'(\d+(?:\.\d+)?)(mg|ml|mcg|g|iu|units?)\b', re.IGNORECASE)
        
        # Entity detection patterns
        self.entity_patterns = self._initialize_entity_patterns()
    
    def _initialize_entity_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize entity detection patterns and keywords."""
        return {
            'medication': {
                'keywords': ['mg', 'mcg', 'g', 'ml', 'tablet', 'capsule', 'injection', 'oral', 'iv', 'po', 'im', 'sc', 'syrup', 'patch', 'cream', 'ointment'],
                'patterns': [r'\d+\s*(mg|mcg|g|ml)', r'(tablet|capsule|injection|syrup|patch|cream|ointment)', r'(oral|iv|po|im|sc)'],
                'confidence_boost': 0.2
            },
            'procedure': {
                'keywords': ['surgery', 'operation', 'procedure', 'examination', 'test', 'scan', 'imaging', 'biopsy', 'injection', 'removal', 'repair', 'replacement', 'transplant', 'drainage', 'mri', 'ct', 'x-ray', 'ultrasound', 'endoscopy', 'colonoscopy'],
                'patterns': [r'(surgery|operation|procedure|examination|test|scan|imaging|biopsy)', r'(mri|ct|x-ray|ultrasound|endoscopy|colonoscopy)', r'(removal|repair|replacement|transplant|drainage)'],
                'confidence_boost': 0.3
            },
            'condition': {
                'keywords': ['disease', 'disorder', 'syndrome', 'condition', 'illness', 'infection', 'inflammation', 'injury', 'trauma', 'fracture', 'cancer', 'tumor', 'mass', 'lesion', 'abnormality', 'pain', 'ache', 'discomfort'],
                'patterns': [r'(disease|disorder|syndrome|condition|illness|infection|inflammation)', r'(injury|trauma|fracture|cancer|tumor|mass|lesion)', r'(pain|ache|discomfort)'],
                'confidence_boost': 0.2
            },
            'anatomy': {
                'keywords': ['heart', 'lung', 'liver', 'kidney', 'brain', 'spine', 'joint', 'muscle', 'bone', 'vessel', 'nerve', 'gland', 'chest', 'abdomen', 'pelvis', 'head', 'neck', 'back', 'extremity'],
                'patterns': [r'(heart|lung|liver|kidney|brain|spine|joint|muscle|bone)', r'(vessel|nerve|gland)', r'(chest|abdomen|pelvis|head|neck|back|extremity)'],
                'confidence_boost': 0.1
            },
            'lab': {
                'keywords': ['test', 'analysis', 'measurement', 'level', 'count', 'concentration', 'value', 'result', 'finding', 'marker', 'indicator', 'screening', 'monitoring', 'assessment', 'cbc', 'bmp', 'lft', 'rft', 'pt', 'ptt', 'inr', 'hgb', 'hct', 'wbc', 'rbc', 'plt'],
                'patterns': [r'(test|analysis|measurement|level|count|concentration|value|result|finding)', r'(cbc|bmp|lft|rft|pt|ptt|inr|hgb|hct|wbc|rbc|plt)', r'(screening|monitoring|assessment)'],
                'confidence_boost': 0.3
            },
            'symptom': {
                'keywords': ['pain', 'ache', 'discomfort', 'nausea', 'vomiting', 'diarrhea', 'constipation', 'dizziness', 'fatigue', 'weakness', 'numbness', 'tingling', 'shortness of breath', 'difficulty breathing', 'rapid heartbeat', 'slow heartbeat', 'irregular heartbeat', 'chest tightness', 'wheezing', 'coughing', 'sneezing', 'runny nose', 'stuffy nose', 'sore throat', 'ear pain', 'eye pain', 'blurred vision', 'double vision', 'hearing loss', 'ringing in ears', 'memory loss', 'confusion', 'sleep problems', 'weight loss', 'weight gain', 'loss of appetite', 'excessive thirst', 'excessive urination', 'frequent urination', 'painful urination', 'blood in urine', 'blood in stool', 'black stool', 'yellow skin', 'pale skin', 'blue skin', 'rash', 'itchy skin', 'dry skin', 'hair loss', 'nail changes'],
                'patterns': [r'(pain|ache|discomfort|nausea|vomiting|diarrhea|constipation)', r'(dizziness|fatigue|weakness|numbness|tingling)', r'(shortness of breath|difficulty breathing|rapid heartbeat|slow heartbeat|irregular heartbeat)', r'(chest tightness|wheezing|coughing|sneezing)', r'(runny nose|stuffy nose|sore throat|ear pain|eye pain)', r'(blurred vision|double vision|hearing loss|ringing in ears)', r'(memory loss|confusion|sleep problems)', r'(weight loss|weight gain|loss of appetite)', r'(excessive thirst|excessive urination|frequent urination|painful urination)', r'(blood in urine|blood in stool|black stool)', r'(yellow skin|pale skin|blue skin|rash|itchy skin|dry skin)', r'(hair loss|nail changes)'],
                'confidence_boost': 0.2
            }
        }
    
    def preprocess_text(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive text preprocessing pipeline.
        
        Args:
            text: Input text to preprocess
            context: Optional context for processing
        
        Returns:
            Comprehensive preprocessing results
        """
        if not text:
            return self._empty_result(text)
        
        current_text = text
        processing_results = []
        total_confidence = 1.0
        all_transformations = []
        
        # Stage 1: Normalization
        if self.config.enable_normalization:
            result = self._normalize_text(current_text)
            if result:
                current_text = result.processed_text
                processing_results.append(result)
                total_confidence *= result.confidence
                all_transformations.extend(result.transformations)
        
        # Stage 2: Noise Removal
        if self.config.enable_noise_removal:
            result = self._remove_noise(current_text)
            if result:
                current_text = result.processed_text
                processing_results.append(result)
                total_confidence *= result.confidence
                all_transformations.extend(result.transformations)
        
        # Stage 3: Abbreviation Expansion
        if self.config.enable_abbreviation_expansion:
            result = self._expand_abbreviations(current_text, context)
            if result:
                current_text = result.processed_text
                processing_results.append(result)
                total_confidence *= result.confidence
                all_transformations.extend(result.transformations)
        
        # Stage 4: Token Completion
        if self.config.enable_token_completion:
            result = self._complete_tokens(current_text, context)
            if result:
                current_text = result.processed_text
                processing_results.append(result)
                total_confidence *= result.confidence
                all_transformations.extend(result.transformations)
        
        # Stage 5: Terminology Mapping
        if self.config.enable_terminology_mapping:
            result = self._map_terminology(current_text, context)
            if result:
                current_text = result.processed_text
                processing_results.append(result)
                total_confidence *= result.confidence
                all_transformations.extend(result.transformations)
        
        # Stage 6: Semantic Analysis
        semantic_analysis = None
        if self.config.enable_semantic_analysis:
            semantic_analysis = self._analyze_semantics(current_text, context)
        
        # Store processing history
        self.processing_history.append(ProcessingResult(
            original_text=text,
            processed_text=current_text,
            stage=ProcessingStage.SEMANTIC_ANALYSIS,
            confidence=total_confidence,
            metadata={'context': context, 'stages_completed': len(processing_results)},
            transformations=all_transformations
        ))
        
        return {
            'original_text': text,
            'processed_text': current_text,
            'processing_stages': processing_results,
            'semantic_analysis': semantic_analysis,
            'overall_confidence': total_confidence,
            'transformations_applied': all_transformations,
            'processing_summary': self._generate_summary(processing_results)
        }
    
    def preprocess_tuple(self, input_tuple: Tuple[str, str], context: Optional[str] = None) -> Tuple[str, str]:
        """
        Preprocess a tuple of [description, entity] format.
        
        Args:
            input_tuple: Tuple of [description, entity] where description will be preprocessed
            context: Optional context for processing
        
        Returns:
            Tuple of [preprocessed_description, entity]
        """
        if not isinstance(input_tuple, (list, tuple)) or len(input_tuple) != 2:
            raise ValueError("Input must be a tuple/list of [description, entity]")
        
        description, entity = input_tuple
        
        if not description:
            return (description, entity)
        
        # Preprocess only the description
        result = self.preprocess_text(description, context)
        preprocessed_description = result['processed_text']
        
        return (preprocessed_description, entity)
    
    def _normalize_text(self, text: str) -> Optional[ProcessingResult]:
        """Normalize text using enhanced normalization."""
        try:
            normalized = normalize_medical_text(
                text, 
                aggressive=self.config.aggressive_normalization,
                preserve_structure=self.config.preserve_structure
            )
            
            components = extract_medical_components(text)
            
            transformations = []
            if normalized != text:
                transformations.append("text_normalization")
            
            return ProcessingResult(
                original_text=text,
                processed_text=normalized,
                stage=ProcessingStage.NORMALIZATION,
                confidence=0.95,
                metadata={'components': components},
                transformations=transformations
            )
        except Exception as e:
            print(f"Error in normalization: {e}")
            return None
    
    def _remove_noise(self, text: str) -> Optional[ProcessingResult]:
        """Remove medical noise from text."""
        try:
            cleaned = self.noise_remover.clean_text(
                text,
                remove_measurements=self.config.remove_measurements,
                remove_frequency=self.config.remove_frequency,
                remove_routes=self.config.remove_routes,
                remove_brands=self.config.remove_brands,
                remove_special_chars=self.config.remove_special_chars,
                remove_artifacts=self.config.remove_artifacts
            )
            
            removed_components = self.noise_remover.get_removed_components(text)
            
            transformations = []
            if cleaned != text:
                transformations.append("noise_removal")
            
            return ProcessingResult(
                original_text=text,
                processed_text=cleaned,
                stage=ProcessingStage.NOISE_REMOVAL,
                confidence=0.9,
                metadata={'removed_components': removed_components},
                transformations=transformations
            )
        except Exception as e:
            print(f"Error in noise removal: {e}")
            return None
    
    def _expand_abbreviations(self, text: str, context: Optional[str]) -> Optional[ProcessingResult]:
        """Expand medical abbreviations."""
        try:
            expansion_result = self.abbreviation_expander.expand_with_confidence(text, context)
            expanded_text = expansion_result['expanded']
            
            transformations = []
            if expanded_text != text:
                transformations.append("abbreviation_expansion")
            
            confidence = expansion_result['confidence']
            if confidence < self.config.abbreviation_threshold:
                confidence = 0.5  # Lower confidence for poor abbreviation expansion
            
            return ProcessingResult(
                original_text=text,
                processed_text=expanded_text,
                stage=ProcessingStage.ABBREVIATION_EXPANSION,
                confidence=confidence,
                metadata=expansion_result,
                transformations=transformations
            )
        except Exception as e:
            print(f"Error in abbreviation expansion: {e}")
            return None
    
    def _complete_tokens(self, text: str, context: Optional[str]) -> Optional[ProcessingResult]:
        """Complete partial/truncated tokens."""
        try:
            completion_analysis = self.token_completer.analyze_completion_opportunities(text)
            completed_text = self.token_completer.complete_text_tokens(
                text, 
                context,
                min_length=self.config.min_token_length,
                max_length=self.config.max_token_length
            )
            
            transformations = []
            if completed_text != text:
                transformations.append("token_completion")
            
            confidence = 0.8 if completion_analysis['completion_count'] > 0 else 1.0
            if completion_analysis['completion_count'] > 0:
                confidence = min(confidence, self.config.completion_threshold)
            
            return ProcessingResult(
                original_text=text,
                processed_text=completed_text,
                stage=ProcessingStage.TOKEN_COMPLETION,
                confidence=confidence,
                metadata=completion_analysis,
                transformations=transformations
            )
        except Exception as e:
            print(f"Error in token completion: {e}")
            return None
    
    def _map_terminology(self, text: str, context: Optional[str]) -> Optional[ProcessingResult]:
        """Map layman terms to clinical terminology."""
        try:
            terminology_suggestions = self.terminology_mapper.get_terminology_suggestions(text)
            clinical_text = terminology_suggestions['clinical_version']
            
            transformations = []
            if clinical_text != text:
                transformations.append("terminology_mapping")
            
            confidence = terminology_suggestions['improvement_score']
            if confidence < self.config.terminology_threshold:
                confidence = 0.7  # Default confidence for terminology mapping
            
            return ProcessingResult(
                original_text=text,
                processed_text=clinical_text,
                stage=ProcessingStage.TERMINOLOGY_MAPPING,
                confidence=confidence,
                metadata=terminology_suggestions,
                transformations=transformations
            )
        except Exception as e:
            print(f"Error in terminology mapping: {e}")
            return None
    
    def _analyze_semantics(self, text: str, context: Optional[str]) -> Optional[Dict[str, Any]]:
        """Analyze semantic context of text."""
        try:
            semantic_analysis = self.semantic_matcher.get_semantic_analysis(text)
            
            # Add context if provided
            if context:
                semantic_analysis['provided_context'] = context
            
            return semantic_analysis
        except Exception as e:
            print(f"Error in semantic analysis: {e}")
            return None
    
    def _generate_summary(self, processing_results: List[ProcessingResult]) -> Dict[str, Any]:
        """Generate processing summary."""
        if not processing_results:
            return {'stages_completed': 0, 'total_confidence': 1.0}
        
        total_confidence = 1.0
        for result in processing_results:
            total_confidence *= result.confidence
        
        transformations = []
        for result in processing_results:
            transformations.extend(result.transformations)
        
        return {
            'stages_completed': len(processing_results),
            'total_confidence': total_confidence,
            'transformations_applied': list(set(transformations)),
            'processing_successful': total_confidence > 0.5
        }
    
    def _empty_result(self, text: str) -> Dict[str, Any]:
        """Return empty result for empty input."""
        return {
            'original_text': text,
            'processed_text': text,
            'processing_stages': [],
            'semantic_analysis': None,
            'overall_confidence': 0.0,
            'transformations_applied': [],
            'processing_summary': {'stages_completed': 0, 'total_confidence': 0.0}
        }
    
    def batch_preprocess(self, texts: List[str], contexts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Preprocess multiple texts in batch.
        
        Args:
            texts: List of texts to preprocess
            contexts: Optional list of contexts
        
        Returns:
            List of preprocessing results
        """
        if contexts is None:
            contexts = [None] * len(texts)
        
        results = []
        for text, context in zip(texts, contexts):
            result = self.preprocess_text(text, context)
            results.append(result)
        
        return results
    
    def batch_preprocess_tuples(self, input_tuples: List[Tuple[str, str]], contexts: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """
        Preprocess multiple tuples in batch.
        
        Args:
            input_tuples: List of tuples [description, entity] to preprocess
            contexts: Optional list of contexts
        
        Returns:
            List of preprocessed tuples [preprocessed_description, entity]
        """
        if contexts is None:
            contexts = [None] * len(input_tuples)
        
        results = []
        for input_tuple, context in zip(input_tuples, contexts):
            result = self.preprocess_tuple(input_tuple, context)
            results.append(result)
        
        return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about processing history."""
        if not self.processing_history:
            return {'total_processed': 0}
        
        total_processed = len(self.processing_history)
        avg_confidence = sum(r.confidence for r in self.processing_history) / total_processed
        
        transformation_counts = {}
        for result in self.processing_history:
            for transformation in result.transformations:
                transformation_counts[transformation] = transformation_counts.get(transformation, 0) + 1
        
        return {
            'total_processed': total_processed,
            'average_confidence': avg_confidence,
            'transformation_counts': transformation_counts,
            'most_common_transformations': sorted(
                transformation_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def load_vocabulary_data(self, parquet_path: str):
        """Load vocabulary data for semantic matching."""
        try:
            self.vocabulary_data = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"Error loading vocabulary data: {e}")
    
    # ===== COMPREHENSIVE PREPROCESSING METHODS =====
    
    def normalize_text_comprehensive(self, text: str) -> str:
        """Step 1: Input sanitize & normalization."""
        if not text:
            return ""
        
        # Unicode normalize (NFKC)
        normalized = unicodedata.normalize('NFKC', text)
        
        # Trim whitespace
        normalized = normalized.strip()
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Replace fancy quotes and non-printables
        normalized = normalized.replace('"', '"').replace('"', '"')
        normalized = normalized.replace(''', "'").replace(''', "'")
        normalized = ''.join(char for char in normalized if char.isprintable() or char.isspace())
        
        return normalized
    
    def apply_punctuation_rules(self, text: str) -> str:
        """Step 2: Apply punctuation rules."""
        if not text:
            return ""
        
        # Remove specified punctuation
        text = self.punctuation_remove.sub('', text)
        
        # Replace other punctuation with spaces
        text = self.punctuation_replace.sub(' ', text)
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Step 3: Whitespace collapse & tokenization."""
        if not text:
            return []
        
        # Split on whitespace
        tokens = text.split()
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token]
        
        return tokens
    
    def expand_abbreviations(self, tokens: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Step 4: Word-level abbreviation expansion & British→US mapping."""
        expanded_tokens = []
        expansions = {}
        
        for token in tokens:
            # Check for abbreviation expansion
            if token in self.abbrev_map:
                expanded = self.abbrev_map[token]
                expanded_tokens.append(expanded)
                expansions[token] = expanded
            # Check for British→US mapping
            elif token in self.british_us_map:
                us_term = self.british_us_map[token]
                expanded_tokens.append(us_term)
                expansions[token] = us_term
            else:
                expanded_tokens.append(token)
        
        return expanded_tokens, expansions
    
    def normalize_numeric_formats(self, text: str) -> str:
        """Step 5: Normalize numeric formats."""
        if not text:
            return text
        
        # Standardize formats like 2.5mg → 2.5 mg, 5mg/mL → 5 mg/mL
        def normalize_match(match):
            number = match.group(1)
            unit = match.group(2)
            return f"{number} {unit}"
        
        normalized = self.numeric_normalize.sub(normalize_match, text)
        return normalized
    
    def parse_dose_and_form(self, text: str) -> ParsedDose:
        """Step 6: Parse dose, unit, concentration, form."""
        parsed = ParsedDose()
        
        if not text:
            return parsed
        
        # Parse dose
        dose_match = self.dose_pattern.search(text)
        if dose_match:
            parsed.dose_value = float(dose_match.group('value'))
            parsed.dose_unit = dose_match.group('unit').lower()
        
        # Parse concentration
        conc_match = self.concentration_pattern.search(text)
        if conc_match:
            parsed.concentration = {
                'val1': float(conc_match.group('val1')),
                'unit1': conc_match.group('unit1').lower(),
                'val2': float(conc_match.group('val2')),
                'unit2': conc_match.group('unit2').lower()
            }
        
        # Parse form
        form_match = self.form_pattern.search(text)
        if form_match:
            form = form_match.group(1).lower()
            # Map common forms
            form_mapping = {
                'tab': 'tablet',
                'cap': 'capsule',
                'iv': 'injection',
                'po': 'oral',
                'im': 'injection',
                'sc': 'injection',
                'x-ray': 'imaging',
                'xr': 'imaging'
            }
            parsed.form = form_mapping.get(form, form)
        
        return parsed
    
    def detect_combination_products(self, text: str) -> Tuple[bool, List[str]]:
        """Step 7: Detect combination products."""
        if not text:
            return False, []
        
        # Check for combination indicators
        if self.combination_pattern.search(text):
            components = self.combination_pattern.split(text)
            # Filter out empty components and separators
            components = [comp.strip() for comp in components if comp.strip() and comp not in ['+', '/', 'and', '&']]
            return True, components
        
        return False, [text]
    
    def filter_stop_words(self, tokens: List[str]) -> List[str]:
        """Step 8: Filter stop-words and non-informative tokens."""
        filtered = []
        
        for token in tokens:
            # Keep if not a stop word
            if token.lower() not in self.stop_words:
                # Keep if it's a short token that exists in abbrev_map
                if len(token) < 2 and token in self.abbrev_map:
                    filtered.append(token)
                # Keep if it's 2+ characters
                elif len(token) >= 2:
                    filtered.append(token)
        
        return filtered
    
    def apply_fuzzy_correction(self, tokens: List[str]) -> List[str]:
        """Step 9: Optional fuzzy correction for ingredients."""
        vocab = getattr(self, 'ingredient_vocabulary', None)
        if not vocab:
            return tokens

        if getattr(self, '_ingredient_terms', None) is None:
            self._ingredient_terms = sorted({term.lower() for term in vocab})

        corrected_tokens = []
        for token in tokens:
            token_norm = token.lower()
            if token_norm in vocab:
                corrected_tokens.append(token)
                continue

            matches = rf_process.extract(
                token_norm,
                self._ingredient_terms,
                scorer=fuzz.ratio,
                score_cutoff=90,
                limit=3,
            )
            if matches:
                best_term, score, _ = matches[0]
                if score >= 90:
                    corrected_tokens.append(best_term)
                    continue

            corrected_tokens.append(token)

        return corrected_tokens
    
    def detect_entity(self, text: str, tokens: List[str]) -> Tuple[Optional[str], float]:
        """Detect entity type based on text content and tokens."""
        if not text or not tokens:
            return None, 0.0
        
        text_lower = text.lower()
        entity_scores = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            score = 0.0
            keyword_matches = 0
            pattern_matches = 0
            
            # Check keyword matches in both text and tokens
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    keyword_matches += 1
                    score += 0.2  # Increased weight for keyword matches
                # Also check in individual tokens
                for token in tokens:
                    if keyword in token.lower():
                        keyword_matches += 1
                        score += 0.1
            
            # Check pattern matches
            for pattern in patterns['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    pattern_matches += 1
                    score += 0.3  # Increased weight for pattern matches
            
            # Apply confidence boost
            if keyword_matches > 0 or pattern_matches > 0:
                score += patterns['confidence_boost']
            
            # Simple scoring - don't over-normalize
            if score > 0:
                entity_scores[entity_type] = min(score, 1.0)
        
        # Find the entity with highest score
        if entity_scores:
            best_entity = max(entity_scores, key=entity_scores.get)
            best_score = entity_scores[best_entity]
            
            # Lower threshold for entity detection
            if best_score > 0.1:
                return best_entity, best_score
        
        return None, 0.0
    
    def process_component(self, component_text: str) -> Component:
        """Process a single component of a combination product."""
        # Apply full preprocessing pipeline to component
        normalized = self.normalize_text_comprehensive(component_text)
        normalized = self.apply_punctuation_rules(normalized)
        normalized = self.normalize_numeric_formats(normalized)
        
        tokens = self.tokenize_text(normalized)
        expanded_tokens, expansions = self.expand_abbreviations(tokens)
        filtered_tokens = self.filter_stop_words(expanded_tokens)
        
        parsed = self.parse_dose_and_form(component_text)
        
        return Component(
            component_raw=component_text,
            normalized=normalized,
            tokens=filtered_tokens,
            parsed=parsed
        )
    
    def preprocess_comprehensive(self, text: str, entity: Optional[str] = None) -> ComprehensivePreprocessingResult:
        """Main comprehensive preprocessing pipeline following exact specifications."""
        if not text:
            return ComprehensivePreprocessingResult(
                original_raw=text,
                normalized="",
                tokens=[],
                tokens_no_stop=[],
                is_combination=False,
                components=[],
                parsed=ParsedDose(),
                abbrev_expansions={},
                locale_normalizations={},
                preprocess_confidence=0.0,
                entity=None,
                entity_confidence=0.0
            )
        
        original_raw = text
        
        # Step 1: Input sanitize & normalization
        normalized = self.normalize_text_comprehensive(text)
        
        # Step 2: Apply punctuation rules
        normalized = self.apply_punctuation_rules(normalized)
        
        # Step 5: Normalize numeric formats
        normalized = self.normalize_numeric_formats(normalized)
        
        # Step 3: Tokenize
        tokens = self.tokenize_text(normalized)
        
        # Step 4: Expand abbreviations and apply locale mappings
        expanded_tokens, expansions = self.expand_abbreviations(tokens)
        
        # Step 8: Filter stop words
        filtered_tokens = self.filter_stop_words(expanded_tokens)
        
        # Step 9: Optional fuzzy correction
        if hasattr(self, 'ingredient_vocabulary') and self.ingredient_vocabulary:
            filtered_tokens = self.apply_fuzzy_correction(filtered_tokens)
        
        # Step 7: Detect combination products
        is_combination, components_text = self.detect_combination_products(text)
        
        # Process components if combination
        components = []
        if is_combination:
            for comp_text in components_text:
                component = self.process_component(comp_text)
                components.append(component)
        else:
            # Single component
            component = self.process_component(text)
            components.append(component)
        
        # Step 6: Parse dose and form for the main text
        parsed = self.parse_dose_and_form(text)
        
        # Calculate confidence
        confidence = self._calculate_comprehensive_confidence(expansions, filtered_tokens, is_combination)
        
        # Use provided entity or detect if not provided
        if entity is None:
            detected_entity, entity_confidence = self.detect_entity(original_raw, filtered_tokens)
        else:
            detected_entity = entity
            entity_confidence = 1.0  # High confidence for provided entity
        
        return ComprehensivePreprocessingResult(
            original_raw=original_raw,
            normalized=normalized,
            tokens=tokens,
            tokens_no_stop=filtered_tokens,
            is_combination=is_combination,
            components=components,
            parsed=parsed,
            abbrev_expansions=expansions,
            locale_normalizations={k: v for k, v in expansions.items() if k in self.british_us_map},
            preprocess_confidence=confidence,
            entity=detected_entity,
            entity_confidence=entity_confidence
        )
    
    def _calculate_comprehensive_confidence(self, expansions: Dict[str, str], 
                            filtered_tokens: List[str], 
                            is_combination: bool) -> float:
        """Calculate preprocessing confidence score."""
        base_confidence = 1.0
        
        # Reduce confidence for many expansions (might indicate uncertainty)
        if len(expansions) > 3:
            base_confidence *= 0.9
        
        # Reduce confidence for very short token lists
        if len(filtered_tokens) < 2:
            base_confidence *= 0.8
        
        # Slight reduction for combination products (more complex)
        if is_combination:
            base_confidence *= 0.95
        
        return min(base_confidence, 1.0)
    
    def augment_abbrev_map_from_vocabulary(self, vocabulary_data):
        """Auto-augment abbrev_map from RxNorm/SNOMED data."""
        if vocabulary_data is None:
            return
        
        # Find short STR entries that map to longer ones
        for _, row in vocabulary_data.iterrows():
            if 'STR' in row and 'CUI' in row:
                str_val = str(row['STR']).strip()
                cui = str(row['CUI']).strip()
                
                # Look for short strings (< 6 chars) that might be abbreviations
                if len(str_val) < 6 and len(str_val) > 1:
                    # Find other entries with same CUI but longer STR
                    same_cui = vocabulary_data[vocabulary_data['CUI'] == cui]
                    longer_strs = same_cui[same_cui['STR'].str.len() > len(str_val)]
                    
                    if not longer_strs.empty:
                        # Use the longest STR as the expansion
                        longest_str = longer_strs['STR'].str.len().idxmax()
                        expansion = str(longer_strs.loc[longest_str, 'STR']).strip().lower()
                        
                        if expansion and expansion != str_val.lower():
                            self.abbrev_map[str_val.lower()] = expansion.lower()

# Convenience functions
def preprocess_medical_text(text: str, config: Optional[PreprocessingConfig] = None,
                          context: Optional[str] = None) -> Dict[str, Any]:
    """
    Preprocess medical text using the enhanced pipeline.
    
    Args:
        text: Input medical text
        config: Optional preprocessing configuration
        context: Optional context for processing
    
    Returns:
        Preprocessing results
    """
    preprocessor = EnhancedMedicalPreprocessor(config)
    return preprocessor.preprocess_text(text, context)

def preprocess_medical_text_comprehensive(text: str, entity: Optional[str] = None, vocabulary_data: Optional[pd.DataFrame] = None) -> ComprehensivePreprocessingResult:
    """
    Preprocess medical text using the comprehensive pipeline following exact specifications.
    
    Args:
        text: Input medical text
        entity: Optional entity type (e.g., 'medication', 'procedure', 'condition')
        vocabulary_data: Optional vocabulary data for augmentation
    
    Returns:
        Comprehensive preprocessing result
    """
    preprocessor = EnhancedMedicalPreprocessor(vocabulary_data=vocabulary_data)
    if vocabulary_data is not None:
        preprocessor.augment_abbrev_map_from_vocabulary(vocabulary_data)
    return preprocessor.preprocess_comprehensive(text, entity)

def preprocess_medical_tuple_comprehensive(input_tuple: Union[List[str], Tuple[str, str]], vocabulary_data: Optional[pd.DataFrame] = None) -> ComprehensivePreprocessingResult:
    """
    Preprocess medical text tuple in format ["query", "entity"] using comprehensive pipeline.
    
    Args:
        input_tuple: Tuple/list of [query, entity] format
        vocabulary_data: Optional vocabulary data for augmentation
    
    Returns:
        Comprehensive preprocessing result
    """
    if not isinstance(input_tuple, (list, tuple)) or len(input_tuple) != 2:
        raise ValueError("Input must be a tuple/list of [query, entity]")
    
    query, entity = input_tuple
    return preprocess_medical_text_comprehensive(query, entity, vocabulary_data)

def batch_preprocess_medical_texts(texts: List[str], config: Optional[PreprocessingConfig] = None,
                                 contexts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Batch preprocess multiple medical texts.
    
    Args:
        texts: List of medical texts
        config: Optional preprocessing configuration
        contexts: Optional list of contexts
    
    Returns:
        List of preprocessing results
    """
    preprocessor = EnhancedMedicalPreprocessor(config)
    return preprocessor.batch_preprocess(texts, contexts)

def preprocess_medical_tuple(input_tuple: Tuple[str, str], config: Optional[PreprocessingConfig] = None,
                           context: Optional[str] = None) -> Tuple[str, str]:
    """
    Preprocess a medical tuple [description, entity].
    
    Args:
        input_tuple: Tuple of [description, entity]
        config: Optional preprocessing configuration
        context: Optional context for processing
    
    Returns:
        Tuple of [preprocessed_description, entity]
    """
    preprocessor = EnhancedMedicalPreprocessor(config)
    return preprocessor.preprocess_tuple(input_tuple, context)

def batch_preprocess_medical_tuples(input_tuples: List[Tuple[str, str]], config: Optional[PreprocessingConfig] = None,
                                   contexts: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Batch preprocess multiple medical tuples.
    
    Args:
        input_tuples: List of tuples [description, entity]
        config: Optional preprocessing configuration
        contexts: Optional list of contexts
    
    Returns:
        List of preprocessed tuples [preprocessed_description, entity]
    """
    preprocessor = EnhancedMedicalPreprocessor(config)
    return preprocessor.batch_preprocess_tuples(input_tuples, contexts)

if __name__ == "__main__":
    # Test the enhanced preprocessor
    test_texts = [
        "mri pelvis",
        "hcv rna",
        "depression screen",
        "induction for prom",
        "polypectomy",
        "transvaginal ultrasound procedu",
        "coronary angioplasty",
        "metformin 500 mg oral tablet twice daily",
        "chest pain and heart problems",
        "cbc and bmp results are normal"
    ]
    
    # Test tuples
    test_tuples = [
        ["mri pelvis", "procedure"],
        ["hcv rna", "test"],
        ["depression screen", "assessment"],
        ["metformin 500 mg oral tablet twice daily", "medication"],
        ["chest pain and heart problems", "symptom"]
    ]
    
    # Create custom configuration
    config = PreprocessingConfig(
        enable_normalization=True,
        enable_noise_removal=True,
        enable_abbreviation_expansion=True,
        enable_token_completion=True,
        enable_terminology_mapping=True,
        enable_semantic_analysis=True,
        aggressive_normalization=False,
        remove_measurements=True,
        abbreviation_threshold=0.7,
        completion_threshold=0.7,
        terminology_threshold=0.8
    )
    
    preprocessor = EnhancedMedicalPreprocessor(config)
    
    print("=== Testing Single Text Processing ===")
    for text in test_texts:
        print(f"Original: {text}")
        result = preprocessor.preprocess_text(text)
        print(f"Processed: {result['processed_text']}")
        print(f"Confidence: {result['overall_confidence']:.2f}")
        print(f"Transformations: {result['transformations_applied']}")
        print("---")
    
    print("\n=== Testing Tuple Processing ===")
    for input_tuple in test_tuples:
        print(f"Original Tuple: {input_tuple}")
        result_tuple = preprocessor.preprocess_tuple(input_tuple)
        print(f"Processed Tuple: {result_tuple}")
        print("---")
