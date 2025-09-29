"""
Medical Matcher with Multi-Strategy Matching

Provides comprehensive medical term matching using multiple strategies:
- Exact matching
- Fuzzy matching
- Semantic matching
- Abbreviation matching
- Token completion matching
- Layman-clinical mapping
- Context-aware matching

Integrates with the enhanced preprocessor for optimal results.
"""

import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from rapidfuzz import fuzz, process

# Import our custom modules
# Removed circular import - EnhancedMedicalPreprocessor not needed here
from semantic_matcher import MedicalSemanticMatcher, SemanticMatch
from medical_abbreviations import MedicalAbbreviationExpander
from token_completion import MedicalTokenCompleter
from layman_clinical_mapper import LaymanClinicalMapper

class MatchingStrategy(Enum):
    """Enumeration of matching strategies."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    ABBREVIATION = "abbreviation"
    TOKEN_COMPLETION = "token_completion"
    TERMINOLOGY_MAPPING = "terminology_mapping"
    CONTEXT_AWARE = "context_aware"
    MULTI_STRATEGY = "multi_strategy"

@dataclass
class MatchResult:
    """Represents a match result with comprehensive information."""
    query: str
    matched_term: str
    confidence: float
    strategy: MatchingStrategy
    concept_id: Optional[str] = None
    semantic_type: Optional[str] = None
    system: Optional[str] = None
    code: Optional[str] = None
    description: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class MatchingConfig:
    """Configuration for medical matching."""
    # Strategy weights
    exact_weight: float = 1.0
    fuzzy_weight: float = 0.8
    semantic_weight: float = 0.7
    abbreviation_weight: float = 0.6
    token_completion_weight: float = 0.5
    terminology_mapping_weight: float = 0.6
    
    # Thresholds
    exact_threshold: float = 1.0
    fuzzy_threshold: float = 0.8
    semantic_threshold: float = 0.6
    abbreviation_threshold: float = 0.7
    token_completion_threshold: float = 0.7
    terminology_threshold: float = 0.8
    
    # Multi-strategy settings
    enable_exact_matching: bool = True
    enable_fuzzy_matching: bool = True
    enable_semantic_matching: bool = True
    enable_abbreviation_matching: bool = True
    enable_token_completion: bool = True
    enable_terminology_mapping: bool = True
    enable_context_aware_matching: bool = True
    
    # Preprocessing settings
    enable_preprocessing: bool = True
    preprocessing_config: Optional[Any] = None

class MedicalMatcher:
    """Comprehensive medical term matcher with multiple strategies."""
    
    def __init__(self, vocabulary_data: Optional[pd.DataFrame] = None,
                 config: Optional[MatchingConfig] = None):
        self.vocabulary_data = vocabulary_data
        self.config = config or MatchingConfig()
        
        # Initialize matching modules
        self.preprocessor = EnhancedMedicalPreprocessor(
            self.config.preprocessing_config or PreprocessingConfig()
        )
        self.semantic_matcher = MedicalSemanticMatcher(vocabulary_data)
        self.abbreviation_expander = MedicalAbbreviationExpander()
        self.token_completer = MedicalTokenCompleter()
        self.terminology_mapper = LaymanClinicalMapper()
        
        # Matching history
        self.matching_history: List[MatchResult] = []
    
    def match_medical_term(self, query: str, context: Optional[str] = None,
                          max_results: int = 5) -> List[MatchResult]:
        """
        Match a medical term using multiple strategies.
        
        Args:
            query: Query term to match
            context: Optional context for matching
            max_results: Maximum number of results to return
        
        Returns:
            List of match results sorted by confidence
        """
        if not query:
            return []
        
        # Preprocess query if enabled
        processed_query = query
        if self.config.enable_preprocessing:
            preprocessing_result = self.preprocessor.preprocess_text(query, context)
            processed_query = preprocessing_result['processed_text']
        
        all_matches = []
        
        # Strategy 1: Exact Matching
        if self.config.enable_exact_matching:
            exact_matches = self._exact_match(processed_query)
            all_matches.extend(exact_matches)
        
        # Strategy 2: Fuzzy Matching
        if self.config.enable_fuzzy_matching:
            fuzzy_matches = self._fuzzy_match(processed_query)
            all_matches.extend(fuzzy_matches)
        
        # Strategy 3: Semantic Matching
        if self.config.enable_semantic_matching:
            semantic_matches = self._semantic_match(processed_query, context)
            all_matches.extend(semantic_matches)
        
        # Strategy 4: Abbreviation Matching
        if self.config.enable_abbreviation_matching:
            abbreviation_matches = self._abbreviation_match(processed_query, context)
            all_matches.extend(abbreviation_matches)
        
        # Strategy 5: Token Completion Matching
        if self.config.enable_token_completion:
            completion_matches = self._token_completion_match(processed_query, context)
            all_matches.extend(completion_matches)
        
        # Strategy 6: Terminology Mapping
        if self.config.enable_terminology_mapping:
            terminology_matches = self._terminology_mapping_match(processed_query, context)
            all_matches.extend(terminology_matches)
        
        # Strategy 7: Context-Aware Matching
        if self.config.enable_context_aware_matching and context:
            context_matches = self._context_aware_match(processed_query, context)
            all_matches.extend(context_matches)
        
        # Combine and rank results
        combined_matches = self._combine_matches(all_matches)
        
        # Sort by confidence and return top results
        combined_matches.sort(key=lambda x: x.confidence, reverse=True)
        
        # Store in history
        for match in combined_matches[:max_results]:
            self.matching_history.append(match)
        
        return combined_matches[:max_results]
    
    def match_medical_tuple(self, input_tuple: Tuple[str, str], context: Optional[str] = None,
                           max_results: int = 5) -> Tuple[List[MatchResult], str]:
        """
        Match a medical tuple [description, entity] using multiple strategies.
        
        Args:
            input_tuple: Tuple of [description, entity] where description will be matched
            context: Optional context for matching
            max_results: Maximum number of results to return
        
        Returns:
            Tuple of (match_results, entity) where entity is preserved
        """
        if not isinstance(input_tuple, (list, tuple)) or len(input_tuple) != 2:
            raise ValueError("Input must be a tuple/list of [description, entity]")
        
        description, entity = input_tuple
        
        if not description:
            return ([], entity)
        
        # Match the description
        matches = self.match_medical_term(description, context, max_results)
        
        return (matches, entity)
    
    def _exact_match(self, query: str) -> List[MatchResult]:
        """Perform exact matching against vocabulary."""
        if not self.vocabulary_data is None:
            return []
        
        exact_matches = []
        query_lower = query.lower()
        
        # Check for exact matches in vocabulary
        exact_rows = self.vocabulary_data[
            self.vocabulary_data['STR'].str.lower() == query_lower
        ]
        
        for _, row in exact_rows.iterrows():
            match = MatchResult(
                query=query,
                matched_term=row['STR'],
                confidence=self.config.exact_weight,
                strategy=MatchingStrategy.EXACT,
                concept_id=row.get('CUI'),
                semantic_type=row.get('STY'),
                system=row.get('System'),
                code=row.get('CODE'),
                description=row['STR'],
                metadata={'exact_match': True}
            )
            exact_matches.append(match)
        
        return exact_matches
    
    def _fuzzy_match(self, query: str) -> List[MatchResult]:
        """Perform fuzzy matching against vocabulary."""
        if self.vocabulary_data is None:
            return []
        
        fuzzy_matches = []
        vocabulary_terms = self.vocabulary_data['STR'].astype(str).tolist()
        
        # Use rapidfuzz for fuzzy matching
        fuzzy_results = process.extract(
            query,
            vocabulary_terms,
            scorer=fuzz.WRatio,
            limit=10,
            score_cutoff=int(self.config.fuzzy_threshold * 100)
        )
        
        for term, score, _ in fuzzy_results:
            confidence = (score / 100.0) * self.config.fuzzy_weight
            
            # Get additional information from vocabulary
            vocab_row = self.vocabulary_data[
                self.vocabulary_data['STR'] == term
            ].iloc[0] if not self.vocabulary_data[
                self.vocabulary_data['STR'] == term
            ].empty else None
            
            match = MatchResult(
                query=query,
                matched_term=term,
                confidence=confidence,
                strategy=MatchingStrategy.FUZZY,
                concept_id=vocab_row['CUI'] if vocab_row is not None else None,
                semantic_type=vocab_row['STY'] if vocab_row is not None else None,
                system=vocab_row['System'] if vocab_row is not None else None,
                code=vocab_row['CODE'] if vocab_row is not None else None,
                description=term,
                metadata={'fuzzy_score': score}
            )
            fuzzy_matches.append(match)
        
        return fuzzy_matches
    
    def _semantic_match(self, query: str, context: Optional[str]) -> List[MatchResult]:
        """Perform semantic matching."""
        if self.vocabulary_data is None:
            return []
        
        semantic_matches = []
        vocabulary_terms = self.vocabulary_data['STR'].astype(str).tolist()
        
        # Use semantic matcher
        matches = self.semantic_matcher.find_semantic_matches(
            query, vocabulary_terms, context, self.config.semantic_threshold
        )
        
        for match in matches:
            confidence = match.confidence * self.config.semantic_weight
            
            # Get additional information from vocabulary
            vocab_row = self.vocabulary_data[
                self.vocabulary_data['STR'] == match.match
            ].iloc[0] if not self.vocabulary_data[
                self.vocabulary_data['STR'] == match.match
            ].empty else None
            
            match_result = MatchResult(
                query=query,
                matched_term=match.match,
                confidence=confidence,
                strategy=MatchingStrategy.SEMANTIC,
                concept_id=vocab_row['CUI'] if vocab_row is not None else None,
                semantic_type=vocab_row['STY'] if vocab_row is not None else None,
                system=vocab_row['System'] if vocab_row is not None else None,
                code=vocab_row['CODE'] if vocab_row is not None else None,
                description=match.match,
                context=context,
                metadata={'semantic_method': match.method}
            )
            semantic_matches.append(match_result)
        
        return semantic_matches
    
    def _abbreviation_match(self, query: str, context: Optional[str]) -> List[MatchResult]:
        """Perform abbreviation-based matching."""
        if self.vocabulary_data is None:
            return []
        
        abbreviation_matches = []
        
        # Expand abbreviations in query
        expansion_result = self.abbreviation_expander.expand_with_confidence(query, context)
        
        if expansion_result['confidence'] >= self.config.abbreviation_threshold:
            expanded_query = expansion_result['expanded']
            
            # Try to match expanded query
            vocabulary_terms = self.vocabulary_data['STR'].astype(str).tolist()
            
            fuzzy_results = process.extract(
                expanded_query,
                vocabulary_terms,
                scorer=fuzz.WRatio,
                limit=5,
                score_cutoff=70
            )
            
            for term, score, _ in fuzzy_results:
                confidence = (score / 100.0) * self.config.abbreviation_weight
                
                # Get additional information from vocabulary
                vocab_row = self.vocabulary_data[
                    self.vocabulary_data['STR'] == term
                ].iloc[0] if not self.vocabulary_data[
                    self.vocabulary_data['STR'] == term
                ].empty else None
                
                match = MatchResult(
                    query=query,
                    matched_term=term,
                    confidence=confidence,
                    strategy=MatchingStrategy.ABBREVIATION,
                    concept_id=vocab_row['CUI'] if vocab_row is not None else None,
                    semantic_type=vocab_row['STY'] if vocab_row is not None else None,
                    system=vocab_row['System'] if vocab_row is not None else None,
                    code=vocab_row['CODE'] if vocab_row is not None else None,
                    description=term,
                    context=context,
                    metadata={
                        'abbreviation_expansion': expansion_result,
                        'fuzzy_score': score
                    }
                )
                abbreviation_matches.append(match)
        
        return abbreviation_matches
    
    def _token_completion_match(self, query: str, context: Optional[str]) -> List[MatchResult]:
        """Perform token completion-based matching."""
        if self.vocabulary_data is None:
            return []
        
        completion_matches = []
        
        # Analyze completion opportunities
        completion_analysis = self.token_completer.analyze_completion_opportunities(query)
        
        if completion_analysis['completion_count'] > 0:
            # Try to complete tokens and match
            completed_query = self.token_completer.complete_text_tokens(query, context)
            
            if completed_query != query:
                vocabulary_terms = self.vocabulary_data['STR'].astype(str).tolist()
                
                fuzzy_results = process.extract(
                    completed_query,
                    vocabulary_terms,
                    scorer=fuzz.WRatio,
                    limit=5,
                    score_cutoff=70
                )
                
                for term, score, _ in fuzzy_results:
                    confidence = (score / 100.0) * self.config.token_completion_weight
                    
                    # Get additional information from vocabulary
                    vocab_row = self.vocabulary_data[
                        self.vocabulary_data['STR'] == term
                    ].iloc[0] if not self.vocabulary_data[
                        self.vocabulary_data['STR'] == term
                    ].empty else None
                    
                    match = MatchResult(
                        query=query,
                        matched_term=term,
                        confidence=confidence,
                        strategy=MatchingStrategy.TOKEN_COMPLETION,
                        concept_id=vocab_row['CUI'] if vocab_row is not None else None,
                        semantic_type=vocab_row['STY'] if vocab_row is not None else None,
                        system=vocab_row['System'] if vocab_row is not None else None,
                        code=vocab_row['CODE'] if vocab_row is not None else None,
                        description=term,
                        context=context,
                        metadata={
                            'completion_analysis': completion_analysis,
                            'completed_query': completed_query,
                            'fuzzy_score': score
                        }
                    )
                    completion_matches.append(match)
        
        return completion_matches
    
    def _terminology_mapping_match(self, query: str, context: Optional[str]) -> List[MatchResult]:
        """Perform terminology mapping-based matching."""
        if self.vocabulary_data is None:
            return []
        
        terminology_matches = []
        
        # Map layman terms to clinical terminology
        clinical_query = self.terminology_mapper.map_layman_to_clinical(query, context)
        
        if clinical_query != query:
            vocabulary_terms = self.vocabulary_data['STR'].astype(str).tolist()
            
            fuzzy_results = process.extract(
                clinical_query,
                vocabulary_terms,
                scorer=fuzz.WRatio,
                limit=5,
                score_cutoff=70
            )
            
            for term, score, _ in fuzzy_results:
                confidence = (score / 100.0) * self.config.terminology_mapping_weight
                
                # Get additional information from vocabulary
                vocab_row = self.vocabulary_data[
                    self.vocabulary_data['STR'] == term
                ].iloc[0] if not self.vocabulary_data[
                    self.vocabulary_data['STR'] == term
                ].empty else None
                
                match = MatchResult(
                    query=query,
                    matched_term=term,
                    confidence=confidence,
                    strategy=MatchingStrategy.TERMINOLOGY_MAPPING,
                    concept_id=vocab_row['CUI'] if vocab_row is not None else None,
                    semantic_type=vocab_row['STY'] if vocab_row is not None else None,
                    system=vocab_row['System'] if vocab_row is not None else None,
                    code=vocab_row['CODE'] if vocab_row is not None else None,
                    description=term,
                    context=context,
                    metadata={
                        'clinical_query': clinical_query,
                        'fuzzy_score': score
                    }
                )
                terminology_matches.append(match)
        
        return terminology_matches
    
    def _context_aware_match(self, query: str, context: str) -> List[MatchResult]:
        """Perform context-aware matching."""
        if self.vocabulary_data is None:
            return []
        
        context_matches = []
        
        # Use context to filter and rank matches
        vocabulary_terms = self.vocabulary_data['STR'].astype(str).tolist()
        
        # Get semantic context
        semantic_context = self.semantic_matcher.extract_semantic_context(context)
        
        # Filter vocabulary by semantic type if available
        filtered_vocabulary = self.vocabulary_data.copy()
        
        if semantic_context.get('semantic_types'):
            # Filter by semantic types
            semantic_types = semantic_context['semantic_types']
            filtered_vocabulary = filtered_vocabulary[
                filtered_vocabulary['STY'].isin(semantic_types)
            ]
        
        if not filtered_vocabulary.empty:
            filtered_terms = filtered_vocabulary['STR'].astype(str).tolist()
            
            fuzzy_results = process.extract(
                query,
                filtered_terms,
                scorer=fuzz.WRatio,
                limit=5,
                score_cutoff=70
            )
            
            for term, score, _ in fuzzy_results:
                confidence = (score / 100.0) * 0.9  # High weight for context-aware
                
                # Get additional information from vocabulary
                vocab_row = filtered_vocabulary[
                    filtered_vocabulary['STR'] == term
                ].iloc[0] if not filtered_vocabulary[
                    filtered_vocabulary['STR'] == term
                ].empty else None
                
                match = MatchResult(
                    query=query,
                    matched_term=term,
                    confidence=confidence,
                    strategy=MatchingStrategy.CONTEXT_AWARE,
                    concept_id=vocab_row['CUI'] if vocab_row is not None else None,
                    semantic_type=vocab_row['STY'] if vocab_row is not None else None,
                    system=vocab_row['System'] if vocab_row is not None else None,
                    code=vocab_row['CODE'] if vocab_row is not None else None,
                    description=term,
                    context=context,
                    metadata={
                        'semantic_context': semantic_context,
                        'fuzzy_score': score
                    }
                )
                context_matches.append(match)
        
        return context_matches
    
    def _combine_matches(self, all_matches: List[MatchResult]) -> List[MatchResult]:
        """Combine and deduplicate matches from different strategies."""
        # Group matches by matched term
        match_groups = {}
        
        for match in all_matches:
            key = match.matched_term.lower()
            if key not in match_groups:
                match_groups[key] = []
            match_groups[key].append(match)
        
        # Combine matches for each term
        combined_matches = []
        
        for term, matches in match_groups.items():
            if len(matches) == 1:
                combined_matches.append(matches[0])
            else:
                # Combine multiple matches for the same term
                best_match = max(matches, key=lambda x: x.confidence)
                
                # Boost confidence for multiple strategy matches
                strategy_count = len(set(match.strategy for match in matches))
                confidence_boost = min(0.1 * strategy_count, 0.3)
                best_match.confidence = min(best_match.confidence + confidence_boost, 1.0)
                
                # Add metadata about combined strategies
                best_match.metadata = best_match.metadata or {}
                best_match.metadata['combined_strategies'] = [
                    match.strategy.value for match in matches
                ]
                best_match.metadata['strategy_count'] = strategy_count
                
                combined_matches.append(best_match)
        
        return combined_matches
    
    def batch_match(self, queries: List[str], contexts: Optional[List[str]] = None,
                   max_results: int = 5) -> List[List[MatchResult]]:
        """
        Batch match multiple queries.
        
        Args:
            queries: List of queries to match
            contexts: Optional list of contexts
            max_results: Maximum results per query
        
        Returns:
            List of match results for each query
        """
        if contexts is None:
            contexts = [None] * len(queries)
        
        results = []
        for query, context in zip(queries, contexts):
            matches = self.match_medical_term(query, context, max_results)
            results.append(matches)
        
        return results
    
    def batch_match_tuples(self, input_tuples: List[Tuple[str, str]], 
                          contexts: Optional[List[str]] = None,
                          max_results: int = 5) -> List[Tuple[List[MatchResult], str]]:
        """
        Batch match multiple medical tuples.
        
        Args:
            input_tuples: List of tuples [description, entity] to match
            contexts: Optional list of contexts
            max_results: Maximum number of results per tuple
        
        Returns:
            List of tuples (match_results, entity)
        """
        if contexts is None:
            contexts = [None] * len(input_tuples)
        
        results = []
        for input_tuple, context in zip(input_tuples, contexts):
            result = self.match_medical_tuple(input_tuple, context, max_results)
            results.append(result)
        
        return results
    
    def get_matching_statistics(self) -> Dict[str, Any]:
        """Get statistics about matching history."""
        if not self.matching_history:
            return {'total_matches': 0}
        
        total_matches = len(self.matching_history)
        avg_confidence = sum(match.confidence for match in self.matching_history) / total_matches
        
        strategy_counts = {}
        for match in self.matching_history:
            strategy = match.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_matches': total_matches,
            'average_confidence': avg_confidence,
            'strategy_counts': strategy_counts,
            'most_used_strategies': sorted(
                strategy_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def load_vocabulary_data(self, parquet_path: str):
        """Load vocabulary data from parquet file."""
        try:
            self.vocabulary_data = pd.read_parquet(parquet_path)
            self.semantic_matcher.load_vocabulary_data(parquet_path)
            self.token_completer.load_vocabulary_from_parquet(parquet_path)
        except Exception as e:
            print(f"Error loading vocabulary data: {e}")

# Convenience functions
def match_medical_term(query: str, vocabulary_data: pd.DataFrame,
                      context: Optional[str] = None,
                      config: Optional[MatchingConfig] = None) -> List[MatchResult]:
    """
    Match a medical term using the comprehensive matcher.
    
    Args:
        query: Query term to match
        vocabulary_data: Vocabulary data for matching
        context: Optional context for matching
        config: Optional matching configuration
    
    Returns:
        List of match results
    """
    matcher = MedicalMatcher(vocabulary_data, config)
    return matcher.match_medical_term(query, context)

def batch_match_medical_terms(queries: List[str], vocabulary_data: pd.DataFrame,
                            contexts: Optional[List[str]] = None,
                            config: Optional[MatchingConfig] = None) -> List[List[MatchResult]]:
    """
    Batch match multiple medical terms.
    
    Args:
        queries: List of queries to match
        vocabulary_data: Vocabulary data for matching
        contexts: Optional list of contexts
        config: Optional matching configuration
    
    Returns:
        List of match results for each query
    """
    matcher = MedicalMatcher(vocabulary_data, config)
    return matcher.batch_match(queries, contexts)

def match_medical_tuple(input_tuple: Tuple[str, str], vocabulary_data: pd.DataFrame,
                       context: Optional[str] = None,
                       config: Optional[MatchingConfig] = None) -> Tuple[List[MatchResult], str]:
    """
    Match a medical tuple using the comprehensive matcher.
    
    Args:
        input_tuple: Tuple of [description, entity] to match
        vocabulary_data: Vocabulary data for matching
        context: Optional context for matching
        config: Optional matching configuration
    
    Returns:
        Tuple of (match_results, entity)
    """
    matcher = MedicalMatcher(vocabulary_data, config)
    return matcher.match_medical_tuple(input_tuple, context)

def batch_match_medical_tuples(input_tuples: List[Tuple[str, str]], vocabulary_data: pd.DataFrame,
                              contexts: Optional[List[str]] = None,
                              config: Optional[MatchingConfig] = None) -> List[Tuple[List[MatchResult], str]]:
    """
    Batch match multiple medical tuples.
    
    Args:
        input_tuples: List of tuples [description, entity] to match
        vocabulary_data: Vocabulary data for matching
        contexts: Optional list of contexts
        config: Optional matching configuration
    
    Returns:
        List of tuples (match_results, entity)
    """
    matcher = MedicalMatcher(vocabulary_data, config)
    return matcher.batch_match_tuples(input_tuples, contexts)

if __name__ == "__main__":
    # Test the medical matcher
    test_queries = [
        "mri pelvis",
        "hcv rna",
        "depression screen",
        "transvaginal ultrasound procedu",
        "coronary angioplasty",
        "metformin 500 mg oral tablet twice daily",
        "chest pain and heart problems"
    ]
    
    # Create matching configuration
    config = MatchingConfig(
        enable_exact_matching=True,
        enable_fuzzy_matching=True,
        enable_semantic_matching=True,
        enable_abbreviation_matching=True,
        enable_token_completion=True,
        enable_terminology_mapping=True,
        enable_context_aware_matching=True,
        fuzzy_threshold=0.7,
        semantic_threshold=0.6,
        abbreviation_threshold=0.7,
        token_completion_threshold=0.7,
        terminology_threshold=0.8
    )
    
    # Note: In real usage, you would load vocabulary data from parquet files
    # matcher = MedicalMatcher(vocabulary_data, config)
    
    print("Medical Matcher created successfully!")
    print("To use with real data, load vocabulary from parquet files:")
    print("matcher.load_vocabulary_data('path/to/vocabulary.parquet')")

