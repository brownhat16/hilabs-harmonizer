"""
Token Completion Module for Medical Text

Handles partial/truncated medical terms by attempting to complete them using:
- Prefix matching against medical vocabulary
- Common medical term patterns
- Context-aware completion
- Confidence scoring for completions
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd

@dataclass
class CompletionCandidate:
    """Represents a completion candidate with confidence score."""
    original: str
    completed: str
    confidence: float
    method: str
    context: Optional[str] = None

class MedicalTokenCompleter:
    """Completes partial/truncated medical terms."""
    
    def __init__(self, vocabulary: Optional[List[str]] = None):
        self.vocabulary = vocabulary or []
        self.vocab_trie = self._build_trie()
        self.medical_patterns = self._initialize_medical_patterns()
        self.common_prefixes = self._initialize_common_prefixes()
        self.common_suffixes = self._initialize_common_suffixes()
    
    def _build_trie(self) -> Dict:
        """Build a trie structure for efficient prefix matching."""
        trie = {}
        for word in self.vocabulary:
            current = trie
            for char in word.lower():
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['$'] = word  # Mark end of word
        return trie
    
    def _initialize_medical_patterns(self) -> Dict[str, List[str]]:
        """Initialize common medical term patterns for completion."""
        return {
            'procedures': [
                'ectomy', 'otomy', 'oscopy', 'plasty', 'centesis', 'graphy',
                'therapy', 'treatment', 'procedure', 'surgery', 'operation'
            ],
            'conditions': [
                'itis', 'osis', 'emia', 'uria', 'pathy', 'syndrome',
                'disorder', 'disease', 'condition', 'abnormality'
            ],
            'anatomy': [
                'muscle', 'bone', 'joint', 'organ', 'tissue', 'vessel',
                'nerve', 'artery', 'vein', 'gland', 'cavity'
            ],
            'medications': [
                'in', 'ol', 'ide', 'ate', 'ine', 'mab', 'nib', 'zole',
                'mycin', 'cillin', 'pam', 'pine', 'sone'
            ],
            'measurements': [
                'gram', 'meter', 'scope', 'graph', 'meter', 'analyzer',
                'monitor', 'detector', 'sensor'
            ]
        }
    
    def _initialize_common_prefixes(self) -> List[str]:
        """Initialize common medical prefixes."""
        return [
            'anti', 'auto', 'bi', 'co', 'de', 'dis', 'ex', 'hyper', 'hypo',
            'inter', 'intra', 'micro', 'macro', 'mono', 'multi', 'neo',
            'non', 'over', 'peri', 'post', 'pre', 'pro', 're', 'sub',
            'super', 'trans', 'ultra', 'under', 'uni'
        ]
    
    def _initialize_common_suffixes(self) -> List[str]:
        """Initialize common medical suffixes."""
        return [
            'al', 'ar', 'ary', 'eal', 'ial', 'ic', 'ical', 'ine', 'ive',
            'oid', 'ose', 'ous', 'tic', 'ular', 'ure', 'y'
        ]
    
    def find_prefix_matches(self, partial: str, max_results: int = 10) -> List[str]:
        """
        Find vocabulary words that start with the given prefix.
        
        Args:
            partial: Partial word to complete
            max_results: Maximum number of results to return
        
        Returns:
            List of matching words
        """
        if not partial or not self.vocab_trie:
            return []
        
        partial_lower = partial.lower()
        current = self.vocab_trie
        
        # Navigate to the prefix node
        for char in partial_lower:
            if char not in current:
                return []
            current = current[char]
        
        # Collect all words with this prefix
        matches = []
        self._collect_words(current, matches, max_results)
        return matches
    
    def _collect_words(self, node: Dict, matches: List[str], max_results: int):
        """Recursively collect words from trie node."""
        if len(matches) >= max_results:
            return
        
        if '$' in node:
            matches.append(node['$'])
        
        for char, child_node in node.items():
            if char != '$':
                self._collect_words(child_node, matches, max_results)
    
    def complete_by_pattern(self, partial: str, context: Optional[str] = None) -> List[CompletionCandidate]:
        """
        Complete partial terms using medical patterns.
        
        Args:
            partial: Partial term to complete
            context: Optional context hint
        
        Returns:
            List of completion candidates
        """
        candidates = []
        partial_lower = partial.lower()
        
        # Try pattern-based completion
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                if partial_lower.endswith(pattern[:len(partial_lower)]):
                    # Found a potential pattern match
                    completed = partial + pattern[len(partial_lower):]
                    confidence = 0.7  # Medium confidence for pattern matching
                    
                    candidates.append(CompletionCandidate(
                        original=partial,
                        completed=completed,
                        confidence=confidence,
                        method=f"pattern_{category}",
                        context=context
                    ))
        
        return candidates
    
    def complete_by_prefix_suffix(self, partial: str) -> List[CompletionCandidate]:
        """
        Complete partial terms using common medical prefixes and suffixes.
        
        Args:
            partial: Partial term to complete
        
        Returns:
            List of completion candidates
        """
        candidates = []
        partial_lower = partial.lower()
        
        # Try prefix completion
        for prefix in self.common_prefixes:
            if partial_lower.startswith(prefix[:len(partial_lower)]):
                completed = prefix + partial[len(partial_lower):]
                confidence = 0.6  # Lower confidence for prefix matching
                
                candidates.append(CompletionCandidate(
                    original=partial,
                    completed=completed,
                    confidence=confidence,
                    method="prefix_completion"
                ))
        
        # Try suffix completion
        for suffix in self.common_suffixes:
            if partial_lower.endswith(suffix[:len(partial_lower)]):
                completed = partial + suffix[len(partial_lower):]
                confidence = 0.6  # Lower confidence for suffix matching
                
                candidates.append(CompletionCandidate(
                    original=partial,
                    completed=completed,
                    confidence=confidence,
                    method="suffix_completion"
                ))
        
        return candidates
    
    def complete_token(self, partial: str, context: Optional[str] = None, 
                      max_candidates: int = 5) -> List[CompletionCandidate]:
        """
        Complete a partial token using multiple strategies.
        
        Args:
            partial: Partial token to complete
            context: Optional context for completion
            max_candidates: Maximum number of candidates to return
        
        Returns:
            List of completion candidates sorted by confidence
        """
        if not partial or len(partial) < 2:
            return []
        
        all_candidates = []
        
        # Strategy 1: Prefix matching against vocabulary
        if self.vocabulary:
            prefix_matches = self.find_prefix_matches(partial, max_candidates)
            for match in prefix_matches:
                confidence = 0.9  # High confidence for exact vocabulary match
                all_candidates.append(CompletionCandidate(
                    original=partial,
                    completed=match,
                    confidence=confidence,
                    method="vocabulary_match",
                    context=context
                ))
        
        # Strategy 2: Pattern-based completion
        pattern_candidates = self.complete_by_pattern(partial, context)
        all_candidates.extend(pattern_candidates)
        
        # Strategy 3: Prefix/suffix completion
        prefix_suffix_candidates = self.complete_by_prefix_suffix(partial)
        all_candidates.extend(prefix_suffix_candidates)
        
        # Sort by confidence and return top candidates
        all_candidates.sort(key=lambda x: x.confidence, reverse=True)
        return all_candidates[:max_candidates]
    
    def complete_text_tokens(self, text: str, context: Optional[str] = None,
                           min_length: int = 3, max_length: int = 15) -> str:
        """
        Complete all partial tokens in text.
        
        Args:
            text: Input text with potential partial tokens
            context: Optional context for completion
            min_length: Minimum length to consider for completion
            max_length: Maximum length to consider for completion
        
        Returns:
            Text with completed tokens
        """
        if not text:
            return text
        
        # Find potential partial tokens (words that might be incomplete)
        words = text.split()
        completed_words = []
        
        for word in words:
            # Clean word of punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check if word might be incomplete
            if (min_length <= len(clean_word) <= max_length and 
                not clean_word.endswith(('ing', 'ed', 'er', 'ly', 'tion', 'sion'))):
                
                # Try to complete the word
                candidates = self.complete_token(clean_word, context, max_candidates=1)
                
                if candidates and candidates[0].confidence > 0.7:
                    # Use the best completion
                    completed_word = word.replace(clean_word, candidates[0].completed)
                    completed_words.append(completed_word)
                else:
                    completed_words.append(word)
            else:
                completed_words.append(word)
        
        return ' '.join(completed_words)
    
    def analyze_completion_opportunities(self, text: str) -> Dict[str, any]:
        """
        Analyze text for completion opportunities.
        
        Args:
            text: Input text
        
        Returns:
            Analysis results with completion suggestions
        """
        if not text:
            return {'original': text, 'completions': [], 'completion_count': 0}
        
        words = text.split()
        completions = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            
            if 3 <= len(clean_word) <= 15:
                candidates = self.complete_token(clean_word, max_candidates=3)
                
                if candidates:
                    completions.append({
                        'original': word,
                        'clean': clean_word,
                        'candidates': [
                            {
                                'completed': c.completed,
                                'confidence': c.confidence,
                                'method': c.method
                            }
                            for c in candidates
                        ]
                    })
        
        return {
            'original': text,
            'completions': completions,
            'completion_count': len(completions)
        }
    
    def load_vocabulary_from_parquet(self, parquet_path: str, column: str = 'STR'):
        """Load vocabulary from parquet file."""
        try:
            df = pd.read_parquet(parquet_path, columns=[column])
            self.vocabulary = df[column].dropna().astype(str).tolist()
            self.vocab_trie = self._build_trie()
        except Exception as e:
            print(f"Error loading vocabulary from {parquet_path}: {e}")
    
    def add_custom_vocabulary(self, words: List[str]):
        """Add custom words to vocabulary."""
        self.vocabulary.extend(words)
        self.vocab_trie = self._build_trie()

# Convenience functions
def complete_medical_tokens(text: str, vocabulary: Optional[List[str]] = None) -> str:
    """
    Complete partial medical tokens in text.
    
    Args:
        text: Input text with potential partial tokens
        vocabulary: Optional vocabulary for completion
    
    Returns:
        Text with completed tokens
    """
    completer = MedicalTokenCompleter(vocabulary)
    return completer.complete_text_tokens(text)

def analyze_token_completion(text: str, vocabulary: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Analyze text for token completion opportunities.
    
    Args:
        text: Input text
        vocabulary: Optional vocabulary for completion
    
    Returns:
        Analysis results
    """
    completer = MedicalTokenCompleter(vocabulary)
    return completer.analyze_completion_opportunities(text)

if __name__ == "__main__":
    # Test the token completer
    test_texts = [
        "transvaginal ultrasound procedu",
        "coronary angioplasty",
        "magnetic resonance imag",
        "electrocardiog",
        "computed tomog",
        "ultrasound of fetal anat"
    ]
    
    # Sample medical vocabulary
    sample_vocab = [
        "transvaginal", "ultrasound", "procedure", "coronary", "angioplasty",
        "magnetic", "resonance", "imaging", "electrocardiogram", "computed",
        "tomography", "fetal", "anatomy", "percutaneous", "transluminal"
    ]
    
    completer = MedicalTokenCompleter(sample_vocab)
    
    for text in test_texts:
        print(f"Original: {text}")
        print(f"Completed: {completer.complete_text_tokens(text)}")
        analysis = completer.analyze_completion_opportunities(text)
        print(f"Analysis: {analysis['completion_count']} completion opportunities")
        print("---")

