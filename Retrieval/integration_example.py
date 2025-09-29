#!/usr/bin/env python3
"""
Simple integration example showing how to use Retrieval module with preprocessed data.

This script demonstrates the complete workflow:
1. Take preprocessed data from DataProcessing module
2. Convert it to RetrievalEngine format
3. Match against terminology databases
4. Return structured results
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import from the Retrieval module
from .matcher import RetrievalEngine, MatchResult


class MedicalDataMatcher:
    """
    A wrapper class that integrates DataProcessing output with Retrieval module.
    
    This class handles the conversion between DataProcessing format and RetrievalEngine format,
    and provides a simple interface for matching preprocessed medical data.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the matcher with terminology databases.
        
        Args:
            base_path: Path to the project root. If None, uses current file's parent.
        """
        if base_path is None:
            base_path = Path(__file__).resolve().parents[1]
        
        self.base_path = base_path
        self.engine = self._load_engine()
    
    def _load_engine(self) -> RetrievalEngine:
        """Load the RetrievalEngine with RxNorm and SNOMED data."""
        rx_path = self.base_path / "Data" / "rxnorm_all_data.parquet"
        sn_path = self.base_path / "Data" / "snomed_all_data.parquet"
        return RetrievalEngine(str(rx_path), str(sn_path))
    
    def _convert_format(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert DataProcessing output format to RetrievalEngine input format.
        
        Maps field names and structures between the two formats.
        """
        normalized_text = preprocessed_data.get("Normalized", "") or ""
        tokens_no_stop = preprocessed_data.get("Tokens (no stop)", []) or []

        # Try to extract dose info from normalized text first, then from tokens
        dose_value, dose_unit = self._extract_dose_from_text(normalized_text)
        if dose_value is None and tokens_no_stop:
            dose_value, dose_unit = self._extract_dose_from_text(" ".join(map(str, tokens_no_stop)))

        parsed: Dict[str, Any] = {
            "form": preprocessed_data.get("Form", ""),
            "components": preprocessed_data.get("Components", 0),
            "ingredient": preprocessed_data.get("Ingredient"),
        }
        if dose_value is not None:
            parsed["dose_value"] = dose_value
        if dose_unit is not None:
            parsed["dose_unit"] = dose_unit

        return {
            "normalized": normalized_text,
            "tokens_no_stop": tokens_no_stop,
            "parsed": parsed,
            "entity": preprocessed_data.get("Entity", "").lower(),
            "is_combination": preprocessed_data.get("Is combination", False),
            "abbrev_expansions": preprocessed_data.get("Abbrev expansions", {}),
            "preprocess_confidence": preprocessed_data.get("Preprocess confidence", 0.0),
            "entity_confidence": preprocessed_data.get("Entity confidence", 0.0),
        }

    def _extract_dose_from_text(self, text: str) -> tuple[Optional[float], Optional[str]]:
        """Extract dose value and unit from a text string.

        Supports common units and patterns like "81 mg", "0.5 mg", "5mcg", "10 mL", "400 IU".
        Returns (value, unit) or (None, None) if not found.
        """
        if not text:
            return None, None

        lowered = str(text).lower()

        # Define unit variants
        unit_patterns = [
            r"mg", r"mcg", r"µg", r"ug", r"g", r"kg",
            r"ml", r"m\s*l", r"l",
            r"iu", r"units", r"unit"
        ]
        units_regex = "|".join(unit_patterns)

        # Patterns to match number + unit with optional space
        patterns = [
            rf"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>{units_regex})\b",
            rf"\b(?P<unit>{units_regex})\s*(?P<val>\d+(?:\.\d+)?)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match:
                try:
                    value = float(match.group("val"))
                except Exception:
                    value = None
                unit = match.group("unit").replace(" ", "") if match.group("unit") else None
                # Normalize microgram variants to mcg
                if unit in {"µg", "ug"}:
                    unit = "mcg"
                return value, unit

        return None, None
    
    def match(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match preprocessed medical data against terminology databases.
        
        Args:
            preprocessed_data: Output from DataProcessing module in the format:
                {
                    "Normalized": "1/2 tablet",
                    "Tokens": ['1/2', 'tablet'],
                    "Tokens (no stop)": ['1/2', 'tablet'],
                    "Is combination": True,
                    "Entity": "medication",
                    "Entity confidence": 1.00,
                    "Components": 2,
                    "Form": "tablet",
                    "Preprocess confidence": 0.95
                }
        
        Returns:
            Dictionary containing match results with the following structure:
            {
                "success": bool,
                "action": str,  # AUTO_ACCEPT, REVIEW, FALLBACK_TO_INGREDIENT, NO_MATCH
                "chosen_match": Optional[Dict],  # Best match if any
                "top_candidates": List[Dict],  # Top 5 candidates
                "confidence": float,  # Overall confidence score
                "recommendations": List[str]  # Action recommendations
            }
        """
        try:
            # Convert format
            retrieval_input = self._convert_format(preprocessed_data)
            
            # Perform matching
            result = self.engine.map_one(retrieval_input)
            
            # Format response
            return self._format_response(preprocessed_data, result)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": "ERROR",
                "chosen_match": None,
                "top_candidates": [],
                "confidence": 0.0,
                "recommendations": ["Check input format and try again"]
            }
    
    def _format_response(self, original_data: Dict[str, Any], result: MatchResult) -> Dict[str, Any]:
        """Format the RetrievalEngine result into a user-friendly response."""
        
        # Extract chosen match
        chosen_match = None
        if result.chosen:
            chosen_match = {
                "system": result.chosen.system,
                "code": result.chosen.code,
                "string": result.chosen.string,
                "tty": result.chosen.tty,
                "cui": result.chosen.cui,
                "score": round(result.chosen.score, 4),
                "confidence": self._calculate_confidence(result.chosen, original_data),
                "reasons": result.chosen.reason_flags,
            }
        
        # Extract top candidates
        top_candidates = []
        for candidate in result.top_candidates:
            top_candidates.append({
                "system": candidate.system,
                "code": candidate.code,
                "string": candidate.string,
                "tty": candidate.tty,
                "cui": candidate.cui,
                "score": round(candidate.score, 4),
                "confidence": self._calculate_confidence(candidate, original_data),
                "reasons": candidate.reason_flags,
            })

        # Standardized outputs
        standardized_output = [
            self._standardize_candidate(original_data.get("Entity", ""), cand)
            for cand in result.top_candidates
        ]
        standardized_chosen = (
            self._standardize_candidate(original_data.get("Entity", ""), result.chosen)
            if result.chosen else None
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(result, original_data)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(result, original_data)
        
        return {
            "success": True,
            "action": result.action,
            "chosen_match": chosen_match,
            "top_candidates": top_candidates,
            "confidence": overall_confidence,
            "recommendations": recommendations,
            "original_input": original_data,
            "standardized_output": standardized_output,
            "standardized_chosen": standardized_chosen,
        }

    def _standardize_candidate(self, entity: str, candidate) -> Dict[str, Any]:
        """Return a standardized representation for downstream systems.

        Fields:
          - Type of Standard: input entity (e.g., medication, diagnosis)
          - Code Standard: terminology code (e.g., RxCUI, SNOMED code)
          - System Standard: terminology system (RXNORM or SNOMED)
          - Description: human-readable term/string
        """
        if candidate is None:
            return {
                "Type of Standard": (entity or "").lower(),
                "Code Standard": None,
                "System Standard": None,
                "Description": None,
            }
        return {
            "Type of Standard": (entity or "").lower(),
            "Code Standard": candidate.code,
            "System Standard": candidate.system,
            "Description": candidate.string,
        }
    
    def _calculate_confidence(self, candidate, original_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a candidate match."""
        base_score = candidate.score
        
        # Boost confidence if entity types match
        entity_confidence = original_data.get("Entity confidence", 0.0)
        preprocess_confidence = original_data.get("Preprocess confidence", 0.0)
        
        # Weighted combination
        confidence = (
            0.6 * base_score +
            0.2 * entity_confidence +
            0.2 * preprocess_confidence
        )
        
        return min(1.0, confidence)
    
    def _calculate_overall_confidence(self, result: MatchResult, original_data: Dict[str, Any]) -> float:
        """Calculate overall confidence for the matching result."""
        if not result.top_candidates:
            return 0.0
        
        best_candidate = result.top_candidates[0]
        return self._calculate_confidence(best_candidate, original_data)
    
    def _generate_recommendations(self, result: MatchResult, original_data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on the match result."""
        recommendations = []
        
        if result.action == "AUTO_ACCEPT":
            recommendations.append("High confidence match - can be automatically accepted")
        elif result.action == "REVIEW":
            recommendations.append("Manual review recommended - check match quality")
            if original_data.get("Is combination"):
                recommendations.append("Combination medication detected - verify all components")
        elif result.action == "FALLBACK_TO_INGREDIENT":
            recommendations.append("Consider matching to ingredient level instead of specific formulation")
        elif result.action == "NO_MATCH":
            recommendations.append("No suitable match found - may need manual mapping or additional preprocessing")
        
        # Add specific recommendations based on entity type
        entity = original_data.get("Entity", "").lower()
        if entity == "medication":
            if not original_data.get("Form"):
                recommendations.append("Consider adding form information for better medication matching")
        elif entity == "diagnosis":
            recommendations.append("Verify diagnosis code matches clinical context")
        
        return recommendations


def main():
    """Example usage of the MedicalDataMatcher."""
    
    # Initialize matcher
    matcher = MedicalDataMatcher()
    
    # Example preprocessed data (your format)
    test_cases = [
        {
            "name": "Half tablet medication",
            "data": {
                "Normalized": "1/2 tablet",
                "Tokens": ['1/2', 'tablet'],
                "Tokens (no stop)": ['1/2', 'tablet'],
                "Is combination": True,
                "Entity": "medication",
                "Entity confidence": 1.00,
                "Components": 2,
                "Form": "tablet",
                "Preprocess confidence": 0.95
            }
        },
        {
            "name": "Aspirin with dose",
            "data": {
                "Normalized": "aspirin 81 mg oral tablet",
                "Tokens": ['aspirin', '81', 'mg', 'oral', 'tablet'],
                "Tokens (no stop)": ['aspirin', '81', 'mg', 'oral', 'tablet'],
                "Is combination": False,
                "Entity": "medication",
                "Entity confidence": 1.00,
                "Components": 1,
                "Form": "tablet",
                "Preprocess confidence": 0.98
            }
        }
    ]
    
    # Test each case
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST CASE: {test_case['name']}")
        print('='*60)
        
        # Perform matching
        result = matcher.match(test_case['data'])
        
        # Display results
        print(f"Success: {result['success']}")
        print(f"Action: {result['action']}")
        print(f"Overall Confidence: {result['confidence']:.3f}")
        
        if result['chosen_match']:
            chosen = result['chosen_match']
            print(f"Chosen Match: {chosen['system']}:{chosen['code']} - {chosen['string']}")
            print(f"Match Score: {chosen['score']:.3f}")
            print(f"Match Confidence: {chosen['confidence']:.3f}")
        
        print(f"Top Candidates: {len(result['top_candidates'])}")
        for i, candidate in enumerate(result['top_candidates'][:3], 1):
            print(f"  {i}. {candidate['system']}:{candidate['code']} - {candidate['string']} (score: {candidate['score']:.3f})")
        
        print("Recommendations:")
        for rec in result['recommendations']:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()
