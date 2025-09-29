#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

# Ensure DataProcessing and Retrieval modules are importable
BASE_PATH = Path(__file__).resolve().parent
DP_PATH = BASE_PATH / "DataProcessing"
if str(DP_PATH) not in sys.path:
    sys.path.insert(0, str(DP_PATH))

# Imports
from DataProcessing.enhanced_preprocessor import (
    EnhancedMedicalPreprocessor,
    ComprehensivePreprocessingResult,
)
from Retrieval.matcher import RetrievalEngine


def _preproc_to_retrieval_input(result: ComprehensivePreprocessingResult) -> Dict[str, Any]:
    parsed = result.parsed or None
    return {
        "normalized": result.normalized or "",
        "tokens_no_stop": result.tokens_no_stop or [],
        "parsed": {
            "form": (parsed.form if parsed else None),
            "components": len(result.components or []),
            "dose_value": (parsed.dose_value if parsed else None),
            "dose_unit": (parsed.dose_unit if parsed else None),
        },
        "entity": (result.entity or "").lower(),
        "is_combination": bool(result.is_combination),
        "abbrev_expansions": result.abbrev_expansions or {},
        "preprocess_confidence": float(result.preprocess_confidence or 0.0),
        "entity_confidence": float(result.entity_confidence or 0.0),
    }


def _standardize_output(system: Optional[str], code: Optional[str], description: Optional[str]) -> Dict[str, Any]:
    return {
        "Standard Code": code,
        "Standard System": system,
        "Standard Description": description,
    }


def _base_key(path: Path) -> tuple[str, float, float]:
    resolved = path.resolve()
    rx_path = resolved / "Data" / "rxnorm_all_data.parquet"
    sn_path = resolved / "Data" / "snomed_all_data.parquet"
    try:
        rx_mtime = rx_path.stat().st_mtime
    except OSError:
        rx_mtime = 0.0
    try:
        sn_mtime = sn_path.stat().st_mtime
    except OSError:
        sn_mtime = 0.0
    return (str(resolved), rx_mtime, sn_mtime)


@lru_cache(maxsize=None)
def _get_preprocessor(base_path_str: str, rx_mtime: float, sn_mtime: float) -> EnhancedMedicalPreprocessor:
    base_path = Path(base_path_str)
    rx_path = base_path / "Data" / "rxnorm_all_data.parquet"
    sn_path = base_path / "Data" / "snomed_all_data.parquet"
    preprocessor = EnhancedMedicalPreprocessor()

    try:
        import pandas as pd

        rx_df = pd.read_parquet(rx_path)
        sn_df = pd.read_parquet(sn_path)

        try:
            preprocessor.augment_abbrev_map_from_vocabulary(rx_df)
        except Exception:
            pass

        vocab: set[str] = set()

        def add_terms(series):
            try:
                for s in series.astype(str).str.lower().tolist():
                    s = s.strip()
                    if not s:
                        continue
                    vocab.add(s)
                    for tok in s.replace("/", " ").replace(",", " ").split():
                        tok = tok.strip()
                        if tok.isalpha() and len(tok) >= 3:
                            vocab.add(tok)
            except Exception:
                pass

        try:
            rx_ttys = {"IN", "PIN", "MIN", "BN", "SCD", "SBD"}
            rx_terms = rx_df.loc[rx_df["TTY"].isin(rx_ttys), "STR"]
            add_terms(rx_terms)
        except Exception:
            pass

        try:
            if "TTY" in sn_df.columns:
                sn_terms = sn_df.loc[sn_df["TTY"].astype(str).str.upper() == "PT", "STR"]
            else:
                sn_terms = sn_df["STR"]
            add_terms(sn_terms)
        except Exception:
            pass

        if vocab:
            preprocessor.ingredient_vocabulary = vocab
    except Exception:
        pass

    return preprocessor


@lru_cache(maxsize=None)
def _get_engine(base_path_str: str, rx_mtime: float, sn_mtime: float) -> RetrievalEngine:
    base_path = Path(base_path_str)
    rx_path = base_path / "Data" / "rxnorm_all_data.parquet"
    sn_path = base_path / "Data" / "snomed_all_data.parquet"
    return RetrievalEngine(str(rx_path), str(sn_path))

def reset_cache() -> None:
    """Invalidate cached preprocessor and retrieval engine instances."""
    _get_preprocessor.cache_clear()
    _get_engine.cache_clear()


def harmonize(query: str, entity: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    if base_path is None:
        base_path = BASE_PATH

    cache_key = _base_key(base_path)
    preprocessor = _get_preprocessor(*cache_key)
    # Avoid unbounded growth of processing history when reusing the instance
    if hasattr(preprocessor, "processing_history"):
        preprocessor.processing_history.clear()

    preproc = preprocessor.preprocess_comprehensive(query, entity)

    retrieval_input = _preproc_to_retrieval_input(preproc)

    engine = _get_engine(*cache_key)
    result = engine.map_one(retrieval_input)

    # Fallback disabled: do not use ingredient vocabulary-based fuzzy correction

    # 4) Standardize output (prefer chosen if available, else best candidate)
    chosen = result.chosen if result.chosen else (result.top_candidates[0] if result.top_candidates else None)
    standardized = _standardize_output(
        chosen.system if chosen else None,
        chosen.code if chosen else None,
        chosen.string if chosen else None,
    )

    return {
        "input": {"query": query, "entity": entity},
        "preprocessed": retrieval_input,
        "action": result.action,
        "standardized": standardized,
        "top_candidates": [
            _standardize_output(c.system, c.code, c.string) for c in result.top_candidates
        ],
    }


def main(argv: list[str]) -> int:
    # Usage: python main.py "query text" medication
    if len(argv) < 3:
        print("Usage: python main.py \"query text\" <entity>")
        print("Example: python main.py \"aspirin 81 mg oral tablet\" medication")
        demo = harmonize("paracetaxol", "medication")
        print(json.dumps(demo["standardized"], indent=2))
        return 0

    query = argv[1]
    entity = argv[2]
    output = harmonize(query, entity)
    print(json.dumps(output["standardized"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
