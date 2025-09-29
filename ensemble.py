from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Retrieval.matcher import RetrievalEngine, Candidate as CandidateV1, MatchResult
from Retrieval2 import load_engine_v2
from Retrieval2.engine import RetrievalEngineV2, RetrievalResultV2, RetrievalCandidate

import main as main_v1


MED_PRIORITY = {
    "SCD": 1,
    "SBD": 2,
    "GPCK": 3,
    "BPCK": 4,
    "PIN": 5,
    "IN": 6,
    "BN": 7,
    "SY": 8,
    "TMSY": 8,
    "PSN": 9,
    "DF": 10,
}

SNOMED_PRIORITY = {
    "PT": 1,
    "FN": 2,
    "SY": 3,
}


@lru_cache(maxsize=None)
def _get_engine_v1_cached(root: str, rx_mtime: float, sn_mtime: float) -> RetrievalEngine:
    base_path = Path(root)
    rx_path = base_path / "Data" / "rxnorm_all_data.parquet"
    sn_path = base_path / "Data" / "snomed_all_data.parquet"
    return RetrievalEngine(str(rx_path), str(sn_path))


@lru_cache(maxsize=None)
def _get_engine_v2_cached(root: str, rx_mtime: float, sn_mtime: float) -> RetrievalEngineV2:
    base_path = Path(root)
    return load_engine_v2(base_path)


def _priority(entity: str, tty: str) -> int:
    if entity == "medication":
        return MED_PRIORITY.get(tty.upper(), 99)
    return SNOMED_PRIORITY.get(tty.upper(), 99)


def _serialize_v1(cand: Optional[CandidateV1]) -> Dict[str, Any]:
    if cand is None:
        return {}
    return {
        "system": cand.system,
        "code": cand.code,
        "term": cand.string,
        "tty": cand.tty,
        "cui": cand.cui,
        "score": round(cand.score, 4),
    }


def _serialize_v2(cand: Optional[RetrievalCandidate]) -> Dict[str, Any]:
    if cand is None:
        return {}
    return {
        "system": cand.system,
        "code": cand.code,
        "term": cand.term,
        "tty": cand.tty,
        "cui": cand.cui,
        "score": round(cand.combined, 4),
        "lexical": round(cand.lexical, 4),
        "semantic": round(cand.semantic, 4),
    }


class EnsembleRetriever:
    def __init__(self, base_path: Path) -> None:
        cache_key = main_v1._base_key(base_path)
        self.engine_v1 = _get_engine_v1_cached(*cache_key)
        self.engine_v2 = _get_engine_v2_cached(*cache_key)
        self.cache_key = cache_key

    def harmonize(self, query: str, entity: str) -> Dict[str, Any]:
        preprocessor = main_v1._get_preprocessor(*self.cache_key)
        if hasattr(preprocessor, "processing_history"):
            preprocessor.processing_history.clear()

        preproc_result = preprocessor.preprocess_comprehensive(query, entity)
        retrieval_input = main_v1._preproc_to_retrieval_input(preproc_result)

        result_v1: MatchResult = self.engine_v1.map_one(retrieval_input)
        result_v2: RetrievalResultV2 = self.engine_v2.map_one(retrieval_input)

        merged = self._blend_results(entity, retrieval_input, result_v1, result_v2)

        return {
            "input": {"query": query, "entity": entity},
            "preprocessed": retrieval_input,
            "ensemble": merged,
            "engine_v1": {
                "action": result_v1.action,
                "chosen": _serialize_v1(result_v1.chosen),
                "top_candidates": [_serialize_v1(c) for c in result_v1.top_candidates],
            },
            "engine_v2": {
                "action": result_v2.action,
                "chosen": _serialize_v2(result_v2.chosen),
                "candidates": [_serialize_v2(c) for c in result_v2.candidates],
            },
        }

    def _blend_results(
        self,
        entity: str,
        retrieval_input: Dict[str, Any],
        result_v1: MatchResult,
        result_v2: RetrievalResultV2,
    ) -> Dict[str, Any]:
        parsed = retrieval_input.get("parsed") or {}
        prefer_v2 = entity == "medication" and parsed.get("dose_value") is not None
        v1_score = result_v1.chosen.score if result_v1.chosen else 0.0
        v2_score = result_v2.chosen.combined if result_v2.chosen else 0.0

        if prefer_v2 and result_v2.chosen:
            return {
                "strategy": "prefer_v2",
                "chosen": _serialize_v2(result_v2.chosen),
                "score": round(v2_score, 4),
            }

        if result_v1.action == "NO_MATCH" and result_v2.chosen:
            return {
                "strategy": "fallback_v2",
                "chosen": _serialize_v2(result_v2.chosen),
                "score": round(v2_score, 4),
            }

        if result_v2.action == "NO_MATCH" and result_v1.chosen:
            return {
                "strategy": "fallback_v1",
                "chosen": _serialize_v1(result_v1.chosen),
                "score": round(v1_score, 4),
            }

        combined = self._build_ensemble_candidates(entity, result_v1, result_v2)
        if not combined:
            return {
                "strategy": "empty",
                "chosen": {},
                "score": 0.0,
            }

        final = combined[0]
        return {
            "strategy": "blend",
            "chosen": final,
            "score": round(final.get("ensemble_score", 0.0), 4),
            "candidates": combined,
        }

    def _build_ensemble_candidates(
        self,
        entity: str,
        result_v1: MatchResult,
        result_v2: RetrievalResultV2,
    ) -> List[Dict[str, Any]]:
        scores: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def add_v1(c: CandidateV1) -> None:
            key = (c.system, c.cui or c.code)
            entry = scores.setdefault(
                key,
                {
                    "system": c.system,
                    "code": c.code,
                    "cui": c.cui or c.code,
                    "term": c.string,
                    "tty": c.tty,
                    "v1_score": 0.0,
                    "v2_score": 0.0,
                },
            )
            entry["v1_score"] = max(entry["v1_score"], c.score)
            if _priority(entity, c.tty) < _priority(entity, entry.get("tty", c.tty)):
                entry["term"] = c.string
                entry["code"] = c.code
                entry["tty"] = c.tty

        def add_v2(c: RetrievalCandidate) -> None:
            key = (c.system, c.cui)
            entry = scores.setdefault(
                key,
                {
                    "system": c.system,
                    "code": c.code,
                    "cui": c.cui,
                    "term": c.term,
                    "tty": c.tty,
                    "v1_score": 0.0,
                    "v2_score": 0.0,
                },
            )
            entry["v2_score"] = max(entry["v2_score"], c.combined)
            if _priority(entity, c.tty) < _priority(entity, entry.get("tty", c.tty)):
                entry["term"] = c.term
                entry["code"] = c.code
                entry["tty"] = c.tty

        for cand in result_v1.top_candidates:
            add_v1(cand)
        for cand in result_v2.candidates:
            add_v2(cand)

        combined: List[Dict[str, Any]] = []
        for entry in scores.values():
            v1 = entry.get("v1_score", 0.0)
            v2 = entry.get("v2_score", 0.0)
            ensemble_score = (0.6 * v1) + (0.4 * v2)
            entry["ensemble_score"] = ensemble_score
            combined.append(entry)

        combined.sort(
            key=lambda e: (
                -e.get("ensemble_score", 0.0),
                _priority(entity, e.get("tty", "")),
                -max(e.get("v1_score", 0.0), e.get("v2_score", 0.0)),
                len(e.get("term", "")),
            )
        )

        return combined


def harmonize_ensemble(query: str, entity: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    if base_path is None:
        base_path = Path(main_v1.BASE_PATH)
    retriever = EnsembleRetriever(base_path)
    return retriever.harmonize(query, entity)


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print("Usage: python ensemble.py \"query text\" <entity>")
        demo = harmonize_ensemble("Paracetamol 500 mg", "medication")
        print(json.dumps(demo, indent=2))
        return 0

    query = argv[1]
    entity = argv[2]
    result = harmonize_ensemble(query, entity)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))

