#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

from Retrieval2 import load_engine_v2
from Retrieval2.engine import RetrievalResultV2

import main as main_v1


BASE_PATH = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def _get_engine_v2_cached(root: str, rx_mtime: float, sn_mtime: float):
    return load_engine_v2(Path(root))


def harmonize_v2(query: str, entity: str, base_path: Path | None = None) -> Dict[str, Any]:
    if base_path is None:
        base_path = BASE_PATH

    cache_key = main_v1._base_key(base_path)
    preprocessor = main_v1._get_preprocessor(*cache_key)
    if hasattr(preprocessor, "processing_history"):
        preprocessor.processing_history.clear()

    preproc_result = preprocessor.preprocess_comprehensive(query, entity)
    retrieval_input = main_v1._preproc_to_retrieval_input(preproc_result)

    engine_v2 = _get_engine_v2_cached(*cache_key)
    result: RetrievalResultV2 = engine_v2.map_one(retrieval_input)

    def serialize_candidate(cand: RetrievalResultV2 | Any) -> Dict[str, Any]:
        if cand is None:
            return {}
        return {
            "system": cand.system,
            "code": cand.code,
            "cui": cand.cui,
            "term": cand.term,
            "tty": cand.tty,
            "combined": round(cand.combined, 4),
            "lexical": round(cand.lexical, 4),
            "semantic": round(cand.semantic, 4),
        }

    return {
        "input": {"query": query, "entity": entity},
        "preprocessed": retrieval_input,
        "action": result.action,
        "chosen": serialize_candidate(result.chosen),
        "candidates": [serialize_candidate(c) for c in result.candidates],
    }


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: python main2.py \"query text\" <entity>")
        print("Example: python main2.py \"aspirin 81 mg oral tablet\" medication")
        demo = harmonize_v2("Paracetamol 500 mg", "medication")
        print(json.dumps(demo, indent=2))
        return 0

    query = argv[1]
    entity = argv[2]
    output = harmonize_v2(query, entity)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

