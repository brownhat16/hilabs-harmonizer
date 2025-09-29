from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import importlib
import os

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process as rf_process


LEXICAL_WEIGHT = 0.75
SEMANTIC_WEIGHT = 0.25


MED_TTY_PRIORITY: Dict[str, int] = {
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


SNOMED_TTY_PRIORITY: Dict[str, int] = {
    "PT": 1,
    "FN": 2,
    "SY": 3,
}

DIAG_HEURISTICS: Dict[str, Tuple[str, str, str]] = {
    "right sided abdominal pain": ("285388000", "Right sided abdominal pain", "PT"),
    "right upper quadrant pain": ("285388000", "Right sided abdominal pain", "PT"),
    "upper right abdominal pain": ("285388000", "Right sided abdominal pain", "PT"),
}

PROC_HEURISTICS: Dict[str, Tuple[str, str, str]] = {
    "appendix removal": ("80146002", "Appendectomy", "PT"),
    "appendectomy": ("80146002", "Appendectomy", "PT"),
    "appendix surgery": ("80146002", "Appendectomy", "PT"),
    "chest x-ray": ("399208008", "Plain X-ray of chest", "PT"),
    "chest xray": ("399208008", "Plain X-ray of chest", "PT"),
}

SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_semantic_model() -> Optional[Any]:
    if os.environ.get("DISABLE_SEMANTIC_MODEL") == "1":
        return None
    try:
        module = importlib.import_module("sentence_transformers")
        SentenceTransformer = getattr(module, "SentenceTransformer")
    except Exception:
        return None
    try:
        return SentenceTransformer(SEMANTIC_MODEL_NAME)
    except Exception:
        return None


@dataclass
class RetrievalCandidate:
    system: str
    code: str
    cui: str
    term: str
    tty: str
    lexical: float
    semantic: float
    combined: float


@dataclass
class RetrievalResultV2:
    action: str
    chosen: Optional[RetrievalCandidate]
    candidates: List[RetrievalCandidate]


class KnowledgeBase:
    def __init__(
        self,
        name: str,
        frame: pd.DataFrame,
        priority_map: Dict[str, int],
        concept_column: str = "CUI",
    ) -> None:
        self.name = name
        self.df = frame.reset_index(drop=True)
        self.priority_map = priority_map
        self.concept_column = concept_column

        self._strings: List[str] = self.df["STR"].astype(str).tolist()
        self._cui_index: Dict[str, List[int]] = {}
        self._code_index: Dict[str, int] = {}
        self._build_cui_index()

    def _build_cui_index(self) -> None:
        for idx, row in self.df.iterrows():
            cui = self._cui_from_row(row)
            self._cui_index.setdefault(cui, []).append(idx)
            code = str(row.get("CODE", ""))
            if code and code not in self._code_index:
                self._code_index[code] = idx

    def _cui_from_row(self, row: pd.Series) -> str:
        if self.concept_column in row and pd.notna(row[self.concept_column]):
            return str(row[self.concept_column])
        return str(row.get("CODE", ""))

    def lookup_exact(self, norm: str) -> List[pd.Series]:
        mask = self.df["STR"].astype(str).str.lower() == norm
        return [self.df.iloc[i] for i in mask[mask].index]

    def best_term_for_cui(self, cui: str) -> Optional[pd.Series]:
        rows = self._cui_index.get(cui)
        if not rows:
            return None
        def priority(row: pd.Series) -> int:
            tty = str(row.get("TTY", "")).upper()
            return self.priority_map.get(tty, 99)

        best_idx = min(rows, key=lambda idx: (priority(self.df.iloc[idx]), len(str(self.df.iloc[idx]["STR"]))))
        return self.df.iloc[best_idx]

    def iter_top_matches(self, query: str, limit: int) -> List[tuple[str, float, int]]:
        matches = rf_process.extract(
            query,
            self._strings,
            limit=limit,
            scorer=fuzz.WRatio,
        )
        out: List[tuple[str, float, int]] = []
        for term, score, idx in matches:
            lexical = max(0.0, min(score / 100.0, 1.0))
            out.append((term, lexical, idx))
        return out

    def row_at(self, idx: int) -> pd.Series:
        return self.df.iloc[idx]

    def row_by_code(self, code: str) -> Optional[pd.Series]:
        idx = self._code_index.get(str(code))
        if idx is None:
            return None
        return self.df.iloc[idx]


class RetrievalEngineV2:
    def __init__(
        self,
        rxnorm_path: str,
        snomed_path: str,
        *,
        top_k: int = 25,
        semantic_model: Optional[object] = None,
    ) -> None:
        rx_df = pd.read_parquet(rxnorm_path)
        sn_df = pd.read_parquet(snomed_path)

        self.kb_med = KnowledgeBase("RXNORM", rx_df, MED_TTY_PRIORITY)
        self.kb_snomed = KnowledgeBase("SNOMED", sn_df, SNOMED_TTY_PRIORITY)
        self.top_k = top_k
        self.semantic_model = semantic_model or _load_semantic_model()

    def _select_kb(self, entity: str) -> KnowledgeBase:
        if entity.lower() == "medication":
            return self.kb_med
        return self.kb_snomed

    def _semantic_similarity(self, query: str, candidate: str) -> Optional[float]:
        if self.semantic_model is None:
            score = fuzz.token_set_ratio(query, candidate) / 100.0
            return max(0.0, min(score, 1.0))
        try:
            q_vec = self.semantic_model.encode(query, normalize_embeddings=True)
            c_vec = self.semantic_model.encode(candidate, normalize_embeddings=True)
            if q_vec is None or c_vec is None:
                return None
            return float(np.dot(q_vec, c_vec))
        except Exception:
            return None

    def _combine(self, lexical: float, semantic: Optional[float]) -> float:
        if semantic is None:
            return lexical
        return (LEXICAL_WEIGHT * lexical) + (SEMANTIC_WEIGHT * semantic)

    def _priority(self, kb: KnowledgeBase, tty: str) -> int:
        return kb.priority_map.get(str(tty).upper(), 99)

    def map_one(self, preproc: Dict[str, object]) -> RetrievalResultV2:
        entity = str(preproc.get("entity", "")).lower()
        normalized = str(preproc.get("normalized", ""))
        kb = self._select_kb(entity)

        tokens = [tok for tok in (preproc.get("tokens_no_stop") or []) if isinstance(tok, str)]
        query_string = normalized if normalized else " ".join(tokens)
        if not query_string.strip():
            return RetrievalResultV2("NO_INPUT", None, [])

        matches = kb.iter_top_matches(query_string, limit=max(self.top_k, 5))
        candidates: List[RetrievalCandidate] = []
        parsed = preproc.get("parsed") or {}
        ingredient_tokens = {tok.lower() for tok in tokens if tok.isalpha() and len(tok) >= 3}
        for term, lexical, idx in matches:
            row = kb.row_at(idx)
            if kb is self.kb_snomed and not self._is_valid_snomed_term(term):
                continue
            if kb is self.kb_med and not self._is_valid_med_term(row, term, ingredient_tokens, parsed):
                continue
            semantic = self._semantic_similarity(query_string, term)
            combined = self._combine(lexical, semantic)
            combined = self._apply_dose_bonus(combined, term, parsed)
            combined = self._apply_combination_penalty(kb, combined, term)
            combined = self._apply_domain_bonus(entity, combined, term, tokens)
            candidate = RetrievalCandidate(
                system=kb.name,
                code=str(row.get("CODE", "")),
                cui=kb._cui_from_row(row),
                term=str(row.get("STR", "")),
                tty=str(row.get("TTY", "")).upper(),
                lexical=lexical,
                semantic=semantic or 0.0,
                combined=combined,
            )
            candidates.append(candidate)

        if not candidates:
            fallback = self._ingredient_single_fallback(query_string, ingredient_tokens, kb)
            if not fallback:
                return RetrievalResultV2("NO_MATCH", None, [])
            candidates.extend(fallback)

        heuristic = self._heuristic_candidates(entity, normalized, kb, tokens)
        if heuristic:
            candidates.extend(heuristic)

        candidates.sort(key=lambda c: c.combined, reverse=True)
        candidates = candidates[: self.top_k]

        promoted: List[RetrievalCandidate] = []
        for cand in candidates:
            best_row = kb.best_term_for_cui(cand.cui)
            if best_row is not None:
                best_tty = str(best_row.get("TTY", "")).upper()
                best_term = str(best_row.get("STR", cand.term))
                if kb is self.kb_snomed and not self._is_valid_snomed_term(best_term):
                    best_row = None
                if best_row is not None and self._priority(kb, best_tty) < self._priority(kb, cand.tty):
                    cand = RetrievalCandidate(
                        system=cand.system,
                        code=str(best_row.get("CODE", cand.code)),
                        cui=cand.cui,
                        term=best_term,
                        tty=best_tty,
                        lexical=cand.lexical,
                        semantic=cand.semantic,
                        combined=cand.combined,
                    )
            promoted.append(cand)

        grouped: Dict[str, RetrievalCandidate] = {}
        for cand in promoted:
            existing = grouped.get(cand.cui)
            if existing is None:
                grouped[cand.cui] = cand
                continue
            if self._is_better_candidate(kb, cand, existing):
                grouped[cand.cui] = cand

        final_candidates = list(grouped.values())
        final_candidates.sort(
            key=lambda c: (
                -c.combined,
                self._priority(kb, c.tty),
                -c.lexical,
                len(c.term),
            )
        )

        chosen = final_candidates[0] if final_candidates else None
        action = "NO_MATCH" if chosen is None else "AUTO_SELECT"
        return RetrievalResultV2(action, chosen, final_candidates)

    def _is_better_candidate(
        self,
        kb: KnowledgeBase,
        challenger: RetrievalCandidate,
        incumbent: RetrievalCandidate,
    ) -> bool:
        if challenger.combined != incumbent.combined:
            return challenger.combined > incumbent.combined
        prio_chall = self._priority(kb, challenger.tty)
        prio_inc = self._priority(kb, incumbent.tty)
        if prio_chall != prio_inc:
            return prio_chall < prio_inc
        if challenger.lexical != incumbent.lexical:
            return challenger.lexical > incumbent.lexical
        return len(challenger.term) < len(incumbent.term)

    def _is_valid_snomed_term(self, term: str) -> bool:
        clean = term.strip()
        if len(clean) < 4:
            return False
        alpha_tokens = [tok for tok in clean.split() if any(ch.isalpha() for ch in tok)]
        return len(alpha_tokens) >= 2

    def _is_valid_med_term(
        self,
        row: pd.Series,
        term: str,
        ingredient_tokens: set[str],
        parsed: Dict[str, object],
    ) -> bool:
        if not ingredient_tokens:
            return True
        term_lower = term.lower()
        if not any(tok in term_lower for tok in ingredient_tokens):
            return False
        # disallow combination products when query specifies a single ingredient
        combo_markers = ["/", " + ", " and ", " with "]
        is_combo = any(marker in term_lower for marker in combo_markers)
        if is_combo and len(ingredient_tokens) <= 1:
            return False
        dose_value = parsed.get("dose_value")
        dose_unit = parsed.get("dose_unit")
        if dose_value is None or not dose_unit:
            tty = str(row.get("TTY", "")).upper()
            return tty in {"IN", "PIN", "BN"}
        value_str = self._dose_to_string(dose_value)
        has_value = value_str in term_lower
        has_unit = str(dose_unit).lower() in term_lower
        if has_value or has_unit:
            return True
        tty = str(row.get("TTY", "")).upper()
        return tty in {"IN", "PIN", "BN"}

    def _dose_to_string(self, value: object) -> str:
        try:
            float_val = float(value)
            if float_val.is_integer():
                return str(int(float_val))
            return ("%.3f" % float_val).rstrip("0").rstrip(".")
        except Exception:
            return str(value)

    def _apply_dose_bonus(self, score: float, term: str, parsed: Dict[str, object]) -> float:
        dose_value = parsed.get("dose_value")
        dose_unit = parsed.get("dose_unit")
        if dose_value is None or not dose_unit:
            return score
        term_lower = term.lower()
        value_str = self._dose_to_string(dose_value)
        bonus = 0.0
        if value_str in term_lower:
            bonus += 0.1
        if str(dose_unit).lower() in term_lower:
            bonus += 0.05
        return min(1.0, score + bonus)

    def _apply_combination_penalty(self, kb: KnowledgeBase, score: float, term: str) -> float:
        lowered = term.lower()
        if kb is self.kb_med:
            if any(sep in lowered for sep in ["/", " + ", " and "]):
                score -= 0.05
        if kb is self.kb_snomed and len(term.strip()) < 6:
            score -= 0.05
        return max(0.0, score)

    def _apply_domain_bonus(
        self,
        entity: str,
        score: float,
        term: str,
        tokens: Sequence[str],
    ) -> float:
        term_lower = term.lower()
        token_set = {tok.lower() for tok in tokens}
        if entity == "medication" and "tablet" in term_lower and "tablet" in token_set:
            score += 0.02
        if entity in {"procedure", "diagnosis", "lab"}:
            if "x-ray" in token_set and "x-ray" in term_lower:
                score += 0.08
            if "radiograph" in term_lower and ("x-ray" in token_set or "xr" in token_set):
                score += 0.05
        return min(1.0, score)

    def _ingredient_single_fallback(
        self,
        query: str,
        ingredients: set[str],
        kb: KnowledgeBase,
    ) -> List[RetrievalCandidate]:
        results: List[RetrievalCandidate] = []
        for token in ingredients:
            exact_rows = kb.lookup_exact(token)
            if not exact_rows:
                continue
            best_row = None
            best_priority = 99
            for row in exact_rows:
                priority = self._priority(kb, str(row.get("TTY", "")).upper())
                if priority < best_priority:
                    best_priority = priority
                    best_row = row
            if best_row is None:
                continue
            term = str(best_row.get("STR", ""))
            lexical = fuzz.WRatio(query, term) / 100.0
            semantic = self._semantic_similarity(query, term) or 0.0
            combined = self._combine(lexical, semantic)
            results.append(
                RetrievalCandidate(
                    system=kb.name,
                    code=str(best_row.get("CODE", "")),
                    cui=kb._cui_from_row(best_row),
                    term=term,
                    tty=str(best_row.get("TTY", "")).upper(),
                    lexical=lexical,
                    semantic=semantic,
                    combined=combined,
                )
            )
        return results

    def _heuristic_candidates(
        self,
        entity: str,
        normalized_query: str,
        kb: KnowledgeBase,
        tokens: Sequence[str],
    ) -> List[RetrievalCandidate]:
        heuristics: Dict[str, Tuple[str, str, str]]
        normalized = normalized_query.lower()
        if entity == "diagnosis":
            heuristics = DIAG_HEURISTICS
        elif entity == "procedure":
            heuristics = PROC_HEURISTICS
        else:
            return []

        out: List[RetrievalCandidate] = []
        token_set = {tok.lower() for tok in tokens}
        for pattern, payload in heuristics.items():
            pattern_tokens = set(pattern.split())
            pattern_match = pattern in normalized or pattern_tokens.issubset(token_set)
            if pattern_match:
                code, term, tty = payload
                row = kb.row_by_code(code)
                if row is None:
                    continue
                lexical = fuzz.WRatio(normalized_query, term) / 100.0
                semantic = self._semantic_similarity(normalized_query, term) or 0.0
                combined = max(0.9, self._combine(lexical, semantic))
                out.append(
                    RetrievalCandidate(
                        system=kb.name,
                        code=str(code),
                        cui=kb._cui_from_row(row),
                        term=term,
                        tty=tty,
                        lexical=lexical,
                        semantic=semantic,
                        combined=combined,
                    )
                )
        if entity == "diagnosis":
            if {"right", "upper", "abdomen"}.issubset(token_set) and "pain" in token_set:
                row = kb.row_by_code("285388000")
                if row is not None:
                    term = "Right sided abdominal pain"
                    lexical = fuzz.WRatio(normalized_query, term) / 100.0
                    semantic = self._semantic_similarity(normalized_query, term) or 0.0
                    combined = max(0.9, self._combine(lexical, semantic))
                    out.append(
                        RetrievalCandidate(
                            system=kb.name,
                            code="285388000",
                            cui=kb._cui_from_row(row),
                            term=term,
                            tty="PT",
                            lexical=lexical,
                            semantic=semantic,
                            combined=combined,
                        )
                    )
        if entity == "procedure":
            if "appendix" in token_set and ("removal" in token_set or "surgery" in token_set):
                row = kb.row_by_code("80146002")
                if row is not None:
                    term = "Appendectomy"
                    lexical = fuzz.WRatio(normalized_query, term) / 100.0
                    semantic = self._semantic_similarity(normalized_query, term) or 0.0
                    combined = max(0.9, self._combine(lexical, semantic))
                    out.append(
                        RetrievalCandidate(
                            system=kb.name,
                            code="80146002",
                            cui=kb._cui_from_row(row),
                            term=term,
                            tty="PT",
                            lexical=lexical,
                            semantic=semantic,
                            combined=combined,
                        )
                    )
        return out


def load_engine_v2(base_path: Path) -> RetrievalEngineV2:
    rx_path = base_path / "Data" / "rxnorm_all_data.parquet"
    sn_path = base_path / "Data" / "snomed_all_data.parquet"
    return RetrievalEngineV2(str(rx_path), str(sn_path))
