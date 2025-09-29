from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from rapidfuzz import fuzz, process as rf_process

TTY_WEIGHTS: Dict[str, float] = {
    "SCD": 1.0,
    "SCDC": 0.95,
    "SBD": 0.90,
    "SBDG": 0.85,
    "SBDC": 0.85,
    "SCDG": 0.82,
    "SCDF": 0.80,
    "SBDF": 0.80,
    "SCDFP": 0.78,
    "SBDFP": 0.78,
    "IN": 0.70,
    "PIN": 0.65,
    "MIN": 0.60,
    "BN": 0.50,
    "SY": 0.25,
    "PT": 1.0,
    "FN": 0.60,
}

STY_HINTS: Dict[str, Tuple[str, ...]] = {
    "medication": ("Drug", "Pharmacologic", "Clinical Drug", "Substance", "Dose Form"),
    "diagnosis": ("Disease", "Disorder", "Finding", "Syndrome"),
    "procedure": ("Procedure", "Surgery", "Therapy"),
    "lab": ("Laboratory", "Measurement", "Test", "Assay"),
}

PRIMARY_SYSTEM: Dict[str, str] = {
    "medication": "RXNORM",
    "diagnosis": "SNOMED",
    "procedure": "SNOMED",
    "lab": "SNOMED",
}


@dataclass
class Candidate:
    system: str
    code: str
    string: str
    tty: str
    sty: str
    cui: Optional[str]
    score: float = 0.0
    reason_flags: List[str] = field(default_factory=list)
    token_overlap: float = 0.0

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.system, self.code, self.string)


@dataclass
class MatchResult:
    action: str
    chosen: Optional[Candidate]
    top_candidates: List[Candidate]
    raw: Dict[str, Any]


class MatchLogger:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def log(self, result: MatchResult) -> None:
        payload = {
            "timestamp": time.time(),
            "raw": result.raw,
            "action": result.action,
            "chosen": result.chosen.as_tuple() if result.chosen else None,
            "top_candidates": [
                {
                    "system": c.system,
                    "code": c.code,
                    "string": c.string,
                    "tty": c.tty,
                    "cui": c.cui,
                    "score": round(c.score, 4),
                    "reasons": c.reason_flags,
                }
                for c in result.top_candidates
            ],
        }
        self.entries.append(payload)

    def flush(self) -> List[Dict[str, Any]]:
        data = list(self.entries)
        self.entries.clear()
        return data


class TerminologyIndex:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.df = frame.reset_index(drop=True)
        self.str_index: Dict[str, List[int]] = defaultdict(list)
        self.token_index: Dict[str, List[int]] = defaultdict(list)
        self.cui_index: Dict[str, List[int]] = defaultdict(list)
        self.doc_freq: Counter[str] = Counter()
        for idx, row in self.df.iterrows():
            norm = normalize(row["STR"])
            self.str_index[norm].append(idx)
            tokens = tokenize(norm)
            seen: set[str] = set()
            for tok in tokens:
                self.token_index[tok].append(idx)
                if tok not in seen:
                    self.doc_freq[tok] += 1
                    seen.add(tok)
            cui = row.get("CUI")
            if isinstance(cui, str) and cui:
                self.cui_index[cui].append(idx)

    def rows_from_ids(self, ids: Iterable[int]) -> List[pd.Series]:
        return [self.df.iloc[i] for i in ids]

    def lookup_exact(self, norm: str) -> List[pd.Series]:
        return self.rows_from_ids(self.str_index.get(norm, []))

    def lookup_tokens(self, tokens: Sequence[str]) -> List[pd.Series]:
        if not tokens:
            return []
        buckets = [self.token_index.get(tok, []) for tok in tokens]
        buckets = [b for b in buckets if b]
        if not buckets:
            return []
        if len(tokens) <= 3:
            candidate_ids = set().union(*buckets)
        else:
            sorted_tokens = sorted(tokens, key=lambda t: self.doc_freq.get(t, math.inf))
            rare = set(self.token_index.get(sorted_tokens[0], []))
            for tok in sorted_tokens[1:3]:
                rare &= set(self.token_index.get(tok, []))
            tail = set().union(*(self.token_index.get(tok, []) for tok in sorted_tokens[3:]))
            candidate_ids = rare | tail
        return self.rows_from_ids(candidate_ids)

    def expand_cui(self, cui: str) -> List[pd.Series]:
        return self.rows_from_ids(self.cui_index.get(cui, []))


class RetrievalEngine:
    def __init__(
        self,
        rxnorm_path: str,
        snomed_path: str,
        abbrev_map: Optional[Dict[str, str]] = None,
    ) -> None:
        rx_df = pd.read_parquet(rxnorm_path)
        sn_df = pd.read_parquet(snomed_path)
        self.indexes = {
            "RXNORM": TerminologyIndex(rx_df),
            "SNOMED": TerminologyIndex(sn_df),
        }
        self.abbrev_map = {k.lower(): v.lower() for k, v in (abbrev_map or {}).items()}
        self.logger = MatchLogger()

    def map_one(self, preproc: Dict[str, Any], *, log: bool = True) -> MatchResult:
        entity = preproc.get("entity", "").lower()
        norm_input = normalize(preproc.get("normalized", ""))
        if entity == "medication":
            norm_input = self._normalize_medication_query(norm_input)
        tokens = [tok.lower() for tok in preproc.get("tokens_no_stop", []) if tok]
        canonical_norm = self._apply_abbrev(norm_input, preproc.get("abbrev_expansions"))
        primary_system = PRIMARY_SYSTEM.get(entity, "SNOMED")

        candidates = self._collect_exact(canonical_norm, entity, primary_system)
        # RxNorm: augment with prioritized fuzzy matches if no strong exact
        if entity == "medication" and not self._has_priority_exact(candidates, entity):
            candidates += self._collect_fuzzy(canonical_norm, entity, primary_system)
            ingredient_fallback = self._ingredient_fallback(preproc)
            if ingredient_fallback:
                candidates.extend(ingredient_fallback)
        candidates = self._expand_with_cuis(candidates, canonical_norm, entity, primary_system)

        if not self._has_priority_exact(candidates, entity):
            token_cands = self._collect_token_candidates(tokens, entity, primary_system)
            # SNOMED: fuzzy if still weak
            if entity in {"diagnosis", "procedure", "lab"}:
                candidates += self._collect_fuzzy(canonical_norm, entity, primary_system)
            candidates = self._dedupe(candidates + token_cands)

        if not candidates:
            result = MatchResult("NO_MATCH", None, [], preproc)
            if log:
                self.logger.log(result)
            return result

        canonical_cui = preproc.get("canonical_cui") or self._infer_canonical_cui(candidates)
        scored = [self._score_candidate(preproc, cand, canonical_cui, canonical_norm) for cand in candidates]
        ranked = self._rank(scored)
        top5 = ranked[:5]
        best = top5[0]
        action = self._decide_action(best, preproc, entity)

        chosen = best if action in {"AUTO_ACCEPT", "FALLBACK_TO_INGREDIENT"} else None
        result = MatchResult(action, chosen, top5, preproc)
        if log:
            self.logger.log(result)
        return result

    # ---- Strategy helpers ----
    def _normalize_medication_query(self, text: str) -> str:
        # Normalize common abbreviations and units per strategy
        base = text
        # Expand common short forms
        repl = {
            "mg ": " mg ",
            "mcg ": " mcg ",
            "ug ": " mcg ",
            "µg ": " mcg ",
            " tab": " tablet",
            " cap": " capsule",
            " inj": " injectable",
            " po ": " oral ",
            " asa ": " aspirin ",
            " apap ": " acetaminophen ",
        }
        for k, v in repl.items():
            base = base.replace(k, v)
        base = self._normalize_per_volume(base)
        base = self._strip_salts(base)
        return normalize(base)

    def _normalize_per_volume(self, text: str) -> str:
        # Convert patterns like 125 mg/5 ml -> 25 mg/ml (approx unify)
        try:
            parts = text.split()
            out = []
            i = 0
            while i < len(parts):
                token = parts[i]
                if i + 2 < len(parts) and "/" in parts[i+1]:
                    # rough pass; leave unchanged for safety
                    out.append(token)
                    i += 1
                else:
                    out.append(token)
                    i += 1
            return " ".join(out)
        except Exception:
            return text

    def _strip_salts(self, text: str) -> str:
        salts = [" hcl", " hydrochloride", " sulfate", " mesylate", " sodium", " potassium"]
        base = text
        for salt in salts:
            base = base.replace(salt, "")
        return base

    def _apply_abbrev(self, norm: str, expansions: Optional[Dict[str, str]]) -> str:
        base = norm
        for src, tgt in (expansions or {}).items():
            base = base.replace(src.lower(), tgt.lower())
        for src, tgt in self.abbrev_map.items():
            base = base.replace(src, tgt)
        return normalize(base)

    def _collect_exact(self, norm: str, entity: str, primary: str) -> List[Candidate]:
        if not norm:
            return []
        systems = [primary]
        if entity != "medication":
            systems += [s for s in self.indexes if s != primary]
        collected: List[Candidate] = []
        for system in systems:
            rows = self.indexes[system].lookup_exact(norm)
            if rows:
                collected.extend(self._rows_to_candidates(rows, system))
                if system == primary:
                    break
        return collected

    def _expand_with_cuis(
        self,
        candidates: List[Candidate],
        norm: str,
        entity: str,
        primary: str,
    ) -> List[Candidate]:
        if not candidates:
            return candidates
        expanded = list(candidates)
        seen_cuis = {cand.cui for cand in candidates if cand.cui}
        search_systems = [primary] if entity == "medication" else list(self.indexes)
        for cui in seen_cuis:
            for system in search_systems:
                index = self.indexes[system]
                rows = index.expand_cui(cui)
                expanded.extend(self._rows_to_candidates(rows, system))
        expanded = self._dedupe(expanded)
        if norm:
            expanded.extend(self._collect_exact(norm, entity, primary))
        return self._dedupe(expanded)

    def _ingredient_fallback(self, preproc: Dict[str, Any]) -> List[Candidate]:
        entity = preproc.get("entity", "").lower()
        if entity != "medication":
            return []
        tokens = preproc.get("tokens_no_stop", []) or []
        parsed = preproc.get("parsed") or {}
        if not tokens or parsed.get("dose_value") is None:
            return []
        ingredient = None
        for tok in tokens:
            if tok.isalpha():
                ingredient = tok
                break
        if not ingredient:
            return []
        rx_index = self.indexes.get("RXNORM")
        if not rx_index:
            return []
        norm_ing = normalize(ingredient)
        rows = rx_index.lookup_exact(norm_ing)
        collected: List[Candidate] = []
        for row in rows:
            if row.get("TTY") not in {"IN", "PIN", "MIN"}:
                continue
            collected.append(
                Candidate(
                    system="RXNORM",
                    code=str(row["CODE"]),
                    string=row["STR"],
                    tty=row["TTY"],
                    sty=row["STY"],
                    cui=row["CUI"] if isinstance(row["CUI"], str) else None,
                )
            )
        return collected

    def _collect_token_candidates(
        self,
        tokens: Sequence[str],
        entity: str,
        primary: str,
    ) -> List[Candidate]:
        if not tokens:
            return []
        candidates: List[Candidate] = []
        search_systems = [primary] if entity == "medication" else list(self.indexes)
        threshold = 0.6 if entity == "medication" else 0.35
        for system in search_systems:
            index = self.indexes[system]
            rows = index.lookup_tokens(tokens)
            for cand in self._rows_to_candidates(rows, system):
                coverage = token_coverage(tokens, normalize(cand.string))
                if coverage >= threshold:
                    cand.token_overlap = coverage
                    candidates.append(cand)
        if entity == "medication":
            primary_cands = [c for c in candidates if c.system == primary]
            others = [c for c in candidates if c.system != primary]
            others.sort(key=lambda c: c.token_overlap, reverse=True)
            return primary_cands + others[:3]
        return candidates

    def _collect_fuzzy(self, norm: str, entity: str, primary: str) -> List[Candidate]:
        if not norm:
            return []
        results: List[Candidate] = []
        # Search only primary system for efficiency
        index = self.indexes.get(primary)
        if not index:
            return results
        # Build term list view
        series = index.df["STR"].astype(str)
        # Thresholds: allow descriptive matches; SNOMED slightly higher
        cutoff = 82 if primary == "SNOMED" else 80
        matches = rf_process.extract(
            norm,
            series.tolist(),
            scorer=fuzz.WRatio,
            limit=20,
            score_cutoff=cutoff,
        )
        # Map back to rows
        str_to_rows: Dict[str, List[int]] = defaultdict(list)
        for idx, s in enumerate(series.tolist()):
            str_to_rows[s].append(idx)
        for term, score, _ in matches:
            for rid in str_to_rows.get(term, []):
                row = index.df.iloc[rid]
                cand = Candidate(
                    system=primary,
                    code=str(row["CODE"]),
                    string=row["STR"],
                    tty=row["TTY"],
                    sty=row["STY"],
                    cui=row["CUI"] if isinstance(row["CUI"], str) else None,
                )
                results.append(cand)
        # RxNorm prioritization: SCD/SBD → PIN → IN → BN
        if primary == "RXNORM":
            order = {"SCD": 0, "SBD": 1, "SCDC": 2, "SBDG": 3, "PIN": 4, "IN": 5, "BN": 6}
            results.sort(key=lambda c: (order.get(c.tty, 99), -fuzz.WRatio(norm, normalize(c.string))))
        return self._dedupe(results)

    def _rows_to_candidates(self, rows: Iterable[pd.Series], system: str) -> List[Candidate]:
        output: List[Candidate] = []
        for row in rows:
            output.append(
                Candidate(
                    system=system,
                    code=str(row["CODE"]),
                    string=row["STR"],
                    tty=row["TTY"],
                    sty=row["STY"],
                    cui=row["CUI"] if isinstance(row["CUI"], str) else None,
                )
            )
        return output

    def _dedupe(self, candidates: List[Candidate]) -> List[Candidate]:
        unique: Dict[Tuple[str, str], Candidate] = {}
        for cand in candidates:
            key = (cand.system, cand.code)
            if key not in unique:
                unique[key] = cand
        return list(unique.values())

    def _infer_canonical_cui(self, candidates: List[Candidate]) -> Optional[str]:
        scores: Dict[str, float] = defaultdict(float)
        for cand in candidates:
            if cand.cui:
                scores[cand.cui] += TTY_WEIGHTS.get(cand.tty, 0.3)
        if not scores:
            return None
        return max(scores.items(), key=lambda kv: kv[1])[0]

    def _score_candidate(
        self,
        preproc: Dict[str, Any],
        cand: Candidate,
        canonical_cui: Optional[str],
        canonical_norm: Optional[str] = None,
    ) -> Candidate:
        entity = preproc.get("entity", "").lower()
        norm = canonical_norm or normalize(preproc.get("normalized", ""))
        query_tokens = [tok.lower() for tok in preproc.get("tokens_no_stop", []) if any(ch.isalpha() for ch in str(tok))]
        parsed = preproc.get("parsed") or {}
        candidate_norm = normalize(cand.string)

        cui_match = 1.0 if canonical_cui and cand.cui == canonical_cui else 0.0
        tty_norm = TTY_WEIGHTS.get(cand.tty, 0.2)
        exact_str = 1.0 if candidate_norm == norm else 0.0
        token_sim = fuzz.token_set_ratio(norm, candidate_norm) / 100.0 if norm else 0.0

        dose_form_score = 0.0
        strength_score = 0.0
        form_score = 0.0
        route_score = 0.0
        if entity == "medication":
            # Strategy-aligned scoring: exact(1.0), +strength(0.3), +form(0.2), +route(0.1)
            value = parsed.get("dose_value")
            unit = (parsed.get("dose_unit") or "").lower()
            form = (parsed.get("form") or "").lower()
            route_terms = {
                "oral",
                "injection",
                "iv",
                "po",
                "im",
                "sc",
                "intravenous",
                "intramuscular",
                "subcutaneous",
                "subcut",
                "sq",
            }
            route_aliases = {
                "po": {"oral"},
                "iv": {"intravenous"},
                "im": {"intramuscular"},
                "sc": {"subcutaneous", "subcut", "sq"},
            }

            candidate_tokens = set(tokenize(candidate_norm))

            value_str = None
            if value is not None:
                try:
                    value_str = str(int(value) if float(value).is_integer() else value).lower()
                except Exception:
                    value_str = str(value).lower()

            numeric_match = 0.0
            if value_str:
                numeric_patterns = {value_str}
                if unit:
                    numeric_patterns.update(
                        {
                            f"{value_str} {unit}",
                            f"{value_str}{unit}",
                            f"{value_str}/{unit}",
                        }
                    )
                numeric_match = max(
                    (fuzz.partial_ratio(pattern, candidate_norm) / 100.0)
                    for pattern in numeric_patterns
                )

            unit_match = 0.0
            if unit:
                unit_match = 1.0 if unit in candidate_tokens else fuzz.partial_ratio(unit, candidate_norm) / 100.0

            strength_score = min(1.0, 0.7 * numeric_match + 0.3 * unit_match)

            if form:
                form_score = fuzz.partial_ratio(form, candidate_norm) / 100.0

            route_hits: List[float] = []

            def _route_score(term: str) -> float:
                term_norm = term.lower()
                if term_norm in candidate_tokens:
                    return 1.0
                # Avoid partial-ratio inflation for very short terms
                if len(term_norm) <= 3:
                    return 0.0
                return fuzz.partial_ratio(term_norm, candidate_norm) / 100.0

            for rt in route_terms:
                route_hits.append(_route_score(rt))
            for abbr, synonyms in route_aliases.items():
                route_hits.append(_route_score(abbr))
                for syn in synonyms:
                    route_hits.append(_route_score(syn))
            route_score = max(route_hits) if route_hits else 0.0

            dose_form_score = min(1.0, 0.5 * strength_score + 0.3 * form_score + 0.2 * route_score)

        sty_binary = 1.0 if sty_matches(cand.sty, entity) else 0.0
        semantic_score = sty_binary
        if entity in {"diagnosis", "procedure", "lab"}:
            cand_sty = normalize(cand.sty) if cand.sty else ""
            hints = STY_HINTS.get(entity, ())
            if cand_sty and hints:
                ratios = [fuzz.partial_ratio(cand_sty, normalize(hint)) / 100.0 for hint in hints]
                if ratios:
                    semantic_score = max(max(ratios), sty_binary)

        if entity == "medication":
            score = (
                0.40 * exact_str
                + 0.25 * dose_form_score
                + 0.20 * min(tty_norm, 1.0)
                + 0.10 * token_sim
                + 0.05 * cui_match
            )
        elif entity in {"diagnosis", "procedure", "lab"}:
            score = (
                0.45 * exact_str
                + 0.25 * token_sim
                + 0.20 * semantic_score
                + 0.10 * cui_match
            )
        else:
            score = (
                0.35 * cui_match
                + 0.25 * min(tty_norm, 1.0)
                + 0.20 * exact_str
                + 0.10 * token_sim
                + 0.08 * dose_form_score
                + 0.02 * semantic_score
            )

        score = min(1.0, score)
        if entity == "medication" and query_tokens:
            if not any(tok in candidate_norm for tok in query_tokens):
                score *= 0.2
            elif cand.tty in {"IN", "PIN", "MIN"}:
                score = max(score, 0.6)
        elif entity in {"diagnosis", "procedure", "lab"} and query_tokens:
            directional_terms = {
                "left",
                "right",
                "upper",
                "lower",
                "anterior",
                "posterior",
                "bilateral",
                "medial",
                "lateral",
            }
            required = {tok for tok in query_tokens if tok in directional_terms}
            if required and not any(tok in candidate_norm for tok in required):
                score *= 0.3
        cand.score = score
        cand.reason_flags = self._reason_flags(
            cand,
            cui_match,
            tty_norm,
            exact_str,
            token_sim,
            dose_form_score,
            semantic_score,
            sty_binary,
        )
        if entity == "medication":
            cand.reason_flags.append(
                f"STRENGTH={strength_score:.2f}|FORM={form_score:.2f}|ROUTE={route_score:.2f}"
            )
        elif entity in {"diagnosis", "procedure", "lab"}:
            cand.reason_flags.append(f"SEMANTIC_RAW={semantic_score:.2f}")
        return cand

    def _reason_flags(
        self,
        cand: Candidate,
        cui_match: float,
        tty_norm: float,
        exact_str: float,
        token_sim: float,
        dose_form: float,
        semantic_score: float,
        sty_binary: float,
    ) -> List[str]:
        flags: List[str] = []
        if cui_match:
            flags.append("CUI_MATCH")
        flags.append(f"TTY={cand.tty}({tty_norm:.2f})")
        if exact_str:
            flags.append("EXACT_STR")
        flags.append(f"TOKEN_SIM={token_sim:.2f}")
        if dose_form:
            flags.append(f"DOSE_FORM={dose_form:.2f}")
        if semantic_score:
            flags.append(f"SEMANTIC={semantic_score:.2f}")
        if sty_binary:
            flags.append("STY_OK")
        return flags

    def _rank(self, candidates: List[Candidate]) -> List[Candidate]:
        # RxNorm prioritization
        def rx_order(tty: str) -> int:
            return {"SCD": 0, "SBD": 1, "SCDC": 2, "SBDG": 3, "PIN": 4, "IN": 5, "BN": 6}.get(tty, 99)
        candidates.sort(
            key=lambda c: (
                c.score,
                - (1 if c.system == "RXNORM" else 0),
                - (1 if c.system == "SNOMED" and c.tty == "PT" else 0),
                -rx_order(c.tty) if c.system == "RXNORM" else 0,
                TTY_WEIGHTS.get(c.tty, 0.0),
                c.token_overlap,
            ),
            reverse=True,
        )
        return candidates

    def _has_priority_exact(self, candidates: List[Candidate], entity: str) -> bool:
        if entity == "medication":
            return any(c.tty in {"SCD", "SCDC"} and "EXACT_STR" in c.reason_flags for c in candidates)
        return any(c.system == "SNOMED" and c.tty == "PT" and "EXACT_STR" in c.reason_flags for c in candidates)

    def _decide_action(self, best: Candidate, preproc: Dict[str, Any], entity: str) -> str:
        score = best.score
        if score >= 0.90:
            return "AUTO_ACCEPT"
        if 0.70 <= score < 0.90:
            if entity == "medication" and preproc.get("is_combination"):
                best.reason_flags.append("COMBINATION_REVIEW")
            return "REVIEW"
        if entity == "medication":
            parsed = preproc.get("parsed") or {}
            value = parsed.get("dose_value")
            unit = parsed.get("dose_unit")
            # Non-standard dose -> fallback to IN
            if (value is not None or unit) and best.tty in {"IN", "PIN"}:
                best.reason_flags.append("FALLBACK_INGREDIENT")
                return "FALLBACK_TO_INGREDIENT"
            if self._missing_ingredient(preproc):
                best.reason_flags.append("MISSING_INGREDIENT")
                return "REVIEW"
        return "NO_MATCH"

    def _missing_ingredient(self, preproc: Dict[str, Any]) -> bool:
        parsed = preproc.get("parsed") or {}
        if parsed.get("ingredient"):
            return False
        tokens = preproc.get("tokens_no_stop", []) or []
        alpha_tokens = [tok for tok in tokens if any(ch.isalpha() for ch in tok)]
        if not alpha_tokens:
            return True
        return all(tok in {"tablet", "capsule", "cap", "tab"} for tok in map(str.lower, alpha_tokens))


def normalize(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def tokenize(text: str) -> List[str]:
    return [tok for tok in text.split() if tok]


def token_coverage(tokens: Sequence[str], candidate_norm: str) -> float:
    if not tokens:
        return 0.0
    cand_tokens = set(tokenize(candidate_norm))
    overlap = sum(1 for tok in tokens if tok in cand_tokens)
    return overlap / max(len(tokens), 1)


def sty_matches(sty: str, entity: str) -> bool:
    hints = STY_HINTS.get(entity, ())
    return any(hint.lower() in sty.lower() for hint in hints)
