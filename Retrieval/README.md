# Retrieval Engine v1

`Retrieval/` contains the production-grade harmonisation engine from the baseline solution. It indexes RxNorm and SNOMED CT to surface candidates that match the preprocessed query.

## Core Concepts

| Component | Description |
|-----------|-------------|
| `matcher.py` | Implements `RetrievalEngine`, `TerminologyIndex`, candidate scoring, and logging. |
| `Candidate` | Dataclass representing a potential normalised match (system, code, string, TTY, CUI, score). |
| `MatchResult` | Final outcome (action, chosen candidate, top candidates, raw payload). |

## Retrieval Flow

1. **Normalise Input** – Uses the preprocessor output (`normalized`, `tokens_no_stop`, parsed dose info).
2. **Exact, Fuzzy, and Token Retrieval** – Combines direct string lookup, fuzzy matching (`rapidfuzz.WRatio`), and token overlap via `TerminologyIndex`.
3. **CUI Expansion** – Once a candidate is found, adds all synonyms sharing the same CUI to strengthen recall.
4. **Scoring & Ranking** – Weighted blend of CUI match, TTY priority, exact match, token similarity, dose/form alignment, and semantic type hints.
5. **Actions** – Based on the best candidate's score/TTY:
   - `AUTO_ACCEPT` – confident match.
   - `REVIEW` – needs manual validation.
   - `FALLBACK_TO_INGREDIENT` – use an ingredient-level code when strength/form is missing.
   - `NO_MATCH` – nothing acceptable located.

## Usage

The engine is instantiated in `main.py`:

```python
engine = RetrievalEngine(rx_path, snomed_path)
result = engine.map_one(preprocessed_dict)
```

It is also leveraged inside the ensemble runner (`ensemble.py`) to provide production-grade candidates for blending.

## Customisation Tips

- Tune scoring coefficients in `_score_candidate` to prioritise different heuristics.
- Adjust fuzzy cut-offs or token coverage thresholds if the data set changes.
- Use `MatchLogger` to collect audit trails of decisions for analysis.

