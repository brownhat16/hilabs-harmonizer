# Retrieval Engine v2 (Prototype)

`Retrieval2/` hosts an experimental retriever that follows the hackathon spec for weighted lexical/semantic scoring, strict heuristics, and term promotion. It is not the default engine but is available via `main2.py` and used by the ensemble layer for cross-checking.

## Highlights

- **KnowledgeBase** – Wraps the RxNorm/SNOMED parquet data with efficient string/CUI/TTY indexes and exposes helpers like `best_term_for_cui`.
- **Weighted Scoring** – Combined score `S = 0.75 * lexical + 0.25 * semantic` (semantic gracefully falls back to token-set similarity when `sentence-transformers` is not available).
- **Ingredient Guardrails** – Ensures single-ingredient medication queries only return terms containing the ingredient; forbids combination products unless the query names multiple components.
- **Dose/Form Bonuses** – Rewards candidates that include matching strength and unit parsed by the preprocessor.
- **SNOMED Cleanup** – Rejects junk strings, gives additional bonus for imaging terminology, and injects deterministic matches for known pain/quadrant/appendectomy phrases.
- **Heuristic Fallbacks** – When fuzzy/lexical retrieval fails, a vocabulary-driven fallback returns the highest-priority IN/PIN term so we never leave a medication empty.

## Usage

```bash
DISABLE_SEMANTIC_MODEL=1 python main2.py "Chest xr" procedure
```

For bulk execution or in environments without SciPy/SentenceTransformers available, set the environment variable `DISABLE_SEMANTIC_MODEL=1` (the batch script does this automatically).

## Integration

`ensemble.py` uses Retrieval v2 to veto incorrect matches produced by v1 and to contribute alternative candidates. It supports the same preprocessor payload and returns harmonised `RetrievalResultV2` objects.

## Extending the Prototype

- Plug in a real embedding model by ensuring SciPy/SentenceTransformers are available; otherwise keep the environment variable disabled.
- Extend `DIAG_HEURISTICS` / `PROC_HEURISTICS` with more canonical patterns and codes.
- Adjust combination penalties or bonuses to reflect local dataset nuances.

This module acts as a living sandbox for ideas before they graduate into the production engine.

