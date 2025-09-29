# Clinical Concept Harmonizer

This project delivers an end-to-end pipeline for harmonising free-text clinical descriptions into standard terminologies. It was built for the HiLabs harmonisation hackathon and maps noisy inputs to:

- **RxNorm** for medications, strength and form driven.
- **SNOMED CT** for diagnoses, procedures, laboratories, and symptoms.

The repository includes preprocessing, retrieval, an ensemble ranking layer, and batch tooling to populate the official `Test.xlsx` submission file.

---

## 📌 Problem Summary

Healthcare data is full of duplicates, misspellings, abbreviations and locale-specific naming. Without normalisation, it is impossible to run analytics, drive clinical decision support, or exchange data safely. We attack this by:

1. Cleaning each description with an advanced medical preprocessor (abbreviation expansion, dose parsing, combination detection).
2. Retrieving candidate codes from RxNorm and/or SNOMED CT with two complementary engines.
3. Blending their results through an ensemble layer that favours high-confidence matches, yet falls back gracefully when data is incomplete.
4. Exporting harmonised codes, systems, and descriptions for bulk validation (`Data/Test.xlsx`).

---

## 🧱 Repository Layout

| Path | Purpose |
|------|---------|
| `Data/` | Raw terminologies, test workbook, and reference column guide. |
| `DataProcessing/` | Enhanced preprocessor implementation (normalisation, dose extraction, entity detection). |
| `Retrieval/` | Production RetrievalEngine (v1) used in the baseline CLI (`main.py`). |
| `Retrieval2/` | Experimental retrieval prototype with weighted lexical/semantic scoring and strict heuristics. |
| `ensemble.py` | Orchestrates both engines, fuses results, and exposes a simple harmonise function. |
| `batch_update_test.py` | Batch runner to fill `Data/Test.xlsx` with harmonised outputs. |
| `main.py` | CLI entry point that uses Retrieval v1. |
| `main2.py` | CLI entry point for Retrieval v2 (prototype). |

Each subdirectory contains its own `README.md` describing the internals.

---

## ⚙️ Installation & Environment

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

> **Note**: The Retrieval v2 prototype can optionally leverage `sentence-transformers`. During batch runs we disable that dependency via `DISABLE_SEMANTIC_MODEL=1` to avoid SciPy/Numpy compatibility issues.

---

## 🚀 Usage

### Single Query (Retriever v1)

```bash
python main.py "Paracetamol 500 mg" medication
```

### Single Query (Retriever v2 prototype)

```bash
DISABLE_SEMANTIC_MODEL=1 python main2.py "Chest xr" procedure
```

### Ensemble Harmonisation

```bash
DISABLE_SEMANTIC_MODEL=1 python - <<'PY'
from ensemble import harmonize_ensemble
from main import BASE_PATH

result = harmonize_ensemble("persistent pain on the upper right portion of your abdomen", "diagnosis", BASE_PATH)
print(result["ensemble"]["chosen"])
PY
```

### Batch Update of `Test.xlsx`

The hackathon submission requires all 400 rows in `Data/Test.xlsx` to be populated with the harmonised code, system, and description:

```bash
DISABLE_SEMANTIC_MODEL=1 python batch_update_test.py
```

This script reads the workbook, runs the ensemble for each row, and overwrites the `Standard System`, `Standard Code`, and `Standard Description` columns in-place.

---

## 🧠 Solution Overview

1. **Preprocessing (`DataProcessing/`)**
   - Normalises unicode, punctuation, casing.
   - Expands medical abbreviations and UK/US synonyms.
   - Extracts tokens without stop-words and parses dosage, units, forms, and combination markers.
   - Detects entity class when not supplied.

2. **Retrieval Engines**
   - **Retriever v1 (`Retrieval/`)**
     - Builds token indexes over RxNorm/SNOMED parquet files.
     - Combines exact, fuzzy, and token-based candidate sourcing with CUI expansion.
     - Scores by TTY priority, CUI matches, dose alignment, and sets actions (`AUTO_ACCEPT`, `REVIEW`, etc.).
   - **Retriever v2 (`Retrieval2/`)**
     - Applies weighted lexical/semantic similarity.
     - Enforces strict ingredient matching, heuristics for location-based SNOMED codes, dose bonuses/penalties, and optional sentence-transformer embeddings.
     - Includes curated heuristics for corner cases (e.g., “appendix removal surgery” → Appendectomy).

3. **Ensemble (`ensemble.py`)**
   - Runs both engines, normalises scores, and applies routing logic:
     - Dose-rich medications prefer v2.
     - Fallback to v2 when v1 yields `NO_MATCH`, and vice versa.
     - Blends candidate scores (60% `v1`, 40% `v2`) to rank CUIs, then outputs the highest priority term.

4. **Batch Pipeline (`batch_update_test.py`)**
   - Disables heavy semantic model during bulk execution.
   - Calls the ensemble per row in `Data/Test.xlsx`.
   - Writes the updated workbook for submission.

---

## ✅ Validation & Testing

| Command | Purpose |
|---------|---------|
| `python main.py` (no arguments) | Show usage & demo run. |
| `python main2.py` (no arguments) | Show usage & demo run for proto engine. |
| `python batch_update_test.py` | Populates `Data/Test.xlsx` with harmonised results (used before submission). |

Manual spot checks were performed using the hackathon’s sample queries: Paracetamol, Aspirin, Asthma, chest X-ray, appendectomy, bilirubin, and right-upper-quadrant abdominal pain.

---

## 📦 Submission Checklist

- [x] Updated `Data/Test.xlsx` with the required columns.
- [x] Public repository containing source code and this README.
- [x] Batch script (`batch_update_test.py`) for reproducibility.
- [x] Directory-level documentation (`DataProcessing/README.md`, `Retrieval/README.md`, `Retrieval2/README.md`).
- [x] Example commands and environment instructions.

---

## 🤝 Acknowledgements

- RxNorm & SNOMED CT reference data supplied with the challenge.
- Open-source libraries: `pandas`, `rapidfuzz`, `sentence-transformers` (optional), `numpy`.

Good luck to everyone participating in the harmonisation hackathon! 🎯

