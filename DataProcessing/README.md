# DataProcessing Module

`DataProcessing/` houses the **EnhancedMedicalPreprocessor** used across the project. It transforms messy free-text inputs into structured artefacts that downstream retrieval engines consume.

## Key Components

| File | Responsibility |
|------|----------------|
| `enhanced_preprocessor.py` | Main preprocessor class, configuration dataclasses, and pipeline stages. |
| `handle_case_space.py` | Low-level text normalisation utilities (unicode, punctuation, component extraction). |
| `medical_abbreviations.py`, `token_completion.py`, `noise_remover.py`, `semantic_matcher.py` | Domain specific helper modules used by the pipeline. |

## Pipeline Overview

1. **Input Normalisation** – Unicode NFKC, quotes cleanup, lowercase, punctuation control.
2. **Tokenisation & Stop-word Removal** – Builds `tokens` and `tokens_no_stop` arrays.
3. **Abbreviation & Locale Expansion** – Expands clinical shorthand (e.g., `paracetamol` → `acetaminophen`, `XR` → `x-ray`).
4. **Dose/Form Parsing** – Regex-based extraction of numeric strengths, units, concentrations, and dosage forms.
5. **Combination Detection** – Splits multi-ingredient products for special handling.
6. **Entity Detection** – Optional classifier that infers medication/procedure/diagnosis/etc. when the caller does not supply it.
7. **Fuzzy Ingredient Correction** – Uses `rapidfuzz` and an ingredient vocabulary (built from RxNorm/SNOMED) to soften misspellings.

The output is a `ComprehensivePreprocessingResult` dataclass containing:

- `normalized`, `tokens_no_stop`
- `parsed` (dose value/unit, form, concentrations)
- `is_combination`, `components`
- `entity`, `entity_confidence`
- `abbrev_expansions`

## Extending the Preprocessor

- Add new abbreviations or locale mappings via `_initialize_abbrev_map` and `_initialize_british_us_map`.
- Control pipeline stages through `PreprocessingConfig` flags.
- Inject extra vocabulary for fuzzy correction with `augment_abbrev_map_from_vocabulary` (called automatically in `main.py`).

This normalised result is the single source of truth for both retrieval engines and is also reused by the ensemble layer and batch updater.

