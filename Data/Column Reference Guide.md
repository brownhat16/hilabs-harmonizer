# Columns Reference Guide

---

## 1. `CUI` — Concept Unique Identifier
- **Definition**: A unique identifier assigned to a concept.  
- **Purpose**: Groups together all terms from different vocabularies (SNOMED CT, RxNorm, ICD, etc.) that mean the same thing (are synonymous).  
- **Format**: Always starts with a `"C"` followed by 7 digits (e.g., `C0011849`).  
- **Example**:
  - `C0011849` → *Diabetes Mellitus*  
    - In RxNorm: `Diabetes Mellitus`  
    - In SNOMED CT: `44054006 | Diabetes mellitus (disorder) |`

---

## 2. `System` — Source Vocabulary (SAB)
- **Definition**: Abbreviation for the source vocabulary or terminology that contributed the concept.  
- **Purpose**: Identifies where the term came from (RxNorm, SNOMED CT, MeSH, ICD-10, etc.).  
- **Example**:
  - `RXNORM` → concepts from the RxNorm vocabulary.  
  - `SNOMEDCT_US` → concepts from the U.S. edition of SNOMED CT.  

---

## 3. `TTY` — Term Type
- **Definition**: Describes the role of the term within the source vocabulary.  
- **Purpose**: Differentiates between preferred names, synonyms, codes, ingredients, brand names, etc.  
- **Common Values**:
  - `PT` (SNOMED CT) → Preferred Term.  
  - `SY` (RxNorm, SNOMED CT) → Synonym.  
  - `SCD` (RxNorm) → Semantic Clinical Drug (normalized clinical drug concept).  
  - `BN` (RxNorm) → Brand Name.  
- **Example**:
  - RxNorm:  
    - `SCD` → *Metformin 500 MG Oral Tablet*  
    - `BN` → *Glucophage*  
  - SNOMED CT:  
    - `PT` → *Diabetes mellitus (disorder)*  
    - `SY` → *Sugar diabetes*  

---

## 4. `CODE` — Source Code
- **Definition**: The identifier used **within the source vocabulary** for the concept.  
- **Purpose**: Provides the original vocabulary-specific code.  
- **Format**: Varies by vocabulary.  
- **Example**:
  - RxNorm: `860975` → *Metformin 500 MG Oral Tablet*  
  - SNOMED CT: `44054006` → *Diabetes mellitus (disorder)*  

---

## 5. `STR` — String (Term Name)
- **Definition**: The actual human-readable name of the concept as it appears in the source vocabulary.  
- **Purpose**: Stores the textual representation of the concept.  
- **Example**:
  - RxNorm: `Metformin 500 MG Oral Tablet`  
  - SNOMED CT: `Diabetes mellitus (disorder)`  

---

## 6. `STY` — Semantic Type
- **Definition**: The broad category or semantic group.  
- **Purpose**: Provides higher-level grouping of concepts beyond source vocabularies.  
- **Examples**:
  - `T047` → *Disease or Syndrome*  
  - `T121` → *Pharmacologic Substance*  
- **Example**:
  - `C0011849` (*Diabetes Mellitus*) → `Disease or Syndrome`  
  - `C0025598` (*Metformin*) → `Pharmacologic Substance`  

---

# Summary Table

| Column  | Meaning                                | Example (RxNorm)                   | Example (SNOMED CT)                    |
|---------|----------------------------------------|------------------------------------|----------------------------------------|
| `CUI`   | Concept Unique Identifier              | `C0025598` (*Metformin*)           | `C0011849` (*Diabetes Mellitus*)       |
| `System`| Source vocabulary abbreviation (SAB)   | `RXNORM`                           | `SNOMEDCT_US`                          |
| `TTY`   | Term type in source vocabulary         | `SCD` (Semantic Clinical Drug)     | `PT` (Preferred Term)                  |
| `CODE`  | Code in source vocabulary              | `860975`                           | `44054006`                             |
| `STR`   | Human-readable string/term             | *Metformin 500 MG Oral Tablet*     | *Diabetes mellitus (disorder)*         |
| `STY`   | Semantic type                          | *Pharmacologic Substance*          | *Disease or Syndrome*                  |

---
