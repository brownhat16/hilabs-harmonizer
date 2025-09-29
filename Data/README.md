# Harmonizing Clinical Concepts

## Repository Overview
This repository contains all the necessary files and instructions required for this problem statement on harmonizing clinical concepts.

## Contents of the Zip Folder
- **`Test.xlsx`**  
  Excel workbook containing 400 input descriptions used for validation. The harmonisation pipeline overwrites/creates the following columns:
  - `Standard System`
  - `Standard Code`
  - `Standard Description`
  Use `python batch_update_test.py` to populate these columns via the ensemble.

- **`snomed_all_data.parquet`**  
  Parquet file containing all target codes and descriptions for the SNOMED CT coding system.

- **`rxnorm_all_data.parquet`**  
  Parquet file containing all target codes and descriptions for the RxNorm coding system.

- **`Column Reference Guide.md`**  
  Markdown file explaining the columns present in the SNOMED CT and RxNorm parquet files.

## Submission Guidelines
To complete your submission:

1. **Modify the `Test.xlsx` file** by appending the following three columns:
   - `Output Coding System`: Either `SNOMEDCT_US` or `RXNORM`
   - `Output Target Code`: Predicted code in **string** format
   - `Output Target Description`: Human-readable description of the predicted code

2. **Your solution in a public code repository.**

3. **Ensure your repository includes:**
   - A clear, concise `README.md` explaining:
     - Your technical solution
     - A short walkthrough of the repository
   - Code that can be easily tested with additional input files
