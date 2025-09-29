"""Batch updater for Data/Test.xlsx using ensemble harmonization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Disable optional semantic model unless explicitly enabled
os.environ.setdefault("DISABLE_SEMANTIC_MODEL", "1")

USE_ENSEMBLE = os.environ.get("USE_ENSEMBLE", "0") == "1"

if USE_ENSEMBLE:
    from ensemble import harmonize_ensemble  # noqa: E402  pylint: disable=C0413
else:
    from main import harmonize as harmonize_v1  # noqa: E402  pylint: disable=C0413

from main import BASE_PATH  # noqa: E402  pylint: disable=C0413


TEST_PATH = Path("Data/Test.xlsx")


def _select_entity(row: pd.Series) -> str:
    if "Entity Type" in row and isinstance(row["Entity Type"], str):
        return row["Entity Type"].strip().lower()
    if "entity" in row and isinstance(row["entity"], str):
        return row["entity"].strip().lower()
    return "diagnosis"


def _extract_query(row: pd.Series) -> str:
    for key in ("Input Entity Description", "Input Description", "Description", "Raw Text"):
        if key in row and isinstance(row[key], str):
            return row[key].strip()
    # fallback to first column value
    return str(row.iloc[0]).strip()


def _choose_candidate(result: Dict[str, object]) -> Dict[str, Optional[str]]:
    if not USE_ENSEMBLE:
        std = result.get("standardized", {})
        return {
            "system": std.get("Standard System"),
            "code": std.get("Standard Code"),
            "description": std.get("Standard Description"),
        }

    ensemble = result.get("ensemble", {}) if isinstance(result, dict) else {}
    chosen = ensemble.get("chosen") if isinstance(ensemble, dict) else None

    if chosen:
        return {
            "system": chosen.get("system"),
            "code": chosen.get("code"),
            "description": chosen.get("term"),
        }

    # fall back to engine-specific choices if ensemble empty
    engine_v1 = result.get("engine_v1", {}) if isinstance(result, dict) else {}
    if engine_v1 and engine_v1.get("chosen"):
        c = engine_v1["chosen"]
        return {
            "system": c.get("system"),
            "code": c.get("code"),
            "description": c.get("term"),
        }

    engine_v2 = result.get("engine_v2", {}) if isinstance(result, dict) else {}
    if engine_v2 and engine_v2.get("chosen"):
        c = engine_v2["chosen"]
        return {
            "system": c.get("system"),
            "code": c.get("code"),
            "description": c.get("term"),
        }

    return {"system": None, "code": None, "description": None}


def update_test_spreadsheet() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Cannot locate {TEST_PATH}")

    df = pd.read_excel(TEST_PATH)
    systems = []
    codes = []
    descriptions = []

    for _, row in df.iterrows():
        query = _extract_query(row)
        entity = _select_entity(row)
        if not query:
            systems.append(None)
            codes.append(None)
            descriptions.append(None)
            continue

        if USE_ENSEMBLE:
            result = harmonize_ensemble(query, entity, BASE_PATH)
        else:
            result = harmonize_v1(query, entity, BASE_PATH)
        chosen = _choose_candidate(result)
        systems.append(chosen["system"])
        codes.append(chosen["code"])
        descriptions.append(chosen["description"])

    df["Standard System"] = systems
    df["Standard Code"] = codes
    df["Standard Description"] = descriptions

    df.to_excel(TEST_PATH, index=False)
    print(f"Updated {len(df)} rows in {TEST_PATH}")


if __name__ == "__main__":
    update_test_spreadsheet()
