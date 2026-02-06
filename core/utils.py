# core/utils.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from jsonschema import Draft202012Validator

# -----------------------------
# Core schema (JSON Schema)
# -----------------------------


def get_corep_json_schema() -> Dict[str, Any]:
    """
    JSON Schema for the structured LLM output contract.
    Kept inline to avoid extra files and reduce prototype complexity.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "COREPStructuredOutput",
        "type": "object",
        "required": ["reporting_date", "templates", "cells", "validation_notes"],
        "properties": {
            "reporting_date": {"type": "string", "description": "ISO date, e.g., 2025-12-31"},
            "entity": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "lei": {"type": "string"}},
                "additionalProperties": False,
            },
            "templates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["template_code", "template_name"],
                    "properties": {
                        "template_code": {"type": "string"},
                        "template_name": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "cells": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "template_code",
                        "field_code",
                        "row_id",
                        "col_id",
                        "label",
                        "value",
                        "unit",
                        "currency",
                        "time_reference",
                        "confidence",
                        "citations",
                    ],
                    "properties": {
                        "template_code": {"type": "string"},
                        "field_code": {"type": "string"},
                        "row_id": {"type": "string"},
                        "col_id": {"type": "string"},
                        "label": {"type": "string"},
                        "value": {
                            "anyOf": [{"type": "number"}, {"type": "string"}, {"type": "null"}]
                        },
                        "unit": {"type": "string"},
                        "currency": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "counterparty_class": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "time_reference": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "calculation": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string"},
                                "formula": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                "inputs": {"type": "array", "items": {"type": "string"}},
                            },
                            "additionalProperties": False,
                        },
                        "citations": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["source_title", "source_url", "paragraph_id", "quote"],
                                "properties": {
                                    "source_title": {"type": "string"},
                                    "source_url": {"type": "string"},
                                    "paragraph_id": {"type": "string"},
                                    "quote": {"type": "string"},
                                },
                                "additionalProperties": False,
                            },
                        },
                        "notes": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                    "additionalProperties": False,
                },
            },
            "validation_notes": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


def validate_structured_output_schema(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates payload against the JSON schema.
    Returns (is_valid, errors).
    """
    schema = get_corep_json_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)

    if not errors:
        return True, []

    msgs: List[str] = []
    for e in errors:
        path = ".".join([str(p) for p in e.path]) if e.path else "<root>"
        msgs.append(f"{path}: {e.message}")
    return False, msgs


# -----------------------------
# Catalog: template extracts (minimal, demo-friendly)
# -----------------------------


def get_allowed_cells_catalog() -> List[Dict[str, Any]]:
    """
    Defines the curated, minimal set of COREP extract cells supported by the prototype.

    Note:
    - row_id/col_id are prototype-defined stable identifiers for rendering.
    - field_code is the stable key used for validation and cross-field computations.
    """
    return [
        # ----------------
        # C01.00 (Own Funds) - key lines
        # ----------------
        {
            "template_code": "C01.00",
            "template_name": "Own Funds",
            "field_code": "OF_CET1_GROSS",
            "row_id": "r010",
            "col_id": "c010",
            "label": "CET1 before regulatory adjustments",
            "expected_unit": "millions",
            "value_type": "money",
            "mandatory": True,
        },
        {
            "template_code": "C01.00",
            "template_name": "Own Funds",
            "field_code": "OF_CET1_DEDUCTIONS",
            "row_id": "r020",
            "col_id": "c010",
            "label": "Total CET1 deductions",
            "expected_unit": "millions",
            "value_type": "money",
            "mandatory": True,
        },
        {
            "template_code": "C01.00",
            "template_name": "Own Funds",
            "field_code": "OF_CET1_NET",
            "row_id": "r030",
            "col_id": "c010",
            "label": "CET1 after regulatory adjustments",
            "expected_unit": "millions",
            "value_type": "money",
            "mandatory": True,
        },
        {
            "template_code": "C01.00",
            "template_name": "Own Funds",
            "field_code": "OF_AT1",
            "row_id": "r040",
            "col_id": "c010",
            "label": "Additional Tier 1 (AT1)",
            "expected_unit": "millions",
            "value_type": "money",
            "mandatory": True,
        },
        {
            "template_code": "C01.00",
            "template_name": "Own Funds",
            "field_code": "OF_TIER1_TOTAL",
            "row_id": "r050",
            "col_id": "c010",
            "label": "Tier 1 capital",
            "expected_unit": "millions",
            "value_type": "money",
            "mandatory": True,
        },
        {
            "template_code": "C01.00",
            "template_name": "Own Funds",
            "field_code": "OF_T2",
            "row_id": "r060",
            "col_id": "c010",
            "label": "Tier 2 (T2)",
            "expected_unit": "millions",
            "value_type": "money",
            "mandatory": True,
        },
        {
            "template_code": "C01.00",
            "template_name": "Own Funds",
            "field_code": "OF_TOTAL_OWN_FUNDS",
            "row_id": "r070",
            "col_id": "c010",
            "label": "Total Own Funds",
            "expected_unit": "millions",
            "value_type": "money",
            "mandatory": True,
        },
        # ----------------
        # C02.00 (Ratios extract)
        # ----------------
        {
            "template_code": "C02.00",
            "template_name": "Own Funds Requirements / Ratios",
            "field_code": "CAP_TREA",
            "row_id": "r010",
            "col_id": "c010",
            "label": "Total Risk Exposure Amount (TREA)",
            "expected_unit": "millions",
            "value_type": "money",
            "mandatory": True,
        },
        {
            "template_code": "C02.00",
            "template_name": "Own Funds Requirements / Ratios",
            "field_code": "CAP_CET1_RATIO",
            "row_id": "r020",
            "col_id": "c010",
            "label": "CET1 ratio (CET1 / TREA)",
            "expected_unit": "percent",
            "value_type": "percent",
            "mandatory": True,
        },
        {
            "template_code": "C02.00",
            "template_name": "Own Funds Requirements / Ratios",
            "field_code": "CAP_TIER1_RATIO",
            "row_id": "r030",
            "col_id": "c010",
            "label": "Tier 1 ratio (Tier 1 / TREA)",
            "expected_unit": "percent",
            "value_type": "percent",
            "mandatory": True,
        },
        {
            "template_code": "C02.00",
            "template_name": "Own Funds Requirements / Ratios",
            "field_code": "CAP_TOTAL_CAPITAL_RATIO",
            "row_id": "r040",
            "col_id": "c010",
            "label": "Total capital ratio (Total Own Funds / TREA)",
            "expected_unit": "percent",
            "value_type": "percent",
            "mandatory": True,
        },
    ]


def get_supported_templates() -> Dict[str, str]:
    catalog = get_allowed_cells_catalog()
    out: Dict[str, str] = {}
    for c in catalog:
        out[c["template_code"]] = c["template_name"]
    return out


# -----------------------------
# Validation data structures
# -----------------------------


@dataclass(frozen=True)
class ValidationFlag:
    severity: str  # "error"|"warning"
    field_code: Optional[str]
    message: str
    suggested_fix: Optional[str] = None


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    flags: List[ValidationFlag]
    computed_overrides: List[Dict[str, Any]]


# -----------------------------
# Helpers
# -----------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return None
    return None


def is_nullish(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def currency_normalize(cur: Optional[str]) -> Optional[str]:
    if cur is None:
        return None
    c = cur.strip().upper()
    return c or None


def unit_normalize(unit: str) -> str:
    u = (unit or "").strip().lower()
    if u in {"m", "mn", "million", "millions"}:
        return "millions"
    if u in {"%", "pct", "percent", "percentage"}:
        return "percent"
    if u in {"abs", "absolute"}:
        return "absolute"
    return unit.strip()


def find_cell(cells: List[Dict[str, Any]], field_code: str) -> Optional[Dict[str, Any]]:
    for c in cells:
        if c.get("field_code") == field_code:
            return c
    return None


def index_cells_by_field_code(cells: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for c in cells:
        fc = c.get("field_code")
        if isinstance(fc, str) and fc:
            out[fc] = c
    return out


def check_citations_present(cell: Dict[str, Any]) -> bool:
    citations = cell.get("citations")
    return isinstance(citations, list) and len(citations) >= 1


# -----------------------------
# Deterministic validation rules (prototype)
# -----------------------------


def apply_validation_rules(
    structured_output: Dict[str, Any],
    allowed_catalog: Optional[List[Dict[str, Any]]] = None,
) -> ValidationResult:
    """
    Deterministic validator:
    - Mandatory fields present (non-null)
    - Arithmetic consistency for own funds totals and ratios
    - Unit/currency consistency
    - Citation completeness for populated cells
    Produces computed_overrides suggestions (not automatically applied in this function).
    """
    if allowed_catalog is None:
        allowed_catalog = get_allowed_cells_catalog()

    flags: List[ValidationFlag] = []
    overrides: List[Dict[str, Any]] = []

    cells = structured_output.get("cells") if isinstance(structured_output, dict) else None
    if not isinstance(cells, list):
        return ValidationResult(
            is_valid=False,
            flags=[ValidationFlag("error", None, "Structured output missing 'cells' list")],
            computed_overrides=[],
        )

    by_fc = index_cells_by_field_code(cells)

    # Mandatory checks based on catalog
    for item in allowed_catalog:
        if not item.get("mandatory"):
            continue
        fc = item["field_code"]
        cell = by_fc.get(fc)
        if cell is None:
            flags.append(ValidationFlag("error", fc, "Missing mandatory cell in output", "Add the cell to output"))
            continue
        if is_nullish(cell.get("value")):
            flags.append(ValidationFlag("error", fc, "Mandatory field is null/missing", "Provide value or derivation"))

    # Citation completeness for non-null values
    for c in cells:
        fc = c.get("field_code")
        if not isinstance(fc, str):
            continue
        if not is_nullish(c.get("value")) and not check_citations_present(c):
            flags.append(
                ValidationFlag(
                    "error",
                    fc,
                    "Populated cell has no citations (must cite retrieved regulatory text)",
                    "Add at least one citation or set value to null",
                )
            )

    # Unit/currency consistency for money cells
    money_cells = [item for item in allowed_catalog if item.get("value_type") == "money"]
    currencies = set()
    units = set()
    for item in money_cells:
        fc = item["field_code"]
        cell = by_fc.get(fc)
        if not cell or is_nullish(cell.get("value")):
            continue
        currencies.add(currency_normalize(cell.get("currency")))
        units.add(unit_normalize(cell.get("unit", "")))

    currencies.discard(None)
    if len(currencies) > 1:
        flags.append(
            ValidationFlag(
                "error",
                None,
                f"Currency mismatch across monetary fields: {sorted(list(currencies))}",
                "Ensure all money fields use the same currency",
            )
        )

    if len(units) > 1:
        flags.append(
            ValidationFlag(
                "error",
                None,
                f"Unit mismatch across monetary fields: {sorted(list(units))}",
                "Ensure all money fields use the same unit (e.g., millions)",
            )
        )

    # Percent unit check
    for item in allowed_catalog:
        if item.get("value_type") != "percent":
            continue
        fc = item["field_code"]
        cell = by_fc.get(fc)
        if not cell or is_nullish(cell.get("value")):
            continue
        u = unit_normalize(cell.get("unit", ""))
        if u != "percent":
            flags.append(
                ValidationFlag(
                    "error",
                    fc,
                    f"Ratio field unit should be 'percent' but got '{cell.get('unit')}'",
                    "Set unit='percent' and use numeric percent values",
                )
            )

    # Arithmetic consistency (C01)
    cet1_gross = safe_float(by_fc.get("OF_CET1_GROSS", {}).get("value"))
    cet1_ded = safe_float(by_fc.get("OF_CET1_DEDUCTIONS", {}).get("value"))
    cet1_net = safe_float(by_fc.get("OF_CET1_NET", {}).get("value"))
    at1 = safe_float(by_fc.get("OF_AT1", {}).get("value"))
    tier1 = safe_float(by_fc.get("OF_TIER1_TOTAL", {}).get("value"))
    t2 = safe_float(by_fc.get("OF_T2", {}).get("value"))
    total = safe_float(by_fc.get("OF_TOTAL_OWN_FUNDS", {}).get("value"))

    # Sign sanity for deductions
    if cet1_ded is not None and cet1_ded < 0:
        flags.append(
            ValidationFlag(
                "warning",
                "OF_CET1_DEDUCTIONS",
                "CET1 deductions are negative. Typically deductions reduce CET1; confirm sign convention.",
                "Use a positive number for total deductions (and subtract in CET1 net formula)",
            )
        )

    # Derived recommended values
    if cet1_gross is not None and cet1_ded is not None:
        rec_cet1_net = cet1_gross - cet1_ded
        if cet1_net is None:
            overrides.append(
                {
                    "field_code": "OF_CET1_NET",
                    "recommended_value": round(rec_cet1_net, 6),
                    "reason": "Derived as OF_CET1_GROSS - OF_CET1_DEDUCTIONS",
                }
            )
        elif not _nearly_equal(cet1_net, rec_cet1_net):
            flags.append(
                ValidationFlag(
                    "error",
                    "OF_CET1_NET",
                    f"Arithmetic mismatch: OF_CET1_NET={cet1_net} but expected {rec_cet1_net} (gross - deductions)",
                    "Correct OF_CET1_NET or adjust inputs",
                )
            )

    if cet1_net is not None and at1 is not None:
        rec_tier1 = cet1_net + at1
        if tier1 is None:
            overrides.append(
                {"field_code": "OF_TIER1_TOTAL", "recommended_value": round(rec_tier1, 6), "reason": "Derived as CET1 net + AT1"}
            )
        elif not _nearly_equal(tier1, rec_tier1):
            flags.append(
                ValidationFlag(
                    "error",
                    "OF_TIER1_TOTAL",
                    f"Arithmetic mismatch: OF_TIER1_TOTAL={tier1} but expected {rec_tier1} (CET1 net + AT1)",
                    "Correct Tier 1 total or adjust inputs",
                )
            )

    if tier1 is not None and t2 is not None:
        rec_total = tier1 + t2
        if total is None:
            overrides.append(
                {
                    "field_code": "OF_TOTAL_OWN_FUNDS",
                    "recommended_value": round(rec_total, 6),
                    "reason": "Derived as Tier 1 + Tier 2",
                }
            )
        elif not _nearly_equal(total, rec_total):
            flags.append(
                ValidationFlag(
                    "error",
                    "OF_TOTAL_OWN_FUNDS",
                    f"Arithmetic mismatch: OF_TOTAL_OWN_FUNDS={total} but expected {rec_total} (Tier 1 + Tier 2)",
                    "Correct Total Own Funds or adjust inputs",
                )
            )

    # Arithmetic consistency (C02 ratios)
    trea = safe_float(by_fc.get("CAP_TREA", {}).get("value"))
    cet1_ratio = safe_float(by_fc.get("CAP_CET1_RATIO", {}).get("value"))
    tier1_ratio = safe_float(by_fc.get("CAP_TIER1_RATIO", {}).get("value"))
    total_ratio = safe_float(by_fc.get("CAP_TOTAL_CAPITAL_RATIO", {}).get("value"))

    if trea is not None and trea <= 0:
        flags.append(
            ValidationFlag(
                "error",
                "CAP_TREA",
                "TREA must be positive to compute ratios",
                "Provide a positive Total Risk Exposure Amount",
            )
        )

    if trea and trea > 0:
        if cet1_net is not None:
            rec = (cet1_net / trea) * 100.0
            if cet1_ratio is None:
                overrides.append(
                    {"field_code": "CAP_CET1_RATIO", "recommended_value": round(rec, 6), "reason": "Derived as (CET1 net / TREA) * 100"}
                )
            elif not _nearly_equal(cet1_ratio, rec, tol=1e-4):
                flags.append(
                    ValidationFlag(
                        "error",
                        "CAP_CET1_RATIO",
                        f"Ratio mismatch: CAP_CET1_RATIO={cet1_ratio}% but expected {rec}%",
                        "Correct CET1 ratio or inputs",
                    )
                )

        if tier1 is not None:
            rec = (tier1 / trea) * 100.0
            if tier1_ratio is None:
                overrides.append(
                    {"field_code": "CAP_TIER1_RATIO", "recommended_value": round(rec, 6), "reason": "Derived as (Tier 1 / TREA) * 100"}
                )
            elif not _nearly_equal(tier1_ratio, rec, tol=1e-4):
                flags.append(
                    ValidationFlag(
                        "error",
                        "CAP_TIER1_RATIO",
                        f"Ratio mismatch: CAP_TIER1_RATIO={tier1_ratio}% but expected {rec}%",
                        "Correct Tier 1 ratio or inputs",
                    )
                )

        if total is not None:
            rec = (total / trea) * 100.0
            if total_ratio is None:
                overrides.append(
                    {
                        "field_code": "CAP_TOTAL_CAPITAL_RATIO",
                        "recommended_value": round(rec, 6),
                        "reason": "Derived as (Total Own Funds / TREA) * 100",
                    }
                )
            elif not _nearly_equal(total_ratio, rec, tol=1e-4):
                flags.append(
                    ValidationFlag(
                        "error",
                        "CAP_TOTAL_CAPITAL_RATIO",
                        f"Ratio mismatch: CAP_TOTAL_CAPITAL_RATIO={total_ratio}% but expected {rec}%",
                        "Correct Total capital ratio or inputs",
                    )
                )

    # Ratio ordering (warning)
    if all(v is not None for v in [cet1_ratio, tier1_ratio, total_ratio]):
        if not (total_ratio >= tier1_ratio >= cet1_ratio):
            flags.append(
                ValidationFlag(
                    "warning",
                    None,
                    "Unusual ratio ordering: expected Total >= Tier 1 >= CET1. Confirm values and conventions.",
                    "Verify own funds levels and ratio calculations",
                )
            )

    is_valid = not any(f.severity == "error" for f in flags)
    return ValidationResult(is_valid=is_valid, flags=flags, computed_overrides=overrides)


def _nearly_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# -----------------------------
# Rendering: Excel-like extract tables
# -----------------------------


def build_template_extract_table(
    structured_output: Dict[str, Any],
    template_code: str,
    allowed_catalog: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Produces a clean extract table with stable row ordering and useful columns.
    """
    if allowed_catalog is None:
        allowed_catalog = get_allowed_cells_catalog()

    cells = structured_output.get("cells", [])
    if not isinstance(cells, list):
        cells = []

    by_fc = index_cells_by_field_code(cells)

    rows: List[Dict[str, Any]] = []
    for item in [c for c in allowed_catalog if c["template_code"] == template_code]:
        fc = item["field_code"]
        cell = by_fc.get(fc, {})
        val = cell.get("value", None)
        unit = cell.get("unit", item.get("expected_unit", ""))
        cur = cell.get("currency", None)
        conf = cell.get("confidence", None)
        method = (cell.get("calculation") or {}).get("method", None)
        citations = cell.get("citations") if isinstance(cell.get("citations"), list) else []
        citation_count = len(citations)

        rows.append(
            {
                "row_id": item["row_id"],
                "col_id": item["col_id"],
                "field_code": fc,
                "label": item["label"],
                "value": val,
                "unit": unit,
                "currency": cur,
                "confidence": conf,
                "method": method,
                "citations": citation_count,
                "notes": cell.get("notes", None),
            }
        )

    # Stable ordering by row_id then col_id
    df = pd.DataFrame(rows).sort_values(by=["row_id", "col_id"], ascending=True).reset_index(drop=True)

    # Friendly formatting: keep numeric as numeric; leave nulls
    return df


def extract_cell_citations(
    structured_output: Dict[str, Any],
    field_code: str,
) -> List[Dict[str, Any]]:
    cells = structured_output.get("cells", [])
    if not isinstance(cells, list):
        return []
    cell = find_cell(cells, field_code)
    if not cell:
        return []
    citations = cell.get("citations")
    return citations if isinstance(citations, list) else []


# -----------------------------
# Audit log (deterministic)
# -----------------------------


def build_audit_log(
    structured_output: Dict[str, Any],
    retrieval_top_k: int,
) -> Dict[str, Any]:
    """
    Creates an audit log strictly from the structured output (no new citations invented).
    """
    cells = structured_output.get("cells", [])
    if not isinstance(cells, list):
        cells = []

    reporting_date = structured_output.get("reporting_date", "")
    entries: List[Dict[str, Any]] = []

    for c in cells:
        if is_nullish(c.get("value")):
            continue

        citations = c.get("citations") if isinstance(c.get("citations"), list) else []
        calc = c.get("calculation") if isinstance(c.get("calculation"), dict) else {}
        method = calc.get("method") or "direct"
        formula = calc.get("formula")
        inputs = calc.get("inputs") if isinstance(calc.get("inputs"), list) else []

        # Strictly based on existing citations:
        if citations:
            justification = "Populated based on retrieved regulatory excerpts cited for placement and reporting convention."
            excerpt_ids_used = [x.get("paragraph_id") for x in citations if isinstance(x, dict) and x.get("paragraph_id")]
        else:
            justification = "Missing citation for populated cell. This should be corrected (set null or add citation)."
            excerpt_ids_used = []

        entries.append(
            {
                "template_code": c.get("template_code"),
                "field_code": c.get("field_code"),
                "row_id": c.get("row_id"),
                "col_id": c.get("col_id"),
                "label": c.get("label"),
                "value": c.get("value"),
                "unit": c.get("unit"),
                "currency": c.get("currency"),
                "method": method,
                "formula": formula if formula is not None else None,
                "inputs": inputs,
                "citations": citations,
                "justification": justification,
                "retrieval_trace": {"top_k": int(retrieval_top_k), "excerpt_ids_used": excerpt_ids_used},
            }
        )

    return {
        "generated_at": utc_now_iso(),
        "reporting_date": reporting_date,
        "entries": entries,
    }


# -----------------------------
# Exports
# -----------------------------


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_template_extracts(
    structured_output: Dict[str, Any],
    out_dir: str | Path,
    write_xlsx: bool = True,
) -> Dict[str, str]:
    """
    Writes CSV (and optionally XLSX) for supported templates.
    Returns paths dict.
    """
    out_dir_p = ensure_dir(out_dir)
    supported = get_supported_templates()

    paths: Dict[str, str] = {}

    # CSV per template
    for template_code in supported.keys():
        df = build_template_extract_table(structured_output, template_code)
        csv_path = out_dir_p / f"corep_{template_code.replace('.', '')}_extract.csv"
        df.to_csv(csv_path, index=False)
        paths[f"{template_code}_csv"] = str(csv_path)

    # XLSX workbook (optional)
    if write_xlsx:
        xlsx_path = out_dir_p / "corep_extracts.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for template_code, template_name in supported.items():
                df = build_template_extract_table(structured_output, template_code)
                sheet = template_code.replace(".", "")
                df.to_excel(writer, sheet_name=sheet, index=False)
        paths["xlsx"] = str(xlsx_path)

    return paths


def export_audit_and_validation(
    structured_output: Dict[str, Any],
    validation: ValidationResult,
    audit_log: Dict[str, Any],
    out_dir: str | Path,
) -> Dict[str, str]:
    out_dir_p = ensure_dir(out_dir)

    validation_payload = {
        "is_valid": validation.is_valid,
        "flags": [
            {
                "severity": f.severity,
                "field_code": f.field_code,
                "message": f.message,
                "suggested_fix": f.suggested_fix,
            }
            for f in validation.flags
        ],
        "computed_overrides": validation.computed_overrides,
        "generated_at": utc_now_iso(),
    }

    validation_path = out_dir_p / "validation_report.json"
    audit_path = out_dir_p / "audit_log.json"

    write_json(validation_path, validation_payload)
    write_json(audit_path, audit_log)

    return {"validation_report": str(validation_path), "audit_log": str(audit_path)}


# -----------------------------
# Environment / config helpers
# -----------------------------


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    openai_model: str
    rag_source_dir: str
    rag_index_dir: str
    rag_top_k: int
    default_currency: str
    default_unit: str


def load_config_from_env() -> AppConfig:
    """
    Loads config from environment variables.
    Assumes python-dotenv is used upstream (app/pipeline) to load .env.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

    rag_source_dir = os.getenv("RAG_SOURCE_DIR", "./data").strip()
    rag_index_dir = os.getenv("RAG_INDEX_DIR", "./data/index").strip()
    top_k_raw = os.getenv("RAG_TOP_K", "6").strip()

    default_currency = os.getenv("DEFAULT_REPORTING_CURRENCY", "GBP").strip().upper()
    default_unit = unit_normalize(os.getenv("DEFAULT_UNIT", "millions").strip())

    try:
        top_k = int(top_k_raw)
    except ValueError:
        top_k = 6

    if not key:
        raise ValueError("OPENAI_API_KEY is missing. Please set it in .env.")

    return AppConfig(
        openai_api_key=key,
        openai_model=model,
        rag_source_dir=rag_source_dir,
        rag_index_dir=rag_index_dir,
        rag_top_k=max(1, min(top_k, 20)),
        default_currency=default_currency,
        default_unit=default_unit,
    )


# -----------------------------
# Scenario normalization
# -----------------------------


def normalize_scenario(
    scenario: Dict[str, Any],
    default_currency: str,
    default_unit: str,
) -> Dict[str, Any]:
    """
    Normalizes scenario fields so downstream logic is predictable.
    Keeps prototype simple: leaves unknown keys intact, just normalizes unit/currency and known numeric fields.
    """
    out = dict(scenario)

    out["currency"] = currency_normalize(out.get("currency")) or default_currency
    out["unit"] = unit_normalize(str(out.get("unit") or default_unit))

    # Normalize known numeric fields if present
    for k in [
        "cet1_gross",
        "cet1_deductions",
        "at1",
        "t2",
        "trea",
        "cet1_net",
        "tier1_total",
        "total_own_funds",
    ]:
        if k in out:
            v = safe_float(out.get(k))
            out[k] = v

    return out
