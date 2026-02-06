# core/llm_prompts.py
from __future__ import annotations

"""
Prompt pack for the COREP Regulatory Reporting Assistant prototype.

Design goals:
- Strict JSON-only outputs (no markdown)
- Null-if-unknown guardrails
- Mandatory citations per populated field
- Stable, reusable templates with minimal runtime string logic
"""

SYSTEM_PROMPT = """You are a Product & Reporting Analyst Assistant for UK bank regulatory reporting (PRA Rulebook / EBA COREP).
You provide information support for reporting analysts. You are NOT a legal advisor. Always recommend human review.

Hard rules:
- Use ONLY the provided retrieved regulatory text excerpts as authoritative sources.
- Do NOT fabricate rules, definitions, paragraph numbers, or template requirements.
- If you cannot find support in the retrieved text, set the field value to null and explain what is missing in "notes".
- Every populated field MUST include at least 1 citation object with source_url + paragraph_id + a short quote (<= 25 words).
- Output MUST be valid JSON matching the provided JSON schema. No markdown, no extra text.

Style rules:
- Prefer numeric values for "value" where possible.
- Keep units consistent with the user scenario; if ambiguous, set value null and ask for unit in notes.
- If a value is computed, include calculation.method="derived", a formula string, and list of input field_codes.
"""

RETRIEVAL_GROUNDED_ANSWER_PROMPT = """TASK:
Given:
(1) the user question,
(2) the reporting scenario,
(3) retrieved regulatory excerpts (with paragraph_id + source_url),
produce structured COREP output JSON that populates ONLY these templates: C01.00 and/or C02.00 (as requested).
Populate only the agreed "template extract" field list provided by the application.

INPUTS:
- user_question: {user_question}
- scenario_json: {scenario_json}
- reporting_date: {reporting_date}
- requested_templates: {requested_templates}
- allowed_cells_catalog: {allowed_cells_catalog}
- retrieved_excerpts: {retrieved_excerpts}

REQUIREMENTS:
- Use retrieved_excerpts to decide where each scenario item maps in the templates.
- If the mapping, sign convention, ratio convention, or required definition is not supported by retrieved_excerpts, set value=null and explain in notes.
- Each populated cell MUST cite the specific excerpt(s) that justify the mapping/convention.
- If user provided a numeric value, you may set calculation.method="user_provided" but you MUST still cite at least one excerpt supporting the reporting placement/convention.
- Do not create any cells outside allowed_cells_catalog.
- Do not guess missing numbers. Use null if not given and not derivable.

OUTPUT:
Return JSON that matches the schema exactly:
- templates: list of template objects
- cells: list of cell objects for allowed cells only
- validation_notes: initial notes (e.g., assumptions, missing inputs)
No additional keys. No additional text.
"""

VALIDATION_PROMPT = """TASK:
You are validating the previously generated COREP structured JSON for C01.00 and C02.00 template extracts.

INPUTS:
- structured_output_json: {structured_output_json}
- validation_rules: {validation_rules}
- retrieved_excerpts: {retrieved_excerpts}

CHECKS:
- Missing required cells: identify any allowed/mandatory field_code with value=null.
- Arithmetic consistency: totals and ratios.
- Unit/currency consistency: all monetary values must share currency and unit; ratios must be percent.
- Sign conventions: deductions should not increase CET1; if unsure, flag.
- Citation completeness: every non-null cell must have >=1 citation; if not, flag.

OUTPUT (JSON only):
{{
  "is_valid": boolean,
  "flags": [
    {{
      "severity": "error"|"warning",
      "field_code": "string|null",
      "message": "string",
      "suggested_fix": "string|null"
    }}
  ],
  "computed_overrides": [
    {{
      "field_code": "string",
      "recommended_value": number|null,
      "reason": "string"
    }}
  ]
}}

Rules:
- Do not fabricate citations.
- If a rule needs regulatory support but none exists in retrieved_excerpts, produce a warning instead of inventing support.
"""

AUDIT_LOG_PROMPT = """TASK:
Create an audit log for each populated cell (value != null) that explains:
- what was populated
- whether it was user_provided or derived
- which regulatory excerpt(s) justify the placement/convention
- what inputs were used (if derived)

INPUTS:
- structured_output_json: {structured_output_json}
- retrieved_excerpts: {retrieved_excerpts}

OUTPUT (JSON only):
{{
  "generated_at": "ISO-8601 datetime",
  "reporting_date": "YYYY-MM-DD",
  "entries": [
    {{
      "template_code": "C01.00|C02.00",
      "field_code": "string",
      "row_id": "string",
      "col_id": "string",
      "label": "string",
      "value": "number|string",
      "unit": "string",
      "currency": "string|null",
      "method": "user_provided|derived|direct",
      "formula": "string|null",
      "inputs": ["field_code", "..."],
      "citations": [
        {{"source_title":"string","source_url":"string","paragraph_id":"string","quote":"string"}}
      ],
      "justification": "1-3 sentences, strictly based on the citations",
      "retrieval_trace": {{
        "top_k": number,
        "excerpt_ids_used": ["paragraph_id", "..."]
      }}
    }}
  ]
}}

Rules:
- For each entry, use the citations already present in structured_output_json; do not invent new ones.
- If a populated cell has no citations, create an entry with justification explaining "missing citation" and mark severity in the text.
"""


def format_prompt(template: str, **kwargs: object) -> str:
    """
    Safe formatter for prompt templates.
    Ensures all kwargs are converted to string and template placeholders are filled.
    """
    safe_kwargs = {k: str(v) for k, v in kwargs.items()}
    return template.format(**safe_kwargs)
