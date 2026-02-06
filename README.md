```markdown
# COREP Reporting Assistant (Prototype)

Streamlit prototype for an **LLM-assisted PRA COREP Regulatory Reporting Assistant** for UK banks. The demo supports a constrained subset of COREP focused on:

- **C01.00 Own Funds (extract)**
- **C02.00 Capital Ratios (extract)**

It accepts a natural-language question + simple scenario inputs, retrieves relevant guidance excerpts (RAG), generates/repairs structured JSON output, runs deterministic validation checks, and exports CSV/XLSX plus an audit log showing which excerpts justify each populated field.

**Important**: This tool is for **information support for reporting analysts**. It is **not legal advice**. Outputs require **human review**.

---

## Features

- Streamlit UI with premium-style layout
- Lightweight local RAG (PDF/TXT/MD ingestion) with paragraph-level IDs
- Structured output contract (JSON) consumed by the rest of the pipeline
- Deterministic validation checks (missing fields, arithmetic consistency, unit/ratio sanity)
- Deterministic audit log derived from citations attached to each field
- Exports:
  - `outputs/corep_C0100_extract.csv`
  - `outputs/corep_C0200_extract.csv`
  - `outputs/corep_extracts.xlsx` (optional toggle)
  - `outputs/audit_log.json`
  - `outputs/validation_report.json`

---

## Repository Structure

```

corep_reporting_assistant/
app.py
core/
llm_prompts.py
rag.py
pipeline.py
utils.py
data/
README.md
demo_corep_instructions.txt
index/                 (generated; ignored by git)
outputs/
.gitkeep               (committed)
*.csv, *.xlsx, *.json  (generated; ignored by git)
.env                     (local only; ignored by git)
.env.example             (committed)
.gitignore
requirements.txt
README.md

````

---

## Setup

### 1) Create and activate a virtual environment (recommended)

Windows (PowerShell or CMD):
```bash
python -m venv .venv
.venv\Scripts\activate
````

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Copy `.env.example` to `.env` and set your OpenAI API key:

```bash
copy .env.example .env
```

Edit `.env`:

* `OPENAI_API_KEY=...`
* `OPENAI_MODEL=gpt-4.1-mini` (default)

Note: `.env` must never be committed.

---

## Add Documents for Retrieval (RAG)

The demo requires at least one source document in `./data` to produce grounded citations.

### Minimum demo corpus (quick start)

A ready-to-use test file is included:

* `data/demo_corep_instructions.txt`

You can replace/add:

* PRA Rulebook extracts (PDF/TXT)
* EBA COREP / ITS instructions (PDF/TXT/MD)

### Refresh corpus

From the app sidebar, click **Refresh corpus** to rebuild the local retrieval index at:

* `data/index/` (generated and ignored by git)

---

## Run the App

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

---

## How the Demo Works (High-Level)

1. **User input**

   * Question: what to populate
   * Scenario: CET1, deductions, AT1, T2, TREA (simple numeric inputs)

2. **Retrieval (RAG)**

   * Indexes `./data` documents
   * Retrieves top-k excerpts relevant to the question/templates

3. **LLM output (structured JSON)**

   * Attempts to produce a schema-shaped JSON object
   * If the SDK doesn’t support JSON mode or output is invalid, pipeline repairs/normalizes

4. **Deterministic autofill fallback**

   * If mandatory fields are missing but scenario inputs are present,
     the pipeline deterministically populates the extract values
     and attaches citations from retrieved excerpts
   * This keeps the prototype stable for demos while preserving traceability

5. **Validation**

   * Missing mandatory fields
   * Arithmetic checks:

     * CET1_NET = CET1_GROSS - CET1_DEDUCTIONS
     * TIER1_TOTAL = CET1_NET + AT1
     * TOTAL_OWN_FUNDS = TIER1_TOTAL + T2
     * Ratios derived from TREA
   * Ratio sanity checks and unit consistency

6. **Exports + Audit**

   * CSV/XLSX extracts generated
   * `audit_log.json` includes per-field citations and notes
   * `validation_report.json` captures flags

---

## Expected Calculations (Example)

If:

* CET1 gross = 520
* CET1 deductions = 45
* AT1 = 80
* T2 = 60
* TREA = 4750

Then:

* CET1 net = 475
* Tier 1 = 555
* Total own funds = 615
* CET1 ratio = 475 / 4750 * 100 = 10.00%

---

## Outputs

After a run, check `outputs/`:

* `corep_C0100_extract.csv`
* `corep_C0200_extract.csv`
* `corep_extracts.xlsx` (if enabled)
* `audit_log.json`
* `validation_report.json`

Note: Output files are intentionally not tracked by git (except `outputs/.gitkeep`).

---

## Troubleshooting

### Corpus status shows NOT READY

* Add at least one `.pdf`, `.txt`, or `.md` file into `data/`
* Click **Refresh corpus** in the sidebar

### LLM errors or empty structured output

* Verify `.env` contains a valid `OPENAI_API_KEY`
* Ensure you restarted Streamlit after changing `.env`
* Ensure retrieval has excerpts (non-empty Retrieval Transparency panel)

### Validation FAIL (mandatory fields null)

* If retrieval is empty, fields must remain null for traceability
* Ensure you have at least one document in `data/` and refresh corpus
* If documents exist, the deterministic autofill fallback will populate values using scenario inputs

### Windows file path issues

Run the app from the project root:

```bash
cd path\to\corep_reporting_assistant
streamlit run app.py
```

---

## Safety and Traceability

* Every populated field must include at least one citation from retrieved excerpts.
* If the required citation does not exist, the value should remain `null`.
* This prototype is designed to support analysts, not replace regulatory interpretation.

---
::contentReference[oaicite:0]{index=0}
```
