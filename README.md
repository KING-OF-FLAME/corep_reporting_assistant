
# COREP Reporting Assistant (Prototype)

Streamlit prototype for an **LLM-assisted PRA COREP Regulatory Reporting Assistant** for UK banks.

The demo supports a constrained subset of COREP focused on:
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
│
├── app.py                          # Main Streamlit application entry point
│
├── core/                           # Core business logic modules
│   ├── __init__.py                # Package initializer
│   ├── llm_prompts.py             # LLM prompt templates and formatting
│   ├── rag.py                     # RAG retrieval logic and indexing
│   ├── pipeline.py                # Main processing pipeline orchestration
│   └── utils.py                   # Helper utilities and common functions
│
├── data/                          # Document corpus for RAG retrieval
│   ├── README.md                  # Instructions for adding documents
│   ├── demo_corep_instructions.txt # Sample regulatory guidance document
│   └── index/                     # Generated retrieval index
│       └── (vector embeddings and metadata - auto-generated, git-ignored)
│
├── outputs/                       # Generated reports and audit logs
│   ├── .gitkeep                   # Placeholder to track directory (committed)
│   ├── corep_C0100_extract.csv   # Own Funds extract (generated, git-ignored)
│   ├── corep_C0200_extract.csv   # Capital Ratios extract (generated, git-ignored)
│   ├── corep_extracts.xlsx       # Combined Excel workbook (generated, git-ignored)
│   ├── audit_log.json            # Field-level citations log (generated, git-ignored)
│   └── validation_report.json    # Validation results (generated, git-ignored)
│
├── .env                           # Local environment configuration (git-ignored)
├── .env.example                   # Environment variables template (committed)
├── .gitignore                     # Git ignore rules for sensitive/generated files
├── requirements.txt               # Python package dependencies
└── README.md                      # This documentation file
```

---

## Detailed Directory Explanation

### Root Level Files

- **`app.py`**: Main Streamlit application that orchestrates the UI, user inputs, and pipeline execution
- **`.env`**: Contains sensitive configuration (API keys) - never committed to git
- **`.env.example`**: Template showing required environment variables - committed for reference
- **`.gitignore`**: Specifies files/directories to exclude from version control
- **`requirements.txt`**: Lists all Python dependencies with versions
- **`README.md`**: Complete project documentation (this file)

### `core/` Directory

Contains all business logic separated into focused modules:

- **`__init__.py`**: Makes `core` a Python package
- **`llm_prompts.py`**: Defines system prompts, user prompt templates, and output schemas
- **`rag.py`**: Handles document ingestion, chunking, embedding, and retrieval
- **`pipeline.py`**: Orchestrates the complete workflow from input to validated output
- **`utils.py`**: Common helper functions (JSON parsing, file I/O, formatting)

### `data/` Directory

Storage for regulatory documents used in RAG retrieval:

- **`README.md`**: Instructions on adding and managing source documents
- **`demo_corep_instructions.txt`**: Sample regulatory text for testing
- **`index/`**: Auto-generated directory containing vector embeddings and retrieval metadata (git-ignored)

**Supported formats**: PDF, TXT, MD

### `outputs/` Directory

All generated files from pipeline execution:

- **`.gitkeep`**: Empty file to ensure `outputs/` directory is tracked by git
- **`corep_C0100_extract.csv`**: Own Funds data in CSV format
- **`corep_C0200_extract.csv`**: Capital Ratios data in CSV format
- **`corep_extracts.xlsx`**: Optional combined Excel workbook with multiple sheets
- **`audit_log.json`**: Complete audit trail with field-level citations
- **`validation_report.json`**: Validation results including errors and warnings

All generated outputs are git-ignored to prevent committing potentially sensitive data.

---

## Setup Instructions

### 1) Create and activate a virtual environment (recommended)

**Windows (PowerShell or CMD):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
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

**Windows:**
```bash
copy .env.example .env
```

**macOS/Linux:**
```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
```

**Important**: `.env` must never be committed to version control.

---

## Add Documents for Retrieval (RAG)

The demo requires at least one source document in `./data` to produce grounded citations.

### Minimum demo corpus (quick start)

A ready-to-use test file is included:
- `data/demo_corep_instructions.txt`

### Adding your own documents

Place regulatory documents in `data/`:
- PRA Rulebook extracts (PDF/TXT)
- EBA COREP / ITS instructions (PDF/TXT/MD)
- Internal compliance guidelines (TXT/MD)

### Refresh corpus

After adding documents:
1. Launch the Streamlit app
2. In the sidebar, click **Refresh corpus**
3. Wait for indexing to complete
4. The retrieval index will be rebuilt in `data/index/`

---

## Run the Application

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

---

## How the Demo Works (High-Level)

### Workflow Steps

1. **User Input**
   - Natural language question (e.g., "Populate C01.00 Own Funds")
   - Scenario inputs: CET1, deductions, AT1, T2, TREA

2. **Retrieval (RAG)**
   - Indexes all documents in `./data`
   - Retrieves top-k excerpts relevant to the question and templates
   - Each excerpt includes paragraph ID for citation tracking

3. **LLM Generation (Structured JSON)**
   - Sends retrieved excerpts + question + schema to LLM
   - Attempts to produce schema-compliant JSON
   - Pipeline repairs/normalizes invalid outputs

4. **Deterministic Autofill Fallback**
   - If mandatory fields are missing but scenario inputs exist
   - Pipeline deterministically populates values
   - Attaches citations from retrieved excerpts
   - Ensures demo stability while preserving traceability

5. **Validation**
   - **Missing field checks**: Flags null mandatory fields
   - **Arithmetic consistency**:
     - `CET1_NET = CET1_GROSS - CET1_DEDUCTIONS`
     - `TIER1_TOTAL = CET1_NET + AT1`
     - `TOTAL_OWN_FUNDS = TIER1_TOTAL + T2`
   - **Ratio calculations**: All ratios derived from TREA
   - **Sanity checks**: Unit consistency, ratio bounds
   - **Citation requirements**: Every field must have at least one citation

6. **Export + Audit**
   - CSV/XLSX extracts generated in `outputs/`
   - `audit_log.json` includes per-field citations and notes
   - `validation_report.json` captures all flags and warnings

---

## Expected Calculations (Example)

### Input Scenario:
- CET1 gross = 520
- CET1 deductions = 45
- AT1 = 80
- T2 = 60
- TREA = 4750

### Calculated Outputs:
- **CET1 net** = 520 - 45 = **475**
- **Tier 1 total** = 475 + 80 = **555**
- **Total own funds** = 555 + 60 = **615**
- **CET1 ratio** = (475 / 4750) × 100 = **10.00%**
- **Tier 1 ratio** = (555 / 4750) × 100 = **11.68%**
- **Total capital ratio** = (615 / 4750) × 100 = **12.95%**

---

## Output Files

After running the pipeline, check `outputs/` directory:

| File | Description |
|------|-------------|
| `corep_C0100_extract.csv` | Own Funds extract in CSV format |
| `corep_C0200_extract.csv` | Capital Ratios extract in CSV format |
| `corep_extracts.xlsx` | Combined Excel workbook (if enabled in UI) |
| `audit_log.json` | Complete audit trail with field-level citations |
| `validation_report.json` | Validation results with errors and warnings |

**Note**: Output files are intentionally not tracked by git (except `outputs/.gitkeep`).

---

## Troubleshooting

### Corpus status shows "NOT READY"

**Problem**: No documents indexed for retrieval

**Solution**:
1. Add at least one `.pdf`, `.txt`, or `.md` file into `data/`
2. Click **Refresh corpus** in the sidebar
3. Wait for indexing to complete

---

### LLM errors or empty structured output

**Problem**: API connection or authentication issues

**Solution**:
1. Verify `.env` contains a valid `OPENAI_API_KEY`
2. Ensure you restarted Streamlit after changing `.env`
3. Check API key has sufficient credits/quota
4. Ensure retrieval has excerpts (check Retrieval Transparency panel)

---

### Validation FAIL (mandatory fields null)

**Problem**: Fields remain null after processing

**Causes & Solutions**:
- **Empty retrieval**: Ensure documents exist in `data/` and refresh corpus
- **Missing citations**: If no relevant excerpts found, fields stay null for traceability
- **Autofill disabled**: Check that scenario inputs are provided
- **Expected behavior**: If documents exist, autofill fallback will populate using scenario inputs

---

### Windows file path issues

**Problem**: Module import errors or path not found

**Solution**:
Always run from project root:
```bash
cd C:\path\to\corep_reporting_assistant
streamlit run app.py
```

---

### Output files not appearing

**Problem**: Files not generated in `outputs/` directory

**Solution**:
1. Check console for error messages
2. Ensure `outputs/` directory exists
3. Verify write permissions on `outputs/` directory
4. Check validation passed (files only generated on success)

---

## Safety and Traceability

### Design Principles

1. **Citation Requirement**: Every populated field must include at least one citation from retrieved excerpts
2. **Null Preservation**: If the required citation does not exist, the value remains `null`
3. **Human-in-the-Loop**: This prototype is designed to support analysts, not replace regulatory interpretation
4. **Audit Trail**: Complete lineage from source document → retrieval → field population
5. **Validation Transparency**: All checks and failures explicitly logged

### Important Disclaimers

- ⚠️ **Not Legal Advice**: This tool provides information support only
- ⚠️ **Human Review Required**: All outputs must be verified by qualified analysts
- ⚠️ **Prototype Status**: Not intended for production regulatory submission
- ⚠️ **Data Privacy**: Ensure no sensitive/confidential data in demo documents

---

## Development Notes

### Technology Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4 Turbo
- **RAG**: Custom lightweight implementation with vector embeddings
- **Data Processing**: Pandas, NumPy
- **Export Formats**: CSV, XLSX (openpyxl), JSON

### Extensibility

To add new COREP templates:
1. Define schema in `core/llm_prompts.py`
2. Add validation rules in `core/pipeline.py`
3. Update export logic for new template
4. Add corresponding documents to `data/`

---

## License

This is a prototype for demonstration purposes. Use at your own risk.

---

## Contact

For questions or feedback about this prototype, please contact the development team.

---

## Version History

- **v0.1.0** (Initial Prototype)
  - C01.00 Own Funds extract
  - C02.00 Capital Ratios extract
  - Basic RAG with PDF/TXT/MD support
  - Deterministic validation
  - Audit logging
  - CSV/XLSX export

---

## Future Enhancements

- [ ] Additional COREP templates (C03.00, C04.00, etc.)
- [ ] Advanced validation rules engine
- [ ] Multi-period comparison
- [ ] Enhanced document preprocessing
- [ ] Collaborative review workflow
- [ ] API endpoint for integration
- [ ] Advanced citation visualization

---

**Last Updated**: 2025-02-06
```
