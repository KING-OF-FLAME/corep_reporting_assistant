Below are the **complete contents** for the non-code foundation files you requested: `requirements.txt`, `.env.example`, `.gitignore`, and `README.md` for the project:

Directory (unchanged)

```
corep_reporting_assistant/
  app.py
  core/
    pipeline.py
    rag.py
    llm_prompts.py
    utils.py
  data/
    README.md
  outputs/
    .gitkeep
  .env.example
  .gitignore
  requirements.txt
  README.md
```

---

## `requirements.txt`

```txt
streamlit>=1.31,<2.0
openai>=1.40,<2.0
python-dotenv>=1.0,<2.0

pydantic>=2.7,<3.0
jsonschema>=4.21,<5.0

pandas>=2.1,<3.0
numpy>=1.26,<3.0

requests>=2.31,<3.0
pypdf>=4.0,<5.0

scikit-learn>=1.3,<2.0

openpyxl>=3.1,<4.0
```

Why this is “minimal but powerful”

* Streamlit UI
* OpenAI client
* Simple, reliable retrieval via TF-IDF (scikit-learn) instead of heavy vector DB dependencies
* PDF parsing + XLSX export
* Strong schema validation (jsonschema) + typed models (pydantic)

---

## `.env.example`

```env
# OpenAI
OPENAI_API_KEY=replace_me
OPENAI_MODEL=gpt-4.1-mini

# Retrieval / RAG
# Where you keep source docs to ingest (PDF / TXT). Default is ./data
RAG_SOURCE_DIR=./data

# Where the chunk index is saved (jsonl + pickle). Default is ./data/index
RAG_INDEX_DIR=./data/index

# Retrieval config
RAG_TOP_K=6

# App defaults
DEFAULT_REPORTING_CURRENCY=GBP
DEFAULT_UNIT=millions
```

---

## `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg-info/
dist/
build/

# Virtual envs
.venv/
venv/
env/

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Env files
.env

# Index / caches
data/index/
*.pickle
*.pkl

# Outputs
outputs/*
!outputs/.gitkeep

# Logs
*.log
```

---

## `README.md`

```md
# COREP Regulatory Reporting Assistant (Prototype)

A Streamlit demo prototype for an LLM-assisted **PRA COREP reporting analyst helper** focused on a small, traceable subset of COREP:
- **C 01.00 (Own Funds)** – key lines only
- **C 02.00 (Capital ratios / requirements extract)** – key ratios only

This is **information support** for regulatory reporting analysts. It is **not legal advice**. Always perform **human review** before any submission.

## What the demo does (end-to-end)
1. Accepts a natural-language question + a simple reporting scenario (numbers like CET1, deductions, TREA).
2. Retrieves relevant regulatory text excerpts (PRA Rulebook references + BoE COREP instructions + EBA pages you ingest).
3. Uses an OpenAI model (`gpt-4.1-mini`) to produce **structured JSON** aligned to a fixed schema for the chosen template extract.
4. Runs basic validation checks (missing fields, arithmetic consistency, units/currency).
5. Renders a clean, human-readable "Excel-like" extract table in the UI.
6. Exports:
   - CSV (and optionally XLSX) extracts
   - `audit_log.json` with per-cell citations and justification
   - `validation_report.json`

## Project structure
```

corep_reporting_assistant/
app.py
core/
pipeline.py
rag.py
llm_prompts.py
utils.py
data/
README.md
outputs/
.gitkeep
.env.example
.gitignore
requirements.txt
README.md

````

## Setup
### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
````

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

Required:

* `OPENAI_API_KEY`
* `OPENAI_MODEL` (default: `gpt-4.1-mini`)

## Add documents for retrieval (RAG)

Put source documents into `./data`:

* PDFs (e.g., BoE COREP own funds instructions)
* Text files (`.txt`) if you want to include extracted sections or notes

The prototype will ingest from `RAG_SOURCE_DIR` (default `./data`) and build an index in `RAG_INDEX_DIR` (default `./data/index`).

Notes:

* This prototype uses a lightweight TF-IDF retrieval approach (fast + minimal dependencies).
* Citations use stable `paragraph_id` values created during ingestion.

## Run the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## Outputs

Exports are written to `./outputs`:

* `corep_C01_extract.csv`
* `corep_C02_extract.csv`
* `audit_log.json`
* `validation_report.json`

`outputs/.gitkeep` ensures the directory exists in git.

## Prototype scope guardrails

* Only populates a small, curated set of key lines in C01 and C02 (template "extracts").
* Every populated cell must have at least one citation to retrieved regulatory text.
* If a field cannot be supported by retrieved text or scenario inputs, it is set to `null` with a clear note.


