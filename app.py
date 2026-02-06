# app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.pipeline import refresh_rag_index, run_corep_pipeline
from core.utils import (
    build_template_extract_table,
    extract_cell_citations,
    get_supported_templates,
    load_config_from_env,
)

# -----------------------------
# Page config (premium, clean)
# -----------------------------

st.set_page_config(
    page_title="COREP Reporting Assistant (Prototype)",
    page_icon="🧾",
    layout="wide",
)

# NOTE: user requested no emojis generally for code/output; page_icon is harmless UI.
# If you want zero emojis, change page_icon to None and remove the icon in title below.


# -----------------------------
# Styling (clean premium look)
# -----------------------------


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          :root {
            --bg: #0b1020;
            --panel: #0f1733;
            --panel2: #0c142d;
            --text: #e8edf7;
            --muted: rgba(232,237,247,0.72);
            --accent: #6ea8ff;
            --accent2: #9b8cff;
            --ok: #3ddc97;
            --warn: #ffd166;
            --err: #ff5c7a;
            --border: rgba(255,255,255,0.08);
          }

          .stApp { background: radial-gradient(1200px 900px at 15% 10%, rgba(110,168,255,0.22), transparent 55%),
                               radial-gradient(1000px 700px at 85% 20%, rgba(155,140,255,0.18), transparent 55%),
                               linear-gradient(180deg, var(--bg), #070b14 70%); color: var(--text); }

          /* Remove excessive padding in main container */
          .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }

          /* Headings */
          h1, h2, h3, h4 { color: var(--text) !important; }
          .muted { color: var(--muted); font-size: 0.95rem; }

          /* Panel cards */
          .card {
            background: linear-gradient(180deg, rgba(15,23,51,0.95), rgba(12,20,45,0.95));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.35);
          }
          .cardTitle { font-weight: 600; letter-spacing: 0.2px; }
          .kpiGrid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
          .kpi {
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px 14px;
          }
          .kpiLabel { color: var(--muted); font-size: 0.85rem; }
          .kpiValue { font-size: 1.35rem; font-weight: 700; margin-top: 2px; }
          .pill {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 6px 10px; border-radius: 999px;
            border: 1px solid var(--border); background: rgba(255,255,255,0.03);
            color: var(--muted); font-size: 0.85rem;
          }
          .pill b { color: var(--text); font-weight: 600; }

          /* Streamlit widgets */
          div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"] {
            background: rgba(255,255,255,0.03) !important;
            border-radius: 12px !important;
            border: 1px solid var(--border) !important;
          }
          .stButton>button {
            border-radius: 12px;
            border: 1px solid var(--border);
            background: linear-gradient(90deg, rgba(110,168,255,0.20), rgba(155,140,255,0.18));
            color: var(--text);
            font-weight: 600;
            padding: 0.55rem 0.9rem;
          }
          .stButton>button:hover { border-color: rgba(110,168,255,0.55); }

          /* Tables */
          .stDataFrame { border-radius: 14px; overflow: hidden; border: 1px solid var(--border); }

          /* Badges */
          .badgeOk { color: var(--ok); font-weight: 700; }
          .badgeWarn { color: var(--warn); font-weight: 700; }
          .badgeErr { color: var(--err); font-weight: 700; }

          /* Code blocks */
          pre { background: rgba(255,255,255,0.03) !important; border: 1px solid var(--border) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_css()


# -----------------------------
# Session state initialization
# -----------------------------


def _init_state() -> None:
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "corpus_ready" not in st.session_state:
        st.session_state.corpus_ready = False


_init_state()


# -----------------------------
# Helpers
# -----------------------------


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _format_money(v: Any, unit: str, currency: Optional[str]) -> str:
    if v is None or (isinstance(v, str) and not v.strip()):
        return "—"
    try:
        x = float(v)
        cur = (currency or "").upper()
        if unit == "millions":
            return f"{cur} {x:,.2f}m".strip()
        if unit == "absolute":
            return f"{cur} {x:,.2f}".strip()
        return f"{cur} {x:,.4f}".strip()
    except Exception:
        return str(v)


def _format_percent(v: Any) -> str:
    if v is None or (isinstance(v, str) and not v.strip()):
        return "—"
    try:
        x = float(v)
        return f"{x:.2f}%"
    except Exception:
        return str(v)


def _kpi_value(structured: Dict[str, Any], field_code: str) -> Any:
    cells = structured.get("cells", [])
    if not isinstance(cells, list):
        return None
    for c in cells:
        if c.get("field_code") == field_code:
            return c.get("value")
    return None


# -----------------------------
# App header
# -----------------------------

st.markdown(
    """
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;">
        <div>
          <h2 style="margin:0;">COREP Reporting Assistant (Prototype)</h2>
          <div class="muted" style="margin-top:6px;">
            Analyst support only. Not legal advice. Every populated field must be traceable to retrieved regulatory text.
          </div>
          <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
            <span class="pill">Templates: <b>C01.00</b> Own Funds (extract) + <b>C02.00</b> Ratios (extract)</span>
            <span class="pill">Exports: <b>CSV/XLSX</b> + <b>audit_log.json</b></span>
            <span class="pill">LLM output contract: <b>JSON schema</b></span>
          </div>
        </div>
        <div style="min-width:260px;">
          <div class="pill" style="justify-content:space-between; width:100%;">
            Corpus status:
            <b>{}</b>
          </div>
          <div class="muted" style="margin-top:8px;">
            Add PDFs/TXT to <b>./data</b>, then click Refresh corpus.
          </div>
        </div>
      </div>
    </div>
    """.format("READY" if st.session_state.corpus_ready else "NOT READY"),
    unsafe_allow_html=True,
)

st.write("")

# -----------------------------
# Sidebar: Config + corpus
# -----------------------------

with st.sidebar:
    st.markdown("### Configuration")

    load_dotenv()
    cfg = None
    try:
        cfg = load_config_from_env()
        st.caption(f"Model: {cfg.openai_model}")
        st.caption(f"RAG Source: {cfg.rag_source_dir}")
        st.caption(f"RAG Index: {cfg.rag_index_dir}")
        st.caption(f"Top-K: {cfg.rag_top_k}")
    except Exception as e:
        st.session_state.last_error = str(e)
        st.error("Config error. Check .env")
        st.caption(str(e))

    st.markdown("---")
    st.markdown("### Corpus")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Refresh corpus", use_container_width=True):
            if cfg:
                with st.spinner("Rebuilding retrieval index..."):
                    try:
                        refresh_rag_index(cfg.rag_source_dir, cfg.rag_index_dir)
                        st.session_state.corpus_ready = True
                        st.success("Corpus refreshed.")
                    except Exception as e:
                        st.session_state.last_error = str(e)
                        st.session_state.corpus_ready = False
                        st.error("Failed to refresh corpus.")
                        st.caption(str(e))
            else:
                st.error("Fix config first (OPENAI_API_KEY).")

    with col_b:
        if st.button("Open data folder", use_container_width=True):
            st.info("Data folder path:")
            st.code(str(Path("./data").resolve()))

    st.markdown("---")
    st.markdown("### Safety guardrails")
    st.caption("- No citation → value must be null.")
    st.caption("- No fabrication of rules/paragraphs.")
    st.caption("- Human review recommended.")

# -----------------------------
# Main: Inputs + scenario
# -----------------------------

supported_templates = get_supported_templates()

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Inputs")

    user_question = st.text_area(
        "User question",
        value="Populate COREP C01.00 and C02.00 key lines and ratios for reporting date, using the scenario below. Export CSV and audit log.",
        height=110,
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        reporting_date = st.text_input("Reporting date (YYYY-MM-DD)", value="2025-12-31")
    with c2:
        currency = st.text_input("Currency", value=(cfg.default_currency if cfg else "GBP"))
    with c3:
        unit = st.selectbox("Unit", options=["millions", "absolute"], index=0)

    template_selection = st.multiselect(
        "Templates (extract)",
        options=list(supported_templates.keys()),
        default=["C01.00", "C02.00"],
        help="Prototype supports only these two extracts.",
    )

    st.markdown("#### Scenario (simple)")
    s1, s2, s3 = st.columns([1, 1, 1])
    with s1:
        cet1_gross = st.number_input("CET1 gross", min_value=0.0, value=520.0, step=1.0)
        cet1_deductions = st.number_input("CET1 deductions", min_value=0.0, value=45.0, step=1.0)
    with s2:
        at1 = st.number_input("AT1", min_value=0.0, value=80.0, step=1.0)
        t2 = st.number_input("T2", min_value=0.0, value=60.0, step=1.0)
    with s3:
        trea = st.number_input("TREA", min_value=0.0, value=4750.0, step=10.0)

    st.caption("Tip: You can also paste a custom JSON scenario below (overrides fields above).")

    scenario_json_text = st.text_area(
        "Advanced scenario JSON (optional)",
        value="",
        height=120,
        placeholder='{"cet1_gross":520,"cet1_deductions":45,"at1":80,"t2":60,"trea":4750,"currency":"GBP","unit":"millions"}',
    )

    run_col1, run_col2 = st.columns([1, 1])
    with run_col1:
        run_clicked = st.button("Run assistant", use_container_width=True)
    with run_col2:
        write_xlsx = st.toggle("Write XLSX export", value=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Output preview")

    st.markdown(
        """
        <div class="muted">
          This panel shows KPIs, validation status, and export artifacts after you run the assistant.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_result is None:
        st.info("Run the assistant to see results.")
    else:
        result = st.session_state.last_result
        structured = result.structured_output

        # KPIs
        k1 = _kpi_value(structured, "OF_CET1_NET")
        k2 = _kpi_value(structured, "OF_TIER1_TOTAL")
        k3 = _kpi_value(structured, "OF_TOTAL_OWN_FUNDS")
        k4 = _kpi_value(structured, "CAP_CET1_RATIO")

        st.markdown(
            """
            <div class="kpiGrid">
              <div class="kpi">
                <div class="kpiLabel">CET1 (net)</div>
                <div class="kpiValue">{}</div>
              </div>
              <div class="kpi">
                <div class="kpiLabel">Tier 1</div>
                <div class="kpiValue">{}</div>
              </div>
              <div class="kpi">
                <div class="kpiLabel">Total Own Funds</div>
                <div class="kpiValue">{}</div>
              </div>
              <div class="kpi">
                <div class="kpiLabel">CET1 ratio</div>
                <div class="kpiValue">{}</div>
              </div>
            </div>
            """.format(
                _format_money(k1, unit, currency),
                _format_money(k2, unit, currency),
                _format_money(k3, unit, currency),
                _format_percent(k4),
            ),
            unsafe_allow_html=True,
        )

        st.write("")
        val = result.validation
        badge = '<span class="badgeOk">PASS</span>' if val.is_valid else '<span class="badgeErr">FAIL</span>'
        st.markdown(f"Validation status: {badge}", unsafe_allow_html=True)

        if result.schema_valid:
            st.markdown("Schema: <span class='badgeOk'>VALID</span>", unsafe_allow_html=True)
        else:
            st.markdown("Schema: <span class='badgeWarn'>INVALID</span>", unsafe_allow_html=True)

        if result.schema_errors:
            with st.expander("Schema errors"):
                for e in result.schema_errors:
                    st.code(e)

        if val.flags:
            with st.expander("Validation flags", expanded=True):
                for f in val.flags:
                    cls = "badgeErr" if f.severity == "error" else "badgeWarn"
                    fc = f.field_code or "N/A"
                    st.markdown(f"- <span class='{cls}'>{f.severity.upper()}</span> [{fc}] {f.message}", unsafe_allow_html=True)
                    if f.suggested_fix:
                        st.caption(f"Suggested fix: {f.suggested_fix}")
        else:
            st.success("No validation flags.")

        st.write("")
        st.markdown("#### Exports")
        for k, p in result.export_paths.items():
            st.caption(f"{k}: {p}")
        for k, p in result.audit_paths.items():
            st.caption(f"{k}: {p}")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Run pipeline (action)
# -----------------------------

if run_clicked:
    if not cfg:
        st.error("Missing configuration. Please set OPENAI_API_KEY in .env.")
    else:
        # Scenario assembly
        scenario: Dict[str, Any] = {
            "cet1_gross": float(cet1_gross),
            "cet1_deductions": float(cet1_deductions),
            "at1": float(at1),
            "t2": float(t2),
            "trea": float(trea),
            "currency": currency.strip().upper(),
            "unit": unit,
        }

        # Optional override from JSON
        if scenario_json_text.strip():
            override = _safe_json_loads(scenario_json_text.strip())
            if override is None:
                st.error("Advanced scenario JSON is invalid. Fix JSON or clear the field.")
                st.stop()
            scenario.update(override)

        # Ensure corpus is at least attempted
        if not st.session_state.corpus_ready:
            st.warning("Corpus is not marked READY. If retrieval is empty, add docs to ./data and click Refresh corpus.")

        with st.spinner("Running retrieval + LLM + validation..."):
            try:
                result = run_corep_pipeline(
                    user_question=user_question,
                    scenario=scenario,
                    reporting_date=reporting_date,
                    requested_templates=template_selection,
                    out_dir="./outputs",
                    force_rebuild_index=False,
                    write_xlsx=bool(write_xlsx),
                )
                st.session_state.last_result = result
                st.session_state.last_error = None
                st.success("Run complete.")
            except Exception as e:
                st.session_state.last_error = str(e)
                st.error("Pipeline error.")
                st.code(str(e))

# -----------------------------
# Results viewer (tables + citations + retrieval transparency)
# -----------------------------

if st.session_state.last_result is not None:
    result = st.session_state.last_result
    structured = result.structured_output

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Template extracts")

    tcols = st.columns([1, 1], gap="large")

    if "C01.00" in get_supported_templates():
        with tcols[0]:
            st.markdown("#### C01.00 Own Funds (extract)")
            df_c01 = build_template_extract_table(structured, "C01.00")
            st.dataframe(df_c01, use_container_width=True, hide_index=True)

    if "C02.00" in get_supported_templates():
        with tcols[1]:
            st.markdown("#### C02.00 Ratios (extract)")
            df_c02 = build_template_extract_table(structured, "C02.00")
            st.dataframe(df_c02, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Traceability: citations per field")

    # Field chooser across both templates
    all_rows = []
    for df in [df_c01 if "df_c01" in locals() else None, df_c02 if "df_c02" in locals() else None]:
        if isinstance(df, pd.DataFrame):
            all_rows.extend(df["field_code"].tolist())

    selected_fc = st.selectbox("Select field_code", options=all_rows, index=0 if all_rows else None)

    if selected_fc:
        cits = extract_cell_citations(structured, selected_fc)
        if not cits:
            st.warning("No citations found for this field (value should be null).")
        else:
            for i, c in enumerate(cits, start=1):
                st.markdown(f"**Citation {i}**")
                st.caption(f"Source: {c.get('source_title')}")
                st.caption(f"URL: {c.get('source_url')}")
                st.caption(f"Paragraph ID: {c.get('paragraph_id')}")
                st.code(c.get("quote", ""))

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Retrieval transparency (top excerpts used for grounding)")
    st.caption("These are the top retrieval hits. The assistant must cite paragraph_id values from this set (or set values to null).")

    if result.retrieved_excerpts:
        # Show compact table of retrieval hits
        df_rag = pd.DataFrame(
            [
                {
                    "score": ex.get("score"),
                    "source_title": ex.get("source_title"),
                    "paragraph_id": ex.get("paragraph_id"),
                    "page": ex.get("page"),
                    "heading_path": ex.get("heading_path"),
                    "text_preview": (ex.get("text") or "")[:220],
                }
                for ex in result.retrieved_excerpts
            ]
        )
        st.dataframe(df_rag, use_container_width=True, hide_index=True)

        with st.expander("Full retrieved excerpts (JSON)"):
            st.code(json.dumps(result.retrieved_excerpts, indent=2, ensure_ascii=False))
    else:
        st.warning("No retrieval excerpts found. Add docs to ./data and click Refresh corpus.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Structured output (JSON)")
    st.caption("This is the core contract between the LLM and the rest of the system.")
    st.code(json.dumps(structured, indent=2, ensure_ascii=False))
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------

st.write("")
st.caption("Prototype: Streamlit + TF-IDF RAG + OpenAI structured JSON output + deterministic validation + exports.")
