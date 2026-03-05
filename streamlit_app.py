#!/usr/bin/env python3
"""
Streamlit Frontend for the Pan-India Court Cause List Parser (Regex Only).

Run:
    streamlit run streamlit_app.py
"""

import json
import os
import tempfile

import streamlit as st

from cause_list_parser import run_pipeline

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cause List Parser",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Indian Court Cause List Parser")
st.caption("Upload a cause list PDF from any Indian High Court → get structured JSON.")

# ── Sidebar — settings ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    st.info("⚡ **Regex Mode**: Processes thousands of cases in seconds without an LLM.")
    
    confidence_threshold = 0.3  # Internal default

# ── File upload ──────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Cause List PDF", type=["pdf"])

# ── Process button ───────────────────────────────────────────────────────
if uploaded_file and st.button("🚀 Parse Cause List", type="primary"):
    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Run the pipeline with a spinner
    try:
        with st.spinner("Processing with Regex…"):
            result = run_pipeline(
                tmp_path,
                confidence_threshold=confidence_threshold,
            )
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        st.stop()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ── Display results ──────────────────────────────────────────────────
    cases = result.get("cases", [])
    st.success(f"Done! Extracted **{len(cases)}** cases.")

    # Display basic stats
    if cases:
        st.metric("Total Cases", len(cases))

    # Court metadata
    st.subheader("Court Metadata")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Court", result.get("court_name") or "—")
    col2.metric("Date", result.get("date") or "—")
    col3.metric("Bench", result.get("bench") or "—")
    col4.metric("Court No.", result.get("court_number") or "—")

    # Cases table
    if cases:
        st.subheader(f"Cases ({len(cases)})")
        # Hide confidence column from the dataframe view
        import pandas as pd
        df = pd.DataFrame(cases)
        if "confidence" in df.columns:
            df = df.drop(columns=["confidence"])
        st.dataframe(df, use_container_width=True)

    # JSON download
    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Download JSON",
        data=json_str,
        file_name="cause_list_parsed.json",
        mime="application/json",
    )

    # Raw JSON expander
    with st.expander("View raw JSON"):
        st.json(result)
