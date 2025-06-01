# streamlit_app.py

import streamlit as st
import pandas as pd
import os, json
import openai

from extract_text import extract_text_from_pdf
from analyze import build_few_shot_prompt, call_chatgpt

openai_api_key = st.secrets["openai"]["api_key"]


# --------------------------------------------------
st.set_page_config(
    page_title="Pitch Deck Extractor",
    layout="wide",
)

st.title("📊 Pitch Deck Extractor")

st.markdown("Upload one or more pitch-deck PDFs. We will extract StartupName, FoundingYear, Founders, Industry, Niche, USP, FundingStage, CurrentRevenue, Market (TAM/SAM/SOM), AmountRaised, etc.")

# Step 1: Let user upload multiple PDFs
uploaded_files = st.file_uploader(
    "Drag & drop PDF(s) here (or click to browse)", 
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    # Initialize a list to hold each deck's JSON
    all_results = []

    # Optionally: create a temp folder in memory or on disk
    # We'll just process directly from the file‐buffer, no need to save to disk.
    for pdf_file in uploaded_files:
        st.write(f"Processing **{pdf_file.name}**…")

        # 2a) Extract text from PDF. 
        # extract_text_from_pdf can accept either a filepath or a file‐like buffer.
        # If yours only accepts a filepath, you can write the buffer to a temp file:
        bytes_data = pdf_file.read()
        with open(f"temp_{pdf_file.name}", "wb") as f:
            f.write(bytes_data)
        deck_text = extract_text_from_pdf(f"temp_{pdf_file.name}")
        os.remove(f"temp_{pdf_file.name}")

        # 2b) Build the few‐shot prompt
        prompt = build_few_shot_prompt(deck_text)

        # 2c) Call ChatGPT
        try:
            result = call_chatgpt(prompt, api_key=openai_api_key)
        except Exception as e:
            st.error(f"Error extracting **{pdf_file.name}**: {e}")
            continue

        # 2d) Attach filename to the JSON
        result["__filename"] = pdf_file.name

        # 2e) Post‐processing (if you added any inference logic inside analyze.py, it still applies)
        #    (By default, analyze.py’s own code already does the null→inference pass.)

        all_results.append(result)

    # If we got at least one result, show them in a table
    if all_results:
        # 3) Normalize JSON records into a flat table. 
        #    Each “row” is one deck. Flatten "Market" into three columns: TAM, SAM, SOM.
        rows = []
        for rec in all_results:
            row = {
                "Filename": rec.get("__filename"),
                "StartupName": rec.get("StartupName"),
                "FoundingYear": rec.get("FoundingYear"),
                "Founders": "; ".join(rec.get("Founders") or []),
                "Industry": rec.get("Industry"),
                "Niche": rec.get("Niche"),
                "USP": rec.get("USP"),
                "FundingStage": rec.get("FundingStage"),
                "CurrentRevenue": rec.get("CurrentRevenue"),
                "AmountRaised": rec.get("AmountRaised"),
            }
            market = rec.get("Market") or {}
            row["TAM"] = market.get("TAM")
            row["SAM"] = market.get("SAM")
            row["SOM"] = market.get("SOM")
            rows.append(row)

        df = pd.DataFrame(rows)
        st.markdown("### Extraction Results")
        st.dataframe(df, use_container_width=True)

        # 4) Offer JSON download as well
        json_str = json.dumps(all_results, indent=2)
        st.download_button(
            label="Export as JSON", 
            data=json_str, 
            file_name="All_decks.json",
            mime="application/json"
        )

        # 5) Offer CSV download
        csv_str = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export as CSV", 
            data=csv_str,
            file_name="All_decks.csv",
            mime="text/csv"
        )
