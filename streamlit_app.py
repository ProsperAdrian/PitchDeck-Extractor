# streamlit_app.py

import streamlit as st
import pandas as pd
import os
import json
import openai
import gspread
from google.oauth2.service_account import Credentials

from extract_text import extract_text_from_pdf
from analyze import build_few_shot_prompt, call_chatgpt

# Load OpenAI key from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]


# --------------------------------------------------
st.set_page_config(
    page_title="Pitch Deck Extractor",
    layout="wide",
)

st.title("📊 Pitch Deck Extractor")

st.markdown(
    "Upload one or more pitch-deck PDFs. We will extract StartupName, FoundingYear, "
    "Founders, Industry, Niche, USP, FundingStage, CurrentRevenue, Market (TAM/SAM/SOM), "
    "AmountRaised, etc."
)

# Step 1: Let user upload multiple PDFs
uploaded_files = st.file_uploader(
    "Drag & drop PDF(s) here (or click to browse)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    # Initialize a list to hold each deck's JSON
    all_results = []

    # Process each uploaded PDF
    for pdf_file in uploaded_files:
        st.write(f"Processing **{pdf_file.name}**…")

        # 2a) Extract text from PDF by writing to a temp file
        bytes_data = pdf_file.read()
        temp_filename = f"temp_{pdf_file.name}"
        with open(temp_filename, "wb") as f:
            f.write(bytes_data)
        deck_text = extract_text_from_pdf(temp_filename)
        os.remove(temp_filename)

        # 2b) Build the few-shot prompt
        prompt = build_few_shot_prompt(deck_text)

        # 2c) Call ChatGPT
        try:
            result = call_chatgpt(prompt, api_key=openai_api_key)
        except Exception as e:
            st.error(f"Error extracting **{pdf_file.name}**: {e}")
            continue

        # 2d) Attach filename to the JSON
        result["__filename"] = pdf_file.name

        # 2e) Post-processing (any inference logic in analyze.py already applies)
        all_results.append(result)

    # If we got at least one result, show them in a table
    if all_results:
        # 3) Normalize JSON records into a flat table. 
        # Each “row” is one deck. Flatten "Market" into TAM, SAM, SOM columns.
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

        # ---------------------------------------------------------
        # 4) Export to a New Google Sheet (one file per click)
        st.markdown("---")
        st.markdown("## Export to a New Google Sheet (one file per click)")

        if st.button("Create New Google Sheet from Results"):
            try:
                # 4a) Load service-account JSON from Streamlit secrets
                creds_json = st.secrets["google_sheets"]["credentials_json"]
                creds_dict = json.loads(creds_json)

                # 4b) Build Credentials object with appropriate scopes
                scopes = [
                    "https://www.googleapis.com/auth/drive",
                    "https://www.googleapis.com/auth/spreadsheets"
                ]
                creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)

                # 4c) Authorize gspread
                client = gspread.authorize(creds)

                # 4d) Create a new spreadsheet with a timestamped title
                import datetime
                ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
                sheet_title = f"PitchDeck_Extraction_{ts}"
                spreadsheet = client.create(sheet_title)

                # 4e) Share it publicly (anyone with link can view)
                spreadsheet.share(None, perm_type="anyone", role="reader")

                # 4f) Select the first worksheet
                worksheet = spreadsheet.get_worksheet(0)  # default “Sheet1”

                # 4g) Convert df to list of lists (header + rows)
                data = [df.columns.tolist()] + df.values.tolist()

                # 4h) Write data into the sheet starting at A1
                worksheet.update("A1", data)

                # 4i) Show success and the new-sheet URL
                sheet_url = spreadsheet.url
                st.success(f"✅ Created new sheet: **{sheet_title}**")
                st.write(f"[Open the new Google Sheet here →]({sheet_url})")

            except Exception as e:
                st.error(f"⚠️ Failed to create or write to Google Sheets: {e}")
        # ---------------------------------------------------------

        # 5) Offer JSON download as well
        json_str = json.dumps(all_results, indent=2)
        st.download_button(
            label="Download all results as JSON",
            data=json_str,
            file_name="all_decks_parsed.json",
            mime="application/json"
        )
